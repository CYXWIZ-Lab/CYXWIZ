#!/usr/bin/env python3
"""PyTorch fixture for a residual CNN graph (TOFIX178 1.2): a stem
(Conv2d 3->8, BatchNorm2d, ReLU), one ResNet downsampling block (main path
Conv2d 8->16 stride 2, BatchNorm2d, ReLU, Conv2d 16->16, BatchNorm2d; shortcut
Conv2d 1x1 8->16 stride 2, BatchNorm2d; Add; ReLU), global average pooling and
a Linear 16->2 head. BatchNorm runs in training mode.

The case stores the input ([H,W,C,N]), every layer's parameters in the
backend's layouts (Conv2d weights [kh,kw,Cin,Cout], Linear [out,in], per-channel
vectors [C]), BatchNorm's running statistics before and after the forward, the
logits, a fixed upstream gradient, and the gradients PyTorch computes for the
input and every parameter. Layer names are the Engine graph's node names.

    py -3.12 generate_cnn_graph_fixtures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "cnn_graph_pytorch.json"


def plain(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def hwcn(tensor: torch.Tensor) -> dict[str, Any]:
    """[N,C,H,W] -> [H,W,C,N], row-major values."""
    return plain(tensor.detach().permute(2, 3, 1, 0))


def conv_weight(tensor: torch.Tensor) -> dict[str, Any]:
    """torch [Cout,Cin,kh,kw] -> backend [kh,kw,Cin,Cout]."""
    return plain(tensor.detach().permute(2, 3, 1, 0))


def build() -> list[dict[str, Any]]:
    torch.manual_seed(178)
    convs = {
        # name: (in, out, kernel, stride, padding)
        "Stem conv": (3, 8, 3, 1, 1),
        "Main conv": (8, 16, 3, 2, 1),
        "Main conv 2": (16, 16, 3, 1, 1),
        "Shortcut conv": (8, 16, 1, 2, 0),
    }
    bns = {"Stem BN": 8, "Main BN": 16, "Main BN 2": 16, "Shortcut BN": 16}
    params: dict[str, dict[str, torch.Tensor]] = {}
    for name, (cin, cout, k, _, _) in convs.items():
        params[name] = {"weights": torch.randn(cout, cin, k, k) * (2.0 / (cin * k * k)) ** 0.5,
                        "bias": torch.randn(cout) * 0.1}
    running: dict[str, dict[str, torch.Tensor]] = {}
    for name, channels in bns.items():
        params[name] = {"gamma": torch.randn(channels) * 0.3 + 1.0, "beta": torch.randn(channels) * 0.1}
        running[name] = {"running_mean": torch.randn(channels) * 0.2, "running_var": torch.rand(channels) + 0.5}
    params["Head"] = {"weight": torch.randn(2, 16) * 0.25, "bias": torch.randn(2) * 0.1}
    for layer in params.values():
        for tensor in layer.values():
            tensor.requires_grad_(True)
    before = {name: {k: plain(v) for k, v in stats.items()} for name, stats in running.items()}

    def conv(name: str, t: torch.Tensor) -> torch.Tensor:
        _, _, _, stride, padding = convs[name]
        return functional.conv2d(t, params[name]["weights"], params[name]["bias"], stride=stride, padding=padding)

    def bn(name: str, t: torch.Tensor) -> torch.Tensor:
        return functional.batch_norm(t, running[name]["running_mean"], running[name]["running_var"],
                                     params[name]["gamma"], params[name]["beta"], training=True,
                                     momentum=0.1, eps=1e-5)

    x = (torch.randn(4, 3, 8, 8) * 0.8 + 0.2).requires_grad_(True)
    stem = functional.relu(bn("Stem BN", conv("Stem conv", x)))
    main = functional.relu(bn("Main BN", conv("Main conv", stem)))
    main = bn("Main BN 2", conv("Main conv 2", main))
    shortcut = bn("Shortcut BN", conv("Shortcut conv", stem))
    block = functional.relu(main + shortcut)
    pooled = block.mean(dim=(2, 3))
    logits = functional.linear(pooled, params["Head"]["weight"], params["Head"]["bias"])
    torch.manual_seed(1780)
    grad_out = torch.randn_like(logits)
    logits.backward(grad_out)

    layers: dict[str, Any] = {}
    for name, layer in params.items():
        writer = conv_weight if name in convs else plain
        entry: dict[str, Any] = {}
        for key, tensor in layer.items():
            entry[key] = writer(tensor) if key == "weights" else plain(tensor)
            entry["grad_" + key] = writer(tensor.grad) if key == "weights" else plain(tensor.grad)
        layers[name] = entry
    residual = {
        "name": "resnet_downsampling_block",
        "tolerance": {"atol": 3e-4, "rtol": 3e-4},
        "input": hwcn(x),
        "layers": layers,
        "state": before,
        "state_after": {name: {k: plain(v) for k, v in stats.items()} for name, stats in running.items()},
        "output": plain(logits),
        "grad_output": plain(grad_out),
        "grad_input": hwcn(x.grad),
    }
    return [residual, squeeze_excitation(),
            squeeze_excitation("squeeze_excitation_hard", 1791, 3, functional.hardswish, functional.hardsigmoid, False),
            resnet_stem(), flatten_head(),
            frozen_stem("frozen_stem_keep", 1912, True), frozen_stem("frozen_stem_update", 1913, False)]


def frozen_stem(case_name: str, seed: int, keep_statistics: bool) -> dict[str, Any]:
    """Fine-tuning's first run: the resnet_stem graph with its convolution and BatchNorm frozen
    (requires_grad off) and the Head training. keep_statistics = the BatchNorm also runs on its running
    statistics (module.eval(), the Engine's default for a frozen BatchNorm); otherwise it normalises with
    the batch and its running statistics move, as PyTorch does when only requires_grad is off. The frozen
    layers carry no grad_ entries: the Engine must report none for them."""
    torch.manual_seed(seed)
    conv_weight_ = torch.randn(8, 3, 7, 7) * (2.0 / 147) ** 0.5
    gamma = torch.randn(8) * 0.3 + 4.0
    beta = torch.randn(8) * 0.5 + 2.0
    head = {"weight": (torch.randn(2, 8) * 0.25).requires_grad_(True), "bias": (torch.randn(2) * 0.1).requires_grad_(True)}
    running = {"Stem BN": {"running_mean": torch.randn(8) * 0.2, "running_var": torch.rand(8) + 0.5}}
    before = {name: {k: plain(v) for k, v in stats.items()} for name, stats in running.items()}
    x = (torch.randn(4, 3, 16, 16) * 0.8 + 0.2).requires_grad_(True)
    stem = functional.conv2d(x, conv_weight_, None, stride=2, padding=3)
    stem = functional.relu6(functional.batch_norm(stem, running["Stem BN"]["running_mean"],
                                                  running["Stem BN"]["running_var"], gamma, beta,
                                                  training=not keep_statistics, momentum=0.1, eps=1e-5))
    pooled = functional.max_pool2d(stem, kernel_size=3, stride=2, padding=1)
    logits = functional.linear(pooled.mean(dim=(2, 3)), head["weight"], head["bias"])
    torch.manual_seed(seed * 10)
    grad_out = torch.randn_like(logits)
    logits.backward(grad_out)
    layers: dict[str, Any] = {
        "Stem conv": {"weights": conv_weight(conv_weight_)},
        "Stem BN": {"gamma": plain(gamma), "beta": plain(beta)},
        "Head": {"weight": plain(head["weight"]), "grad_weight": plain(head["weight"].grad),
                 "bias": plain(head["bias"]), "grad_bias": plain(head["bias"].grad)},
    }
    return {
        "name": case_name,
        "tolerance": {"atol": 3e-4, "rtol": 3e-4},
        "input": hwcn(x),
        "layers": layers,
        "state": before,
        "state_after": {name: {k: plain(v) for k, v in stats.items()} for name, stats in running.items()},
        "output": plain(logits),
        "grad_output": plain(grad_out),
        "grad_input": hwcn(x.grad),
    }


def flatten_head() -> dict[str, Any]:
    """Conv 3->4 (3x3, padding 1) -> ReLU -> Flatten -> Head 144->2, as VGG and LeNet end. PyTorch flattens
    [C, H, W] channels first and the Engine [H, W, C] channels last, so the Head's weight is written here with
    its columns in the Engine's order; flatten_head.pth keeps PyTorch's order, which the Engine's weights
    file loader has to reorder."""
    torch.manual_seed(1911)
    channels, size = 4, 6
    params: dict[str, dict[str, torch.Tensor]] = {
        "Stem conv": {"weights": torch.randn(channels, 3, 3, 3) * (2.0 / 27) ** 0.5, "bias": torch.randn(channels) * 0.1},
        "Head": {"weight": torch.randn(2, channels * size * size) * 0.1, "bias": torch.randn(2) * 0.1},
    }
    for layer in params.values():
        for tensor in layer.values():
            tensor.requires_grad_(True)
    torch.save({"Stem conv.weight": params["Stem conv"]["weights"].detach().clone(),
                "Stem conv.bias": params["Stem conv"]["bias"].detach().clone(),
                "Head.weight": params["Head"]["weight"].detach().clone(),
                "Head.bias": params["Head"]["bias"].detach().clone()},
               DEFAULT_OUTPUT.parent / "weight_files" / "flatten_head.pth")
    x = (torch.randn(4, 3, size, size) * 0.8 + 0.2).requires_grad_(True)
    features = functional.relu(functional.conv2d(x, params["Stem conv"]["weights"], params["Stem conv"]["bias"],
                                                 stride=1, padding=1))
    logits = functional.linear(features.flatten(1), params["Head"]["weight"], params["Head"]["bias"])
    torch.manual_seed(19110)
    grad_out = torch.randn_like(logits)
    logits.backward(grad_out)

    def channels_last(weight: torch.Tensor) -> torch.Tensor:
        return weight.reshape(2, channels, size, size).permute(0, 2, 3, 1).reshape(2, -1)

    head = params["Head"]
    layers: dict[str, Any] = {
        "Stem conv": {"weights": conv_weight(params["Stem conv"]["weights"]),
                      "grad_weights": conv_weight(params["Stem conv"]["weights"].grad),
                      "bias": plain(params["Stem conv"]["bias"]), "grad_bias": plain(params["Stem conv"]["bias"].grad)},
        "Head": {"weight": plain(channels_last(head["weight"])), "grad_weight": plain(channels_last(head["weight"].grad)),
                 "bias": plain(head["bias"]), "grad_bias": plain(head["bias"].grad)},
    }
    return {
        "name": "flatten_head",
        "tolerance": {"atol": 3e-4, "rtol": 3e-4},
        "input": hwcn(x),
        "layers": layers,
        "state": {},
        "state_after": {},
        "output": plain(logits),
        "grad_output": plain(grad_out),
        "grad_input": hwcn(x.grad),
    }


def resnet_stem() -> dict[str, Any]:
    """ResNet's opening, with ReLU6 in place of ReLU so that node is checked too: Conv 3->8, 7x7, stride 2,
    padding 3, no bias -> BatchNorm -> ReLU6 -> MaxPool 3, stride 2, padding 1 -> global average pool ->
    Head 8->2. The convolution has no bias entry: the node's Bias setting is off."""
    torch.manual_seed(1909)
    params: dict[str, dict[str, torch.Tensor]] = {
        "Stem conv": {"weights": torch.randn(8, 3, 7, 7) * (2.0 / 147) ** 0.5},
        "Stem BN": {"gamma": torch.randn(8) * 0.3 + 4.0, "beta": torch.randn(8) * 0.5 + 2.0},
        "Head": {"weight": torch.randn(2, 8) * 0.25, "bias": torch.randn(2) * 0.1},
    }
    running = {"Stem BN": {"running_mean": torch.randn(8) * 0.2, "running_var": torch.rand(8) + 0.5}}
    for layer in params.values():
        for tensor in layer.values():
            tensor.requires_grad_(True)
    before = {name: {k: plain(v) for k, v in stats.items()} for name, stats in running.items()}
    # The same weights as a PyTorch state-dict named after the graph's nodes, for the Engine's
    # "Start from weights file": saved before the forward pass moves the running statistics. It also
    # holds what a real file has and the graph does not take: BatchNorm's counter and another layer.
    state = {"Stem conv.weight": params["Stem conv"]["weights"].detach().clone(),
             "Stem BN.weight": params["Stem BN"]["gamma"].detach().clone(),
             "Stem BN.bias": params["Stem BN"]["beta"].detach().clone(),
             "Stem BN.running_mean": running["Stem BN"]["running_mean"].clone(),
             "Stem BN.running_var": running["Stem BN"]["running_var"].clone(),
             "Stem BN.num_batches_tracked": torch.tensor(7),
             "Head.weight": params["Head"]["weight"].detach().clone(),
             "Head.bias": params["Head"]["bias"].detach().clone(),
             "aux.weight": torch.zeros(3, 2)}
    torch.save(state, DEFAULT_OUTPUT.parent / "weight_files" / "resnet_stem.pth")
    x = (torch.randn(4, 3, 16, 16) * 0.8 + 0.2).requires_grad_(True)
    stem = functional.conv2d(x, params["Stem conv"]["weights"], None, stride=2, padding=3)
    stem = functional.relu6(functional.batch_norm(stem, running["Stem BN"]["running_mean"],
                                                  running["Stem BN"]["running_var"], params["Stem BN"]["gamma"],
                                                  params["Stem BN"]["beta"], training=True, momentum=0.1, eps=1e-5))
    capped = float((stem == 6.0).float().mean())
    floored = float((stem == 0.0).float().mean())
    assert capped > 0.02 and floored > 0.02, (capped, floored)  # both ends of ReLU6 are exercised
    pooled = functional.max_pool2d(stem, kernel_size=3, stride=2, padding=1)
    logits = functional.linear(pooled.mean(dim=(2, 3)), params["Head"]["weight"], params["Head"]["bias"])
    torch.manual_seed(19090)
    grad_out = torch.randn_like(logits)
    logits.backward(grad_out)
    layers: dict[str, Any] = {}
    for name, layer in params.items():
        entry: dict[str, Any] = {}
        for key, tensor in layer.items():
            writer = conv_weight if key == "weights" else plain
            entry[key] = writer(tensor)
            entry["grad_" + key] = writer(tensor.grad)
        layers[name] = entry
    return {
        "name": "resnet_stem",
        "tolerance": {"atol": 3e-4, "rtol": 3e-4},
        "input": hwcn(x),
        "layers": layers,
        "state": before,
        "state_after": {name: {k: plain(v) for k, v in stats.items()} for name, stats in running.items()},
        "output": plain(logits),
        "grad_output": plain(grad_out),
        "grad_input": hwcn(x.grad),
    }


def squeeze_excitation(case_name: str = "squeeze_excitation_block", seed: int = 1787, squeeze: int = 4,
                       inner=functional.relu, gate_fn=torch.sigmoid, bias: bool = True) -> dict[str, Any]:
    """The Squeeze-Excitation node after a stem (Conv 3->8, BatchNorm, ReLU): each channel's mean ->
    Linear 8->4 -> ReLU -> Linear 4->8 -> Sigmoid scales the stem's feature map (torchvision.ops.
    SqueezeExcitation's arithmetic, its 1x1 convolutions written as Linear); global average pool; Head 8->2.
    The node's parameters are reduce_weight / reduce_bias / expand_weight / expand_bias ([out, in]).
    The second case sets the node's other choices: a typed hidden width, Hardswish inside, a Hardsigmoid
    gate and no biases (MobileNetV3's block uses the Hardsigmoid gate)."""
    torch.manual_seed(seed)
    params: dict[str, dict[str, torch.Tensor]] = {
        "Stem conv": {"weights": torch.randn(8, 3, 3, 3) * (2.0 / 27) ** 0.5, "bias": torch.randn(8) * 0.1},
        "Stem BN": {"gamma": torch.randn(8) * 0.3 + 1.0, "beta": torch.randn(8) * 0.1},
    }
    se: dict[str, torch.Tensor] = {"reduce_weight": torch.randn(squeeze, 8) * 0.4}
    if bias:
        se["reduce_bias"] = torch.randn(squeeze) * 0.1
    se["expand_weight"] = torch.randn(8, squeeze) * 0.4
    if bias:
        se["expand_bias"] = torch.randn(8) * 0.1
    params["SE"] = se
    params["Head"] = {"weight": torch.randn(2, 8) * 0.25, "bias": torch.randn(2) * 0.1}
    running = {"Stem BN": {"running_mean": torch.randn(8) * 0.2, "running_var": torch.rand(8) + 0.5}}
    for layer in params.values():
        for tensor in layer.values():
            tensor.requires_grad_(True)
    before = {name: {k: plain(v) for k, v in stats.items()} for name, stats in running.items()}
    x = (torch.randn(4, 3, 8, 8) * 0.8 + 0.2).requires_grad_(True)
    stem = functional.conv2d(x, params["Stem conv"]["weights"], params["Stem conv"]["bias"], stride=1, padding=1)
    stem = functional.relu(functional.batch_norm(stem, running["Stem BN"]["running_mean"],
                                                 running["Stem BN"]["running_var"], params["Stem BN"]["gamma"],
                                                 params["Stem BN"]["beta"], training=True, momentum=0.1, eps=1e-5))
    squeezed = stem.mean(dim=(2, 3))
    reduced = inner(functional.linear(squeezed, se["reduce_weight"], se.get("reduce_bias")))
    gate = gate_fn(functional.linear(reduced, se["expand_weight"], se.get("expand_bias")))
    scaled = stem * gate[:, :, None, None]
    logits = functional.linear(scaled.mean(dim=(2, 3)), params["Head"]["weight"], params["Head"]["bias"])
    torch.manual_seed(seed * 10)
    grad_out = torch.randn_like(logits)
    logits.backward(grad_out)
    layers: dict[str, Any] = {}
    for name, layer in params.items():
        entry: dict[str, Any] = {}
        for key, tensor in layer.items():
            writer = conv_weight if key == "weights" else plain
            entry[key] = writer(tensor)
            entry["grad_" + key] = writer(tensor.grad)
        layers[name] = entry
    return {
        "name": case_name,
        "tolerance": {"atol": 3e-4, "rtol": 3e-4},
        "input": hwcn(x),
        "layers": layers,
        "state": before,
        "state_after": {name: {k: plain(v) for k, v in stats.items()} for name, stats in running.items()},
        "output": plain(logits),
        "grad_output": plain(grad_out),
        "grad_input": hwcn(x.grad),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "generator": "generate_cnn_graph_fixtures.py",
        "torch_version": torch.__version__,
        "layout": {"activations": "[H,W,C,N] row-major", "conv_weights": "[kh,kw,Cin,Cout]",
                   "linear_weights": "[out,in]", "channel_vectors": "[C]"},
        "cases": build(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1), encoding="utf-8")
    print(f"wrote {args.output} ({len(fixture['cases'])} cases)")


if __name__ == "__main__":
    main()
