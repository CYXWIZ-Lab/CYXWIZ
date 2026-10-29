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
    return [{
        "name": "resnet_downsampling_block",
        "tolerance": {"atol": 3e-4, "rtol": 3e-4},
        "input": hwcn(x),
        "layers": layers,
        "state": before,
        "state_after": {name: {k: plain(v) for k, v in stats.items()} for name, stats in running.items()},
        "output": plain(logits),
        "grad_output": plain(grad_out),
        "grad_input": hwcn(x.grad),
    }]


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
