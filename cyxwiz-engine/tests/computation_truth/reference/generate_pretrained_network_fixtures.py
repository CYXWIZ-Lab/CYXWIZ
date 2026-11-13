#!/usr/bin/env python3
# TOFIX178 1.9f-g
"""Published network architectures from PyTorch, for the Engine's whole-network checks.

For each network a canvas graph is written with scripts/examples/torch_to_cyxgraph.py (nodes named as PyTorch
names its modules), with PyTorch's logits for one input. test_cnn_graph_training starts the graph from the
state-dict and compares.

Small versions, committed (random weights; every BatchNorm given its own statistics):
  resnet18_thin           ResNet-18 an eighth as wide (8 ... 64 channels), 10 classes, built from torchvision's
                          own BasicBlock with torchvision.models.resnet18's names
  mobilenet_v2_thin       torchvision.models.mobilenet_v2(width_mult=0.25, num_classes=10)
  mobilenet_v3_small_thin torchvision.models.mobilenet_v3_small(width_mult=0.5, num_classes=10)
  (the MobileNets with dropout 0, so the training-mode logits do not depend on a random mask)
  -> fixtures/<name>.cyxgraph, fixtures/weight_files/<name>.pth, fixtures/<name>_pytorch.json
     (logits in evaluation and in training mode)

--imagenet DIR, a folder with the published weight files (not committed):
  resnet18-f37072fd.pth, mobilenet_v2-b0353104.pth, mobilenet_v3_small-047dcff4.pth
  -> fixtures/<network>_imagenet.cyxgraph, fixtures/<network>_imagenet_pytorch.json (logits in evaluation)
     which the test compares when CYXWIZ_PRETRAINED_WEIGHTS_DIR names the same folder.

Run: python generate_pretrained_network_fixtures.py [--imagenet DIR]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torchvision
from torch import nn
from torchvision.models.resnet import BasicBlock, conv1x1

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "examples"))
import torch_to_cyxgraph  # noqa: E402


class ThinResNet18(nn.Module):
    """torchvision.models.resnet.ResNet(BasicBlock, [2, 2, 2, 2]) with `base` channels instead of 64."""

    def __init__(self, base: int, num_classes: int) -> None:
        super().__init__()
        self.inplanes = base
        self.conv1 = nn.Conv2d(3, base, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(base, 1)
        self.layer2 = self._make_layer(base * 2, 2)
        self.layer3 = self._make_layer(base * 4, 2)
        self.layer4 = self._make_layer(base * 8, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base * 8, num_classes)

    def _make_layer(self, planes: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride), nn.BatchNorm2d(planes))
        blocks = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        blocks.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def thin_resnet18() -> nn.Module:
    model = ThinResNet18(8, 10)
    assert list(model.state_dict().keys()) == list(torchvision.models.resnet18().state_dict().keys())
    return model


THIN: dict[str, Callable[[], nn.Module]] = {
    "resnet18_thin": thin_resnet18,
    "mobilenet_v2_thin": lambda: torchvision.models.mobilenet_v2(width_mult=0.25, num_classes=10, dropout=0.0),
    "mobilenet_v3_small_thin": lambda: torchvision.models.mobilenet_v3_small(width_mult=0.5, num_classes=10,
                                                                              dropout=0.0),
}
PUBLISHED = {
    "resnet18": (torchvision.models.resnet18, "resnet18-f37072fd.pth"),
    "mobilenet_v2": (torchvision.models.mobilenet_v2, "mobilenet_v2-b0353104.pth"),
    "mobilenet_v3_small": (torchvision.models.mobilenet_v3_small, "mobilenet_v3_small-047dcff4.pth"),
}


def plain(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def hwcn(tensor: torch.Tensor) -> dict[str, Any]:
    """[N, C, H, W] -> the Engine's [H, W, C, N]."""
    return plain(tensor.detach().permute(2, 3, 1, 0))


def layers_with_weights(model: nn.Module) -> int:
    nodes, _ = torch_to_cyxgraph.convert(model)
    return sum(n["type"] in torch_to_cyxgraph.HOLDS_WEIGHTS for n in nodes)


def thin(name: str, build: Callable[[], nn.Module], seed: int) -> None:
    torch.manual_seed(seed)
    model = build()
    # A trained network's BatchNorm is not at its initial values: give each one its own.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.weight.copy_(torch.rand_like(module.weight) * 0.6 + 0.7)
                module.bias.copy_(torch.randn_like(module.bias) * 0.2)
                module.running_mean.copy_(torch.randn_like(module.running_mean) * 0.2)
                module.running_var.copy_(torch.rand_like(module.running_var) + 0.5)
    classes = model.state_dict()[[k for k in model.state_dict() if k.endswith(".weight")][-1]].shape[0]
    torch_to_cyxgraph.network_graph(model, classes).write(
        FIXTURES / f"{name}.cyxgraph", f"{name}: compared with PyTorch in tests (generate_pretrained_network_fixtures.py)")
    torch.save(model.state_dict(), FIXTURES / "weight_files" / f"{name}.pth")
    x = torch.randn(2, 3, 64, 64) * 0.8 + 0.1
    with torch.no_grad():
        evaluated = model.eval()(x)
        trained = copy.deepcopy(model).train()(x)  # batch statistics; the copy keeps the saved running statistics
    fixture = {"generator": "generate_pretrained_network_fixtures.py", "torch_version": torch.__version__,
               "torchvision_version": torchvision.__version__, "tolerance": {"atol": 1e-4, "rtol": 1e-4},
               "layers_with_weights": layers_with_weights(model), "weights_file": f"{name}.pth",
               "input": hwcn(x), "output_eval": plain(evaluated), "output_train": plain(trained)}
    (FIXTURES / f"{name}_pytorch.json").write_text(json.dumps(fixture) + "\n", encoding="utf-8")
    size = (FIXTURES / "weight_files" / f"{name}.pth").stat().st_size
    print(f"{name}: {fixture['layers_with_weights']} layers with weights, {size} bytes of weights")


def imagenet(folder: Path) -> None:
    for name, (build, file_name) in PUBLISHED.items():
        path = folder / file_name
        if not path.is_file():
            print(f"{name}: {path} not found, skipped")
            continue
        model = build()
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        torch_to_cyxgraph.network_graph(model, 1000).write(
            FIXTURES / f"{name}_imagenet.cyxgraph", f"{name} with 1000 classes: checked against {file_name}")
        torch.manual_seed(19180)
        x = torch.randn(2, 3, 64, 64) * 0.8 + 0.1
        with torch.no_grad():
            logits = model(x)
        fixture = {"generator": "generate_pretrained_network_fixtures.py --imagenet", "torch_version": torch.__version__,
                   "torchvision_version": torchvision.__version__, "weights_file": file_name,
                   "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "layers_with_weights": layers_with_weights(model),
                   "tolerance": {"atol": 5e-3, "rtol": 5e-3}, "input": hwcn(x), "output_eval": plain(logits)}
        (FIXTURES / f"{name}_imagenet_pytorch.json").write_text(json.dumps(fixture) + "\n", encoding="utf-8")
        print(f"{name}: top class {int(logits[0].argmax())}, logit {float(logits[0].max()):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imagenet", type=Path, help="a folder with the published weight files")
    args = parser.parse_args()
    for seed, (name, build) in enumerate(THIN.items(), start=1918):
        thin(name, build, seed)
    if args.imagenet:
        imagenet(args.imagenet)


if __name__ == "__main__":
    main()
