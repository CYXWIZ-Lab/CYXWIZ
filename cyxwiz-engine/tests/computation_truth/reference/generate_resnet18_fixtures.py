#!/usr/bin/env python3
# TOFIX178 1.9f
"""ResNet-18 from PyTorch for the Engine's whole-network checks.

resnet18_thin: torchvision's ResNet-18 an eighth as wide (8, 16, 32, 64 channels) with 10 classes. It is built
from torchvision's own BasicBlock in torchvision's layout, and its state-dict has exactly the names of
torchvision.models.resnet18 (asserted below). Small enough to commit:
    fixtures/weight_files/resnet18_thin.pth   the state-dict, as torch.save writes it
    fixtures/resnet18_pytorch.json            an input and the logits in eval and in training mode

--imagenet PATH: the published weights (resnet18-f37072fd.pth, not committed: 45 MB). Writes
    fixtures/resnet18_imagenet_pytorch.json   an input and torchvision.models.resnet18's logits for it
which test_cnn_graph_training compares when CYXWIZ_RESNET18_WEIGHTS points at the same file.

Run: python generate_resnet18_fixtures.py [--imagenet path/to/resnet18-f37072fd.pth]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch import nn
from torchvision.models.resnet import BasicBlock, conv1x1

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


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


def plain(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def hwcn(tensor: torch.Tensor) -> dict[str, Any]:
    """[N, C, H, W] -> the Engine's [H, W, C, N]."""
    return plain(tensor.detach().permute(2, 3, 1, 0))


def thin() -> None:
    torch.manual_seed(1918)
    model = ThinResNet18(8, 10)
    reference = torchvision.models.resnet18()
    assert list(model.state_dict().keys()) == list(reference.state_dict().keys()), "names differ from torchvision"
    # A trained network's BatchNorm is not at its initial values: give each one its own.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.weight.copy_(torch.rand_like(module.weight) * 0.6 + 0.7)
                module.bias.copy_(torch.randn_like(module.bias) * 0.2)
                module.running_mean.copy_(torch.randn_like(module.running_mean) * 0.2)
                module.running_var.copy_(torch.rand_like(module.running_var) + 0.5)
    torch.save(model.state_dict(), FIXTURES / "weight_files" / "resnet18_thin.pth")
    x = torch.randn(2, 3, 64, 64) * 0.8 + 0.1
    with torch.no_grad():
        evaluated = model.eval()(x)
        trained = copy.deepcopy(model).train()(x)  # batch statistics; the copy keeps the saved running statistics
    fixture = {"generator": "generate_resnet18_fixtures.py", "torch_version": torch.__version__,
               "torchvision_version": torchvision.__version__, "tolerance": {"atol": 2e-3, "rtol": 2e-3},
               "layers_with_weights": sum(isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)) for m in model.modules()),
               "input": hwcn(x), "output_eval": plain(evaluated), "output_train": plain(trained)}
    (FIXTURES / "resnet18_pytorch.json").write_text(json.dumps(fixture) + "\n", encoding="utf-8")
    size = (FIXTURES / "weight_files" / "resnet18_thin.pth").stat().st_size
    print(f"wrote resnet18_thin.pth ({size} bytes) and resnet18_pytorch.json; eval logits {evaluated[0, :3].tolist()}")


def imagenet(path: Path) -> None:
    model = torchvision.models.resnet18()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    torch.manual_seed(19180)
    x = torch.randn(2, 3, 64, 64) * 0.8 + 0.1
    with torch.no_grad():
        logits = model(x)
    fixture = {"generator": "generate_resnet18_fixtures.py --imagenet", "torch_version": torch.__version__,
               "torchvision_version": torchvision.__version__, "weights_file": path.name,
               "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "tolerance": {"atol": 5e-3, "rtol": 5e-3}, "input": hwcn(x), "output_eval": plain(logits)}
    (FIXTURES / "resnet18_imagenet_pytorch.json").write_text(json.dumps(fixture) + "\n", encoding="utf-8")
    print(f"wrote resnet18_imagenet_pytorch.json; top class {int(logits[0].argmax())}, logit {float(logits[0].max()):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imagenet", type=Path, help="the published resnet18 weight file")
    args = parser.parse_args()
    thin()
    if args.imagenet:
        imagenet(args.imagenet)


if __name__ == "__main__":
    main()
