#!/usr/bin/env python3
"""PyTorch fixtures for the spatial (CNN) layers CyxWiz runs on [H,W,C,N]
(TOFIX140 A1): Conv2d, MaxPool2d, AvgPool2d, ConvTranspose2d, GroupNorm,
InstanceNorm2d, Upsample (nearest, bilinear) and PixelShuffle.

Every case stores the input, the parameters, the forward output, a fixed
upstream gradient and the gradients PyTorch computes for the input and the
parameters. Tensors are written in the backend's layouts: activations as
[H,W,C,N] (row-major), Conv2d weights as [kh,kw,Cin,Cout], ConvTranspose2d
weights as [kh,kw,Cout,Cin], per-channel vectors as [C].

    py -3.12 generate_spatial_layer_fixtures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "fixtures" / "spatial_layers_pytorch.json"
)


def hwcn(tensor: torch.Tensor) -> dict[str, Any]:
    """[N,C,H,W] -> [H,W,C,N], row-major values."""
    value = tensor.detach().permute(2, 3, 1, 0).contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def plain(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def conv_weight(tensor: torch.Tensor) -> dict[str, Any]:
    """torch [Cout,Cin,kh,kw] -> backend [kh,kw,Cin,Cout]."""
    return plain(tensor.permute(2, 3, 1, 0))


def transpose_weight(tensor: torch.Tensor) -> dict[str, Any]:
    """torch [Cin,Cout,kh,kw] -> backend [kh,kw,Cout,Cin]."""
    return plain(tensor.permute(2, 3, 1, 0))


def case(name: str, layer: str, geometry: dict[str, Any], x: torch.Tensor,
         forward, params: dict[str, torch.Tensor], param_writers: dict[str, Any],
         tolerance=(1e-4, 1e-4)) -> dict[str, Any]:
    x = x.clone().requires_grad_(True)
    for p in params.values():
        p.requires_grad_(True)
    y = forward(x)
    torch.manual_seed(hash(name) % (2**31))
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    out: dict[str, Any] = {
        "name": name,
        "layer": layer,
        "geometry": geometry,
        "tolerance": {"atol": tolerance[0], "rtol": tolerance[1]},
        "input": hwcn(x),
        "output": hwcn(y),
        "grad_output": hwcn(grad_out),
        "grad_input": hwcn(x.grad),
        "parameters": {k: param_writers[k](v) for k, v in params.items()},
        "parameter_gradients": {k: param_writers[k](v.grad) for k, v in params.items()},
    }
    return out


def build() -> list[dict[str, Any]]:
    torch.manual_seed(20261008)
    cases = []

    # Conv2d, padding 'same' at stride 1
    x = torch.randn(2, 3, 8, 8)
    w = torch.randn(4, 3, 3, 3) * 0.3
    b = torch.randn(4) * 0.1
    cases.append(case(
        "conv2d_same_k3", "Conv2D", {"filters": 4, "kernel_size": 3, "stride": 1, "padding": 1},
        x, lambda t: functional.conv2d(t, w, b, stride=1, padding=1),
        {"weights": w, "bias": b}, {"weights": conv_weight, "bias": plain}))

    # Conv2d, stride 2, no padding
    x = torch.randn(2, 2, 7, 7)
    w = torch.randn(3, 2, 3, 3) * 0.3
    b = torch.randn(3) * 0.1
    cases.append(case(
        "conv2d_valid_s2", "Conv2D", {"filters": 3, "kernel_size": 3, "stride": 2, "padding": 0},
        x, lambda t: functional.conv2d(t, w, b, stride=2, padding=0),
        {"weights": w, "bias": b}, {"weights": conv_weight, "bias": plain}))

    # MaxPool2d 2/2
    x = torch.randn(2, 3, 8, 8)
    cases.append(case(
        "maxpool2d_2_2", "MaxPool2D", {"pool_size": 2, "stride": 2, "padding": 0},
        x, lambda t: functional.max_pool2d(t, kernel_size=2, stride=2), {}, {}))

    # MaxPool2d 3/2 (overlapping)
    x = torch.randn(1, 2, 7, 7)
    cases.append(case(
        "maxpool2d_3_2", "MaxPool2D", {"pool_size": 3, "stride": 2, "padding": 0},
        x, lambda t: functional.max_pool2d(t, kernel_size=3, stride=2), {}, {}))

    # AvgPool2d 2/2
    x = torch.randn(2, 3, 8, 8)
    cases.append(case(
        "avgpool2d_2_2", "AvgPool2D", {"pool_size": 2, "stride": 2, "padding": 0},
        x, lambda t: functional.avg_pool2d(t, kernel_size=2, stride=2), {}, {}))

    # ConvTranspose2d k3 s2 p1 op1: 4x4 -> 8x8
    x = torch.randn(2, 3, 4, 4)
    w = torch.randn(3, 2, 3, 3) * 0.3  # [Cin, Cout, kh, kw]
    b = torch.randn(2) * 0.1
    cases.append(case(
        "convtranspose2d_k3_s2", "ConvTranspose2D",
        {"out_channels": 2, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
        x, lambda t: functional.conv_transpose2d(t, w, b, stride=2, padding=1, output_padding=1),
        {"weights": w, "bias": b}, {"weights": transpose_weight, "bias": plain}))

    # GroupNorm 2 groups over 4 channels, affine
    x = torch.randn(2, 4, 5, 5)
    gamma = torch.randn(4) * 0.5 + 1.0
    beta = torch.randn(4) * 0.2
    cases.append(case(
        "groupnorm_g2_c4", "GroupNorm", {"num_groups": 2, "eps": 1e-5, "affine": True},
        x, lambda t: functional.group_norm(t, 2, gamma, beta, eps=1e-5),
        {"gamma": gamma, "beta": beta}, {"gamma": plain, "beta": plain}, tolerance=(2e-4, 2e-4)))

    # InstanceNorm2d, no affine
    x = torch.randn(2, 3, 5, 5)
    cases.append(case(
        "instancenorm_c3", "InstanceNorm", {"eps": 1e-5, "affine": False},
        x, lambda t: functional.instance_norm(t, eps=1e-5), {}, {}, tolerance=(2e-4, 2e-4)))

    # Upsample nearest x2
    x = torch.randn(2, 3, 4, 4)
    cases.append(case(
        "upsample_nearest_x2", "Upsample", {"scale_factor": 2, "mode": 0},
        x, lambda t: functional.interpolate(t, scale_factor=2, mode="nearest"), {}, {}))

    # Upsample bilinear x2 (align_corners=False)
    x = torch.randn(2, 3, 4, 4)
    cases.append(case(
        "upsample_bilinear_x2", "Upsample", {"scale_factor": 2, "mode": 1},
        x, lambda t: functional.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False),
        {}, {}))

    # PixelShuffle r=2, 8 channels -> 2
    x = torch.randn(2, 8, 3, 3)
    cases.append(case(
        "pixel_shuffle_r2", "PixelShuffle", {"upscale_factor": 2},
        x, lambda t: functional.pixel_shuffle(t, 2), {}, {}))

    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "generator": "generate_spatial_layer_fixtures.py",
        "torch_version": torch.__version__,
        "layout": {"activations": "[H,W,C,N] row-major", "conv_weights": "[kh,kw,Cin,Cout]",
                   "conv_transpose_weights": "[kh,kw,Cout,Cin]", "channel_vectors": "[C]"},
        "cases": build(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1), encoding="utf-8")
    print(f"wrote {args.output} ({len(fixture['cases'])} cases)")


if __name__ == "__main__":
    main()
