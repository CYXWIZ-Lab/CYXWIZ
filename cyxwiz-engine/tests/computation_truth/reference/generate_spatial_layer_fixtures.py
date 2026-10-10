#!/usr/bin/env python3
"""PyTorch fixtures for the layers CyxWiz brings out of the blocked catalog
(TOFIX140): the spatial (CNN) layers on [H,W,C,N] - Conv2d, MaxPool2d,
AvgPool2d, ConvTranspose2d, GroupNorm, InstanceNorm2d, Upsample (nearest,
bilinear), PixelShuffle, global average pooling ([H,W,C,N] -> [N,C] rows) -
Conv1d on [L,C,N] sequences, global max pooling, adaptive average pooling,
depthwise Conv2d (groups = C) - and the activations PReLU and SELU on [N, F] rows.

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
import zlib
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


def lcn(tensor: torch.Tensor) -> dict[str, Any]:
    """[N,C,L] -> [L,C,N], row-major values (the backend's Conv1D layout)."""
    value = tensor.detach().permute(2, 1, 0).contiguous()
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
         tolerance=(1e-4, 1e-4), rows: bool = False, rows_out: bool = False,
         sequence: bool = False) -> dict[str, Any]:
    """rows=True: x is [N, F] rows, written as is (no [H,W,C,N] reorder).
    rows_out=True: the output (and its gradient) is [N, F] rows.
    sequence=True: x and the output are [N,C,L] sequences, written [L,C,N]."""
    layout = plain if rows else lcn if sequence else hwcn
    out_layout = plain if rows or rows_out else lcn if sequence else hwcn
    x = x.clone().requires_grad_(True)
    for p in params.values():
        p.requires_grad_(True)
    y = forward(x)
    # crc32, not hash(): str hashes are salted per process, so hash() made
    # every regeneration rewrite every case's upstream gradient.
    torch.manual_seed(zlib.crc32(name.encode()))
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    out: dict[str, Any] = {
        "name": name,
        "layer": layer,
        "geometry": geometry,
        "tolerance": {"atol": tolerance[0], "rtol": tolerance[1]},
        "input": layout(x),
        "output": out_layout(y),
        "grad_output": out_layout(grad_out),
        "grad_input": layout(x.grad),
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

    # Conv2d at an image size (the ArrayFire CUDA path failed here; TOFIX140 A1b)
    x = torch.randn(2, 3, 32, 32)
    w = torch.randn(8, 3, 3, 3) * 0.3
    b = torch.randn(8) * 0.1
    cases.append(case(
        "conv2d_image_32", "Conv2D", {"filters": 8, "kernel_size": 3, "stride": 1, "padding": 1},
        x, lambda t: functional.conv2d(t, w, b, stride=1, padding=1),
        {"weights": w, "bias": b}, {"weights": conv_weight, "bias": plain}, tolerance=(2e-4, 2e-4)))

    # Conv2d with padding >= kernel (ArrayFire unwrap cannot; the provider can)
    x = torch.randn(1, 2, 5, 5)
    w = torch.randn(3, 2, 3, 3) * 0.3
    b = torch.randn(3) * 0.1
    cases.append(case(
        "conv2d_k3_p3", "Conv2D", {"filters": 3, "kernel_size": 3, "stride": 2, "padding": 3},
        x, lambda t: functional.conv2d(t, w, b, stride=2, padding=3),
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

    # Image-sized, padded, overlapping pools (CUDA kernels can fail where small ones pass)
    x = torch.randn(2, 16, 32, 32)
    cases.append(case(
        "maxpool2d_image_32_k3_p1", "MaxPool2D", {"pool_size": 3, "stride": 2, "padding": 1},
        x, lambda t: functional.max_pool2d(t, kernel_size=3, stride=2, padding=1), {}, {}))
    x = torch.randn(2, 8, 32, 32)
    cases.append(case(
        "avgpool2d_image_32_k3_p1", "AvgPool2D", {"pool_size": 3, "stride": 2, "padding": 1},
        x, lambda t: functional.avg_pool2d(t, kernel_size=3, stride=2, padding=1), {}, {}))

    # ConvTranspose2d k3 s2 p1 op1: 4x4 -> 8x8
    x = torch.randn(2, 3, 4, 4)
    w = torch.randn(3, 2, 3, 3) * 0.3  # [Cin, Cout, kh, kw]
    b = torch.randn(2) * 0.1
    cases.append(case(
        "convtranspose2d_k3_s2", "ConvTranspose2D",
        {"out_channels": 2, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
        x, lambda t: functional.conv_transpose2d(t, w, b, stride=2, padding=1, output_padding=1),
        {"weights": w, "bias": b}, {"weights": transpose_weight, "bias": plain}))
    # padding >= kernel (ArrayFire's wrap refuses it) and an image-sized decoder step
    for name, cin, cout, k, s_, p_, op, size in [("convtranspose2d_pad_over_k2", 2, 3, 2, 3, 2, 1, 5),
                                                 ("convtranspose2d_image_16", 8, 4, 4, 2, 1, 0, 16)]:
        x = torch.randn(2, cin, size, size)
        w = torch.randn(cin, cout, k, k) * 0.2
        b = torch.randn(cout) * 0.1
        cases.append(case(
            name, "ConvTranspose2D",
            {"out_channels": cout, "kernel_size": k, "stride": s_, "padding": p_, "output_padding": op},
            x, lambda t, w=w, b=b, s_=s_, p_=p_, op=op: functional.conv_transpose2d(
                t, w, b, stride=s_, padding=p_, output_padding=op),
            {"weights": w, "bias": b}, {"weights": transpose_weight, "bias": plain}, tolerance=(2e-4, 2e-4)))

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

    # Activations on [N, F] rows (TOFIX140 A2)
    x = torch.randn(4, 6)
    a = torch.tensor([0.25])
    cases.append(case(
        "prelu_shared", "PReLU", {"num_parameters": 1, "init": 0.25},
        x, lambda t: functional.prelu(t, a), {"alpha": a}, {"alpha": plain}, rows=True))
    x = torch.randn(4, 6)
    a = torch.rand(6) * 0.5
    cases.append(case(
        "prelu_per_feature", "PReLU", {"num_parameters": 6, "init": 0.25},
        x, lambda t: functional.prelu(t, a), {"alpha": a}, {"alpha": plain}, rows=True))
    x = torch.randn(4, 6) * 2.0
    cases.append(case(
        "selu", "SELU", {}, x, lambda t: functional.selu(t), {}, {}, rows=True))

    # PixelShuffle r=2, 8 channels -> 2
    x = torch.randn(2, 8, 3, 3)
    cases.append(case(
        "pixel_shuffle_r2", "PixelShuffle", {"upscale_factor": 2},
        x, lambda t: functional.pixel_shuffle(t, 2), {}, {}))
    # Image-sized decoder steps (reorder-heavy paths; CUDA kernels can fail at size)
    x = torch.randn(2, 16, 16, 16)
    cases.append(case(
        "pixel_shuffle_image_16_r2", "PixelShuffle", {"upscale_factor": 2},
        x, lambda t: functional.pixel_shuffle(t, 2), {}, {}))
    x = torch.randn(2, 8, 16, 16)
    cases.append(case(
        "upsample_bilinear_image_16_x2", "Upsample", {"scale_factor": 2, "mode": 1},
        x, lambda t: functional.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False),
        {}, {}))
    x = torch.randn(2, 8, 16, 16)
    cases.append(case(
        "upsample_nearest_image_16_x3", "Upsample", {"scale_factor": 3, "mode": 0},
        x, lambda t: functional.interpolate(t, scale_factor=3, mode="nearest"), {}, {}))

    # Global average pooling ends the spatial section: [N,C,H,W] -> [N,C] rows
    # (TOFIX140 A4b), torch adaptive_avg_pool2d(x, 1).flatten(1)
    x = torch.randn(2, 3, 5, 5)
    cases.append(case(
        "global_avg_pool_c3", "GlobalAvgPool", {},
        x, lambda t: functional.adaptive_avg_pool2d(t, 1).flatten(1), {}, {}, rows_out=True))
    x = torch.randn(4, 16, 16, 16)
    cases.append(case(
        "global_avg_pool_image_16", "GlobalAvgPool", {},
        x, lambda t: functional.adaptive_avg_pool2d(t, 1).flatten(1), {}, {}, rows_out=True))

    # Conv1d on [L,C,N] sequences (TOFIX140): weights [Cout, Cin, k] as torch
    x = torch.randn(2, 3, 10)
    w = torch.randn(4, 3, 3) * 0.3
    b = torch.randn(4) * 0.1
    cases.append(case(
        "conv1d_same_k3", "Conv1D", {"filters": 4, "kernel_size": 3, "stride": 1, "padding": 1},
        x, lambda t: functional.conv1d(t, w, b, stride=1, padding=1),
        {"weights": w, "bias": b}, {"weights": plain, "bias": plain}, sequence=True))
    x = torch.randn(3, 2, 17)
    w = torch.randn(5, 2, 5) * 0.3
    b = torch.randn(5) * 0.1
    cases.append(case(
        "conv1d_valid_k5_s2", "Conv1D", {"filters": 5, "kernel_size": 5, "stride": 2, "padding": 0},
        x, lambda t: functional.conv1d(t, w, b, stride=2, padding=0),
        {"weights": w, "bias": b}, {"weights": plain, "bias": plain}, sequence=True))
    x = torch.randn(4, 16, 64)  # a text-CNN size: 16 embedding channels, 64 tokens
    w = torch.randn(8, 16, 3) * 0.2
    b = torch.randn(8) * 0.1
    cases.append(case(
        "conv1d_text_64", "Conv1D", {"filters": 8, "kernel_size": 3, "stride": 1, "padding": 1},
        x, lambda t: functional.conv1d(t, w, b, stride=1, padding=1),
        {"weights": w, "bias": b}, {"weights": plain, "bias": plain}, tolerance=(2e-4, 2e-4),
        sequence=True))
    # Dilation and padding >= kernel (torch allows both; ArrayFire's unwrap does neither)
    x = torch.randn(2, 3, 11)
    w = torch.randn(4, 3, 3) * 0.3
    b = torch.randn(4) * 0.1
    cases.append(case(
        "conv1d_dilated_pad_over_k", "Conv1D",
        {"filters": 4, "kernel_size": 3, "stride": 2, "padding": 4, "dilation": 2},
        x, lambda t: functional.conv1d(t, w, b, stride=2, padding=4, dilation=2),
        {"weights": w, "bias": b}, {"weights": plain, "bias": plain}, sequence=True))

    # Global max pooling: [N,C,H,W] -> [N,C] rows; the gradient goes to the maximum
    x = torch.randn(2, 3, 5, 5)
    cases.append(case(
        "global_max_pool_c3", "GlobalMaxPool", {},
        x, lambda t: functional.adaptive_max_pool2d(t, 1).flatten(1), {}, {}, rows_out=True))
    x = torch.randn(4, 16, 16, 16)
    cases.append(case(
        "global_max_pool_image_16", "GlobalMaxPool", {},
        x, lambda t: functional.adaptive_max_pool2d(t, 1).flatten(1), {}, {}, rows_out=True))

    # Adaptive average pooling: overlapping bins (5 -> 2), uneven bins (7x6 -> 3x4),
    # more outputs than inputs (3 -> 4), and the global case (-> 1)
    for name, size, out in [("adaptive_avg_pool_5_to_2", (5, 5), (2, 2)),
                            ("adaptive_avg_pool_7x6_to_3x4", (7, 6), (3, 4)),
                            ("adaptive_avg_pool_3_to_4", (3, 3), (4, 4)),
                            ("adaptive_avg_pool_8_to_1", (8, 8), (1, 1))]:
        x = torch.randn(2, 3, *size)
        cases.append(case(
            name, "AdaptiveAvgPool", {"output_h": out[0], "output_w": out[1]},
            x, lambda t, out=out: functional.adaptive_avg_pool2d(t, out), {}, {}))

    # Depthwise Conv2d: groups = C, weights [C*M, 1, k, k] -> backend [k, k, 1, C*M]
    for name, c, m, k, s, p, size in [("depthwise_same_k3", 4, 1, 3, 1, 1, 8),
                                      ("depthwise_m2_k3_s2", 3, 2, 3, 2, 0, 9),
                                      ("depthwise_image_32", 16, 1, 3, 1, 1, 32),
                                      # padding >= kernel (torch allows it; ArrayFire's unwrap does not)
                                      ("depthwise_pad_over_k2", 2, 2, 2, 1, 3, 6)]:
        x = torch.randn(2, c, size, size)
        w = torch.randn(c * m, 1, k, k) * 0.4
        b = torch.randn(c * m) * 0.1
        cases.append(case(
            name, "DepthwiseConv2D",
            {"depth_multiplier": m, "kernel_size": k, "stride": s, "padding": p},
            x, lambda t, w=w, b=b, s=s, p=p, c=c: functional.conv2d(t, w, b, stride=s, padding=p, groups=c),
            {"weights": w, "bias": b}, {"weights": conv_weight, "bias": plain}, tolerance=(2e-4, 2e-4)))

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
