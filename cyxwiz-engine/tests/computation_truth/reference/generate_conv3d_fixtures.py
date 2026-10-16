#!/usr/bin/env python3
"""PyTorch fixtures for the Conv3D node (TOFIX140 Group C): a volume sample
[C, D, H, W] (the Engine's [D, H, W, C] shape, rows channel by channel, torch
x.view(N, C, D, H, W)) goes through one or two torch.nn.Conv3d layers (a ReLU
between them), then Flatten and a Linear head.

Each case stores the input rows, every convolution's parameters in the
Engine's layout (weight [F, C*k*k*k] = torch weight.flatten(1), bias [F]), the
Linear head, a fixed upstream gradient, the forward output and every gradient.

    py -3.12 generate_conv3d_fixtures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "conv3d_pytorch.json"


def plain(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def case(name: str, sample: tuple[int, int, int, int], batch: int, convs: list[dict[str, Any]],
         relu_between: bool) -> dict[str, Any]:
    c, d, h, w = sample
    layers = []
    channels = c
    for spec in convs:
        k = spec["kernel_size"]
        padding = (k - 1) // 2 if spec["padding"] == "same" else 0
        layers.append(torch.nn.Conv3d(channels, spec["filters"], k, stride=spec["stride"], padding=padding))
        channels = spec["filters"]

    x = torch.randn(batch, c, d, h, w, requires_grad=True)
    y = x
    for index, layer in enumerate(layers):
        if index > 0 and relu_between:
            y = torch.relu(y)
        y = layer(y)
    features = y.flatten(1)
    head = torch.nn.Linear(features.shape[1], 2)
    output = head(features)
    grad = torch.randn(output.shape)
    output.backward(grad)

    return {
        "name": name,
        "sample": [d, h, w, c],
        "relu_between": relu_between,
        "convs": [
            {**spec,
             "weight": plain(layer.weight.flatten(1)),
             "bias": plain(layer.bias),
             "grad_weight": plain(layer.weight.grad.flatten(1)),
             "grad_bias": plain(layer.bias.grad)}
            for spec, layer in zip(convs, layers)
        ],
        "output_sample": [y.shape[2], y.shape[3], y.shape[4], y.shape[1]],
        "input": plain(x.reshape(batch, -1)),
        "output": plain(output),
        "grad_output": plain(grad),
        "grad_input": plain(x.grad.reshape(batch, -1)),
        "head": {"weight": plain(head.weight), "bias": plain(head.bias),
                 "grad_weight": plain(head.weight.grad), "grad_bias": plain(head.bias.grad)},
        "tolerance": {"absolute": 1e-5, "relative": 1e-4},
    }


def build() -> list[dict[str, Any]]:
    torch.manual_seed(140)
    return [
        case("one_channel_same", (1, 4, 5, 6), 2,
             [{"filters": 3, "kernel_size": 3, "stride": 1, "padding": "same"}], False),
        case("two_channels_valid_stride2", (2, 5, 5, 5), 2,
             [{"filters": 4, "kernel_size": 3, "stride": 2, "padding": "valid"}], False),
        case("stacked_with_relu", (1, 6, 6, 6), 2,
             [{"filters": 2, "kernel_size": 3, "stride": 1, "padding": "same"},
              {"filters": 3, "kernel_size": 3, "stride": 2, "padding": "valid"}], True),
        case("pointwise_three_channels", (3, 2, 3, 4), 3,
             [{"filters": 2, "kernel_size": 1, "stride": 1, "padding": "valid"}], False),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generator": "generate_conv3d_fixtures.py",
        "torch_version": torch.__version__,
        "cases": build(),
    }
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()
