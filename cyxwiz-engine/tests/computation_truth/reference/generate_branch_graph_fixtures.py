#!/usr/bin/env python3
"""PyTorch fixtures for branched graphs (TOFIX140 A2): Split fans a tensor out
into two branches that merge again. Each case stores the input, every Linear
layer's weight [out, in] and bias, a fixed upstream gradient, the forward
output and the gradients PyTorch computes for the input and every layer.

Split follows the node's contract: Output 1 = the first split_size entries
along dim, Output 2 = the rest, i.e. torch.split(x, [s, n - s], dim).

    py -3.12 generate_branch_graph_fixtures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "fixtures" / "branch_graph_pytorch.json"
)


def plain(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def split(x: torch.Tensor, size: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    first, second = torch.split(x, [size, x.shape[dim] - size], dim)
    return first, second


def case(name: str, split_size: int, dim: int, layers: dict[str, torch.nn.Linear],
         x: torch.Tensor, forward, grad_shape: tuple[int, ...]) -> dict[str, Any]:
    x = x.clone().requires_grad_(True)
    output = forward(x)
    grad = torch.randn(grad_shape)
    output.backward(grad)
    return {
        "name": name,
        "split_size": split_size,
        "dim": dim,
        "input": plain(x),
        "output": plain(output),
        "grad_output": plain(grad),
        "grad_input": plain(x.grad),
        "layers": {
            layer_name: {
                "weight": plain(layer.weight),
                "bias": plain(layer.bias),
                "grad_weight": plain(layer.weight.grad),
                "grad_bias": plain(layer.bias.grad),
            }
            for layer_name, layer in layers.items()
        },
        "tolerance": {"absolute": 1e-5, "relative": 1e-4},
    }


def build() -> list[dict[str, Any]]:
    torch.manual_seed(140)
    cases = []

    # Split 2 | 4 -> Dense A (3), Dense B (3) -> Concatenate -> Dense C (2).
    a, b, c = torch.nn.Linear(2, 3), torch.nn.Linear(4, 3), torch.nn.Linear(6, 2)

    def split_concat(x):
        first, second = split(x, 2, 1)
        return c(torch.cat([a(first), b(second)], 1))

    cases.append(case("split_concat", 2, 1, {"Dense A": a, "Dense B": b, "Dense C": c},
                      torch.randn(4, 6), split_concat, (4, 2)))

    # Split 2 | 4 with Output 2 unused: its gradient is zero.
    d = torch.nn.Linear(2, 3)
    cases.append(case("split_one_branch", 2, 1, {"Dense A": d},
                      torch.randn(4, 6), lambda x: d(split(x, 2, 1)[0]), (4, 3)))

    # Split 3 | 3 on dim -1 -> Dense A (2), Dense B (2) -> Add.
    e, f = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)

    def split_add(x):
        first, second = split(x, 3, -1)
        return e(first) + f(second)

    cases.append(case("split_add_last_dim", 3, -1, {"Dense A": e, "Dense B": f},
                      torch.randn(4, 6), split_add, (4, 2)))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generator": "generate_branch_graph_fixtures.py",
        "torch_version": torch.__version__,
        "cases": build(),
    }
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()
