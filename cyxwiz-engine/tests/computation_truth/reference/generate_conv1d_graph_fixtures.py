#!/usr/bin/env python3
"""PyTorch fixtures for Conv1D graphs (TOFIX140): the whole model the Engine
builds from a compiled graph, against torch, for both ways a sequence enters:

  rows:      x [N, F] -> x.view(N, 1, F) -> Conv1d -> ReLU -> flatten -> Linear
  embedding: ids [N, L] -> Embedding -> transpose(1, 2) -> Conv1d -> ReLU
             -> mean over L (global average pool) -> Linear

Each case stores the input, every parameter (torch layouts: Conv1d weight
[Cout, Cin, k], Linear weight [out, in], Embedding weight [num, dim]), the
output, a fixed upstream gradient and torch's parameter gradients. The Engine
test sets the parameters on the built model, runs forward and backward and
compares.

    py -3.12 generate_conv1d_graph_fixtures.py
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
    Path(__file__).resolve().parent.parent / "fixtures" / "conv1d_graph_pytorch.json"
)


def plain(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def finish(name: str, x: torch.Tensor, params: dict[str, torch.Tensor], forward) -> dict[str, Any]:
    for p in params.values():
        p.requires_grad_(True)
    y = forward()
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    return {
        "name": name,
        "input": plain(x),
        "parameters": {k: plain(v) for k, v in params.items()},
        "output": plain(y),
        "grad_output": plain(grad_out),
        "parameter_gradients": {k: plain(v.grad) for k, v in params.items()},
    }


def rows_case() -> dict[str, Any]:
    n, f = 3, 10
    x = torch.randn(n, f)
    params = {
        "conv.weights": torch.randn(4, 1, 3) * 0.4,
        "conv.bias": torch.randn(4) * 0.1,
        "dense.weight": torch.randn(3, 40) * 0.2,
        "dense.bias": torch.randn(3) * 0.1,
    }

    def forward() -> torch.Tensor:
        h = functional.conv1d(x.view(n, 1, f), params["conv.weights"], params["conv.bias"], padding=1)
        h = torch.flatten(functional.relu(h), 1)
        return functional.linear(h, params["dense.weight"], params["dense.bias"])

    case = finish("rows_flatten", x, params, forward)
    case["graph"] = {"features": f, "filters": 4, "kernel_size": 3, "padding": "same", "units": 3}
    return case


def embedding_case() -> dict[str, Any]:
    n, length, vocab, dim = 3, 8, 12, 5
    ids = torch.randint(0, vocab, (n, length), dtype=torch.int32)
    params = {
        "embedding.weight": torch.randn(vocab, dim) * 0.5,
        "conv.weights": torch.randn(6, dim, 3) * 0.3,
        "conv.bias": torch.randn(6) * 0.1,
        "dense.weight": torch.randn(2, 6) * 0.3,
        "dense.bias": torch.randn(2) * 0.1,
    }

    def forward() -> torch.Tensor:
        e = functional.embedding(ids.long(), params["embedding.weight"])  # [N, L, E]
        h = functional.conv1d(e.transpose(1, 2), params["conv.weights"], params["conv.bias"])
        h = functional.relu(h).mean(dim=2)  # adaptive_avg_pool1d(h, 1).flatten(1)
        return functional.linear(h, params["dense.weight"], params["dense.bias"])

    case = finish("embedding_global_avg_pool", ids, params, forward)
    case["graph"] = {"length": length, "vocab": vocab, "embedding_dim": dim, "filters": 6,
                     "kernel_size": 3, "padding": "valid", "units": 2}
    return case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    torch.manual_seed(20261008)
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "torch_version": torch.__version__,
        "tolerance": {"atol": 1e-5, "rtol": 1e-4},
        "cases": [rows_case(), embedding_case()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
