#!/usr/bin/env python3
"""Regenerate the sparse-first Linear PyTorch numerical oracle."""

from __future__ import annotations

import json
from pathlib import Path

import torch


OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "sparse_linear_pytorch.json"
)


def flatten(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().reshape(-1).tolist()


def main() -> None:
    dense = torch.tensor(
        [[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 4.0]],
        dtype=torch.float32,
    )
    weight = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    bias = torch.tensor(
        [0.5, -0.5], dtype=torch.float32, requires_grad=True
    )
    grad_output = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
    )
    output = torch.nn.functional.linear(dense, weight, bias)
    output.backward(grad_output)

    fixture = {
        "schema_version": 1,
        "reference": {
            "framework": "PyTorch",
            "version": torch.__version__,
            "generator": "reference/generate_sparse_linear_fixture.py",
            "operation": "torch.nn.functional.linear and Tensor.backward",
        },
        "case": {
            "name": "csr_first_linear_forward_backward",
            "input_shape": [2, 4],
            "input_dense": flatten(dense),
            "row_offsets": [0, 2, 4],
            "column_indices": [0, 2, 1, 3],
            "values": [1.0, 2.0, 3.0, 4.0],
            "weight_shape": [2, 4],
            "weight": flatten(weight),
            "bias": flatten(bias),
            "grad_output": flatten(grad_output),
            "expected_output": flatten(output),
            "expected_weight_grad": flatten(weight.grad),
            "expected_bias_grad": flatten(bias.grad),
            "atol": 1e-5,
        },
    }
    OUTPUT.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
