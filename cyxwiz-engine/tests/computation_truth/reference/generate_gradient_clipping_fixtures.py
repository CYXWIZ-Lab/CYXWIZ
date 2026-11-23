#!/usr/bin/env python3
"""Generate PyTorch fixtures for gradient clipping by the global norm.

The reference is torch.nn.utils.clip_grad_norm_ with its default
(error_if_nonfinite=False), and torch.optim.Adam stepping on the clipped
gradients, as PPO and the optimizer nodes' "Gradient clip norm" use it:

- a clip that bites, a clip that does not, a NaN norm and an infinite norm;
- three Adam steps with a clip that bites and with one that does not (the
  parameters and Adam's two moments after each), and one step with a NaN
  and with an infinite norm.

Non-finite numbers are written as the strings "nan", "inf" and "-inf" (JSON
has no NaN).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "gradient_clipping_pytorch.json"
SHAPES = {"a": [3, 4], "b": [5]}


def number(x: float) -> Any:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def tensor_fixture(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous().reshape(-1)
    return {"shape": list(tensor.shape), "values": [number(float(v)) for v in value.tolist()]}


def gradients(seed: int, scale: float = 1.0, poison: float | None = None) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    grads = {name: torch.randn(shape, generator=generator) * scale for name, shape in SHAPES.items()}
    if poison is not None:
        grads["a"].view(-1)[5] = poison
    return grads


def clip_case(name: str, max_norm: float, grads: dict[str, torch.Tensor]) -> dict[str, Any]:
    params = {k: torch.zeros_like(g, requires_grad=True) for k, g in grads.items()}
    for k, p in params.items():
        p.grad = grads[k].clone()
    norm = torch.nn.utils.clip_grad_norm_(list(params.values()), max_norm)
    return {
        "name": name,
        "max_norm": max_norm,
        "gradients": {k: tensor_fixture(g) for k, g in grads.items()},
        "norm": number(float(norm)),
        "clipped": {k: tensor_fixture(p.grad) for k, p in params.items()},
    }


def adam_case(name: str, steps: list[dict[str, torch.Tensor]], max_norm: float) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(7)
    init = {k: torch.randn(shape, generator=generator) * 0.5 for k, shape in SHAPES.items()}
    params = {k: v.clone().requires_grad_(True) for k, v in init.items()}
    optimizer = torch.optim.Adam(list(params.values()), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    after = []
    for grads in steps:
        optimizer.zero_grad()
        for k, p in params.items():
            p.grad = grads[k].clone()
        torch.nn.utils.clip_grad_norm_(list(params.values()), max_norm)
        optimizer.step()
        # The moments scale with the clip (the parameters barely do: Adam's step is nearly
        # scale-free), so they show a clip coefficient that is off
        after.append({
            "parameters": {k: tensor_fixture(p) for k, p in params.items()},
            "first_moment": {k: tensor_fixture(optimizer.state[p]["exp_avg"]) for k, p in params.items()},
            "second_moment": {k: tensor_fixture(optimizer.state[p]["exp_avg_sq"]) for k, p in params.items()},
        })
    return {
        "name": name,
        "max_norm": max_norm,
        "lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
        "initial": {k: tensor_fixture(v) for k, v in init.items()},
        "steps": [{"gradients": {k: tensor_fixture(g) for k, g in grads.items()}, **state}
                  for grads, state in zip(steps, after)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    output = parser.parse_args().output
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "generator": "cyxwiz-engine/tests/computation_truth/reference/generate_gradient_clipping_fixtures.py",
        "torch_version": torch.__version__,
        "reference": "torch.nn.utils.clip_grad_norm_ (error_if_nonfinite=False) and torch.optim.Adam",
        "clip": [
            clip_case("clip_bites", 0.5, gradients(1)),
            clip_case("clip_idle", 1000.0, gradients(1)),
            clip_case("nan_norm", 0.5, gradients(2, poison=float("nan"))),
            clip_case("inf_norm", 0.5, gradients(3, poison=float("inf"))),
        ],
        "adam": [
            adam_case("three_clipped_steps", [gradients(10), gradients(11), gradients(12)], 0.5),
            adam_case("three_idle_clip_steps", [gradients(10), gradients(11), gradients(12)], 1000.0),
            adam_case("nan_norm_step", [gradients(13, poison=float("nan"))], 0.5),
            adam_case("inf_norm_step", [gradients(14, poison=float("inf"))], 0.5),
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
