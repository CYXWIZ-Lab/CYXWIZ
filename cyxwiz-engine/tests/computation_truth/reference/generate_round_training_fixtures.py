#!/usr/bin/env python3
"""Fixtures for multi-node training in rounds (TOFIX166 phase 0).

Writes the cases the Engine, node and coordinator must reproduce
(round_training_reference.py is the algorithm):

- int8: a vector, its codes and decode table, the decoded values;
- aggregate: updates with sample counts -> weighted mean and trimmed mean;
- screen: norms -> which updates are kept;
- outer: three Nesterov outer steps (checked here against torch.optim.SGD);
- round: one full round of a small MLP with three workers that take 7, 3
  and 1 steps (the speeds of a fast, a middle and a slow node), Adam inner
  optimizers: the initial weights, each worker's batches, its delta and
  sample count, and theta after the round (float32 and int8 wire).

    py -3.12 generate_round_training_fixtures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import round_training_reference as ref

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "round_training_pytorch.json"
SEED = 166


def plain(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def check_outer_against_torch(theta: torch.Tensor, deltas: list[torch.Tensor]) -> list[torch.Tensor]:
    """Our outer step must equal torch.optim.SGD(nesterov=True) on the same deltas."""
    ours = ref.OuterOptimizer()
    mine = theta.clone()
    param = torch.nn.Parameter(theta.clone())
    sgd = torch.optim.SGD([param], lr=ref.OUTER_LR, momentum=ref.OUTER_MOMENTUM, nesterov=True)
    steps = []
    for delta in deltas:
        mine = ours.step(mine, delta)
        param.grad = delta.clone()
        sgd.step()
        if not torch.allclose(mine, param.detach(), atol=1e-6, rtol=0):
            raise AssertionError("outer step differs from torch.optim.SGD(nesterov=True)")
        steps.append(mine.clone())
    return steps


def int8_case(generator: torch.Generator) -> dict[str, Any]:
    values = torch.randn(64, generator=generator) * 0.01
    values[5] = 0.5    # an outlier, clipped at mean + 6 std
    payload = ref.quantize_int8(values)
    decoded = ref.dequantize_int8(payload)
    return {
        "values": plain(values),
        "codes": [int(c) for c in payload.codes.tolist()],
        "table": plain(payload.table),
        "decoded": plain(decoded),
    }


def aggregate_cases(generator: torch.Generator) -> list[dict[str, Any]]:
    deltas = [torch.randn(8, generator=generator) for _ in range(4)]
    deltas[3] = deltas[3] * 50.0  # one bad worker
    samples = [224, 96, 32, 160]
    cases = []
    for method in ("weighted_mean", "trimmed_mean"):
        cases.append({
            "method": method,
            "trim": 1,
            "deltas": [plain(d) for d in deltas],
            "samples": samples,
            "result": plain(ref.aggregate(deltas, samples, method, 1)),
        })
    cases.append({
        "method": "screen",
        "factor": 3.0,
        "deltas": [plain(d) for d in deltas],
        "kept": ref.screen_by_norm(deltas, 3.0),
    })
    return cases


class Mlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 16)
        self.fc2 = torch.nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def round_case(int8: bool) -> dict[str, Any]:
    # Its own seeds: the float32 and int8 rounds train on the same data.
    generator = torch.Generator().manual_seed(SEED + 1)
    torch.manual_seed(SEED)
    template = Mlp()
    theta = ref.flatten(template.parameters())
    steps = {"fast": 7, "middle": 3, "slow": 1}
    batch = 8
    workers, batches = [], {}
    for name, count in steps.items():
        xs = [torch.randn(batch, 8, generator=generator) for _ in range(count)]
        ys = [torch.randint(0, 4, (batch,), generator=generator) for _ in range(count)]
        batches[name] = list(zip(xs, ys))
        model = Mlp()
        workers.append(ref.Worker(
            name, model, torch.optim.Adam(model.parameters(), lr=0.01),
            (lambda data: (lambda n: iter(data[:n])))(batches[name])))
    reports = [ref.train_round(w, theta, steps[w.name]) for w in workers]
    result = ref.coordinator_round(theta, reports, ref.OuterOptimizer(), int8=int8,
                                   method="weighted_mean")
    return {
        "int8_wire": int8,
        "method": "weighted_mean",
        "inner_optimizer": {"name": "adam", "lr": 0.01, "betas": [0.9, 0.999], "eps": 1e-8},
        "outer": {"lr": ref.OUTER_LR, "momentum": ref.OUTER_MOMENTUM},
        "layers": [{"name": "fc1", "in": 8, "out": 16, "activation": "relu"},
                   {"name": "fc2", "in": 16, "out": 4, "activation": "none"}],
        "parameter_order": "fc1.weight [out,in], fc1.bias, fc2.weight [out,in], fc2.bias",
        "theta": plain(theta),
        "workers": [{
            "name": r.name,
            "steps": r.steps,
            "samples": r.samples,
            "batches": [{"x": plain(x), "y": [int(v) for v in y.tolist()]} for x, y in batches[r.name]],
            "delta": plain(r.delta),
        } for r in reports],
        "theta_next": plain(result.theta),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(SEED)

    theta = torch.randn(6, generator=generator)
    outer_deltas = [torch.randn(6, generator=generator) for _ in range(3)]
    outer_steps = check_outer_against_torch(theta, outer_deltas)

    fixture = {
        "schema_version": SCHEMA_VERSION,
        "generator": "generate_round_training_fixtures.py",
        "torch_version": torch.__version__,
        "tolerance": {"float32": 1e-5, "round": 1e-4},
        "int8": int8_case(generator),
        "aggregate": aggregate_cases(generator),
        "outer": {
            "lr": ref.OUTER_LR, "momentum": ref.OUTER_MOMENTUM,
            "theta": plain(theta),
            "deltas": [plain(d) for d in outer_deltas],
            "thetas": [plain(t) for t in outer_steps],
        },
        "rounds": [round_case(False), round_case(True)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
