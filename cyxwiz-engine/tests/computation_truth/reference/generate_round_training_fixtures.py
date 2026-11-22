#!/usr/bin/env python3
"""Fixtures for multi-node training in rounds (TOFIX166).

Writes the cases the Engine, node and coordinator must reproduce
(round_training_reference.py is the algorithm):

- int8: a vector, its codes and decode table, the decoded values;
- update: an update over a layout of four segments (a large trainable one
  with an outlier, a small one, a statistic, a large one with no spread):
  how each segment travels, its payload, the decoded update, bytes on the wire;
- aggregate: updates with sample counts -> weighted mean and trimmed mean;
- screens: norm per step. A fast honest worker (10 steps against 1 and 1) is
  kept, a x50 update is not, an update that is not finite never is;
- outer: three Nesterov outer steps (checked here against torch.optim.SGD);
- rounds: one full round of a small MLP with three workers that take 7, 3
  and 1 steps (the speeds of a fast, a middle and a slow node), Adam inner
  optimizers: the layout, the initial state, each worker's batches, its delta
  and sample count, and theta after the round. Three of them: float32 wire,
  int8 wire, and a model with BatchNorm (statistics in the state).

    py -3.12 generate_round_training_fixtures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import round_training_reference as ref

SCHEMA_VERSION = 2
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "round_training_pytorch.json"
SEED = 166
HIDDEN = 144   # fc1.weight has 8 x 144 = 1152 values: it travels as int8


def plain(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous()
    # Nine significant digits give a float32 back exactly.
    return {"shape": list(value.shape), "values": [float(f"{v:.9g}") for v in value.reshape(-1).tolist()]}


def plain_layout(layout: list[ref.Segment]) -> list[dict[str, Any]]:
    return [{"name": s.name, "count": s.count, "kind": s.kind,
             "encoding": "int8" if ref.segment_travels_as_int8(s) else "float32"} for s in layout]


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


def update_case(generator: torch.Generator) -> dict[str, Any]:
    layout = [
        ref.Segment("layer0.weight", 1536, ref.TRAINABLE),
        ref.Segment("layer0.bias", 12, ref.TRAINABLE),          # small: float32
        ref.Segment("layer2.running_var", 1200, ref.STATISTIC),  # a statistic: float32 at any size
        ref.Segment("layer4.weight", 1100, ref.TRAINABLE),       # no spread: every code 0
    ]
    weight = torch.randn(1536, generator=generator) * 0.02
    weight[7] = 1.5       # clipped at mean + 6 std
    weight[900] = -2.0    # clipped at mean - 6 std
    bias = torch.randn(12, generator=generator) * 0.1
    variance = torch.rand(1200, generator=generator) * 1e-3
    flat = torch.full((1100,), 0.125)
    delta = torch.cat([weight, bias, variance, flat])
    payloads = ref.encode_update(delta, layout)
    decoded = ref.decode_update(payloads, layout)
    segments = []
    for segment, payload in zip(plain_layout(layout), payloads):
        if isinstance(payload, ref.Int8Payload):
            segment["codes"] = [int(c) for c in payload.codes.tolist()]
            segment["table"] = plain(payload.table)
        segments.append(segment)
    float32_payloads = ref.encode_update(delta, layout, int8=False)
    if not torch.equal(ref.decode_update(float32_payloads, layout), delta):
        raise AssertionError("a float32 update must decode to itself")
    return {
        "int8_min_values": ref.INT8_MIN_VALUES,
        "segments": segments,
        "delta": plain(delta),
        "decoded": plain(decoded),
        "wire_bytes": ref.wire_bytes(payloads),
        "wire_bytes_float32": ref.wire_bytes(float32_payloads),
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
    return cases


def screen_cases(generator: torch.Generator) -> list[dict[str, Any]]:
    def case(name: str, deltas: list[torch.Tensor], steps: list[int], expected: list[bool]) -> dict[str, Any]:
        kept = ref.screen_by_norm(deltas, steps, 3.0)
        if kept != expected:
            raise AssertionError(f"screen case {name!r}: {kept}, expected {expected}")
        return {"name": name, "factor": 3.0, "steps": steps, "kept": kept,
                "deltas": [{"shape": [d.numel()],
                            "values": [v if v == v and abs(v) != float("inf") else None for v in d.tolist()]}
                           for d in deltas]}

    unit = [torch.nn.functional.normalize(torch.randn(8, generator=generator), dim=0) for _ in range(4)]
    not_finite = unit[2].clone()
    not_finite[3] = float("nan")
    return [
        # The norms measured on the reference with speeds 10 / 1 / 1: the fast
        # worker's update is 5.74 times the others'. A raw-norm screen rejects it.
        case("fast honest worker", [unit[0] * 5.74, unit[1], unit[2]], [10, 1, 1], [True, True, True]),
        case("x50 update at equal steps", [unit[0], unit[1], unit[2], unit[3] * 50.0], [4, 4, 4, 4],
             [True, True, True, False]),
        case("claims the steps but is still x50 per step", [unit[0], unit[1], unit[2] * 500.0], [1, 1, 10],
             [True, True, False]),
        case("even count: median of the middle two", [unit[0], unit[1] * 2.0, unit[2] * 4.0, unit[3] * 9.5],
             [1, 1, 1, 1], [True, True, True, False]),
        case("not finite among three", [unit[0], unit[1], not_finite], [2, 2, 2], [True, True, False]),
        case("not finite among two", [unit[0], not_finite], [2, 2], [True, False]),
    ]


class Mlp(torch.nn.Module):
    def __init__(self, batch_norm: bool) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(8, HIDDEN)
        self.bn = torch.nn.BatchNorm1d(HIDDEN) if batch_norm else torch.nn.Identity()
        self.fc2 = torch.nn.Linear(HIDDEN, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.bn(self.fc1(x))))


def round_case(int8: bool, batch_norm: bool) -> dict[str, Any]:
    # Its own seeds: the float32 and int8 rounds train on the same data.
    generator = torch.Generator().manual_seed(SEED + 1)
    torch.manual_seed(SEED)
    template = Mlp(batch_norm)
    layout, tensors = ref.model_state(template)
    theta = ref.flatten(tensors)
    steps = {"fast": 7, "middle": 3, "slow": 1}
    batch = 8
    workers, batches = [], {}
    for name, count in steps.items():
        xs = [torch.randn(batch, 8, generator=generator) for _ in range(count)]
        ys = [torch.randint(0, 4, (batch,), generator=generator) for _ in range(count)]
        batches[name] = list(zip(xs, ys))
        model = Mlp(batch_norm)
        workers.append(ref.Worker(
            name, model, torch.optim.Adam(model.parameters(), lr=0.01),
            (lambda data: (lambda n: iter(data[:n])))(batches[name])))
    reports = [ref.train_round(w, theta, steps[w.name]) for w in workers]
    result = ref.coordinator_round(theta, reports, ref.OuterOptimizer(), int8=int8,
                                   method="weighted_mean", layout=layout)
    if result.rejected:
        raise AssertionError(f"an honest round rejected {result.rejected}")
    layers = [{"name": "fc1", "type": "linear", "in": 8, "out": HIDDEN}]
    if batch_norm:
        layers.append({"name": "bn", "type": "batch_norm_1d", "features": HIDDEN, "eps": 1e-5, "momentum": 0.1})
    layers += [{"name": "relu", "type": "relu"}, {"name": "fc2", "type": "linear", "in": HIDDEN, "out": 4}]
    return {
        "int8_wire": int8,
        "batch_norm": batch_norm,
        "method": "weighted_mean",
        "inner_optimizer": {"name": "adam", "lr": 0.01, "betas": [0.9, 0.999], "eps": 1e-8},
        "outer": {"lr": ref.OUTER_LR, "momentum": ref.OUTER_MOMENTUM},
        "layers": layers,
        "layout": plain_layout(layout),
        "layout_note": "weights are [out, in]; a statistic is averaged by samples and applied without the outer step",
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
        "update": update_case(generator),
        "aggregate": aggregate_cases(generator),
        "screens": screen_cases(generator),
        "outer": {
            "lr": ref.OUTER_LR, "momentum": ref.OUTER_MOMENTUM,
            "theta": plain(theta),
            "deltas": [plain(d) for d in outer_deltas],
            "thetas": [plain(t) for t in outer_steps],
        },
        "rounds": [round_case(False, False), round_case(True, False), round_case(True, True)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
