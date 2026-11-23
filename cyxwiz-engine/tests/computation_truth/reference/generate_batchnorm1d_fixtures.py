"""BatchNorm 1D (torch.nn.BatchNorm1d on [batch, features]) against PyTorch
(TOFIX228 1.4: the empty BatchNorm 1D row of the coverage matrix; ORA-03 for
norm layers; ORA-07 running statistics over several batches).

Each case trains on 4 batches in a row (output, input gradient, gamma / beta
gradients and both running statistics after every batch), then evaluates a
fifth batch with the running statistics. Batch sizes 2, 7 and 64; one case
with eps 1e-3 and momentum 0.3; one whose features have a variance near eps
(std 1e-3), where a 10x wrong epsilon moves the output by far more
than the tolerance (the generator checks that it does).

Run with Python 3.12 and torch (CPU):
    py -3.12 generate_batchnorm1d_fixtures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1
SEED = 2281
TRAIN_BATCHES = 4
TOLERANCE = {"atol": 2.0e-5, "rtol": 2.0e-4}
# The variance-near-eps case amplifies float32 rounding about 300x. The Engine
# keeps the batch statistics and their gradients in float64 where the device
# has it (ORA-16, fixed), so the case holds the tight tolerance there; a
# float32-only device (Intel Iris Xe) keeps the design's float32 gradient
# tolerance (07 section 8), recorded as float32_device_tolerance.
NEAR_EPS_TOLERANCE = {"atol": 1.0e-4, "rtol": 1.0e-3}
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "fixtures" / "batchnorm1d_pytorch.json"


def tensor(t: torch.Tensor) -> dict[str, Any]:
    v = t.detach().cpu().contiguous()
    return {"shape": list(v.shape), "values": v.reshape(-1).tolist()}


def run(features: int, batch: int, eps: float, momentum: float, scale: float, offset: float,
        generator: torch.Generator, eps_override: float | None = None,
        inputs: list[torch.Tensor] | None = None,
        tolerance: dict[str, float] = TOLERANCE) -> tuple[dict[str, Any], list[torch.Tensor]]:
    bn = torch.nn.BatchNorm1d(features, eps=eps if eps_override is None else eps_override, momentum=momentum)
    with torch.no_grad():
        bn.weight.copy_(torch.linspace(0.5, 1.5, features))
        bn.bias.copy_(torch.linspace(-0.25, 0.25, features))
    if inputs is None:
        inputs = [offset + scale * torch.randn(batch, features, generator=generator) for _ in range(TRAIN_BATCHES + 1)]
    grad_outputs = [torch.linspace(-1.0, 1.0, batch * features).reshape(batch, features) * (i + 1)
                    for i in range(TRAIN_BATCHES)]
    bn.train()
    steps = []
    for i in range(TRAIN_BATCHES):
        x = inputs[i].clone().requires_grad_(True)
        bn.zero_grad(set_to_none=True)
        y = bn(x)
        y.backward(grad_outputs[i])
        steps.append({
            "input": tensor(x),
            "grad_output": tensor(grad_outputs[i]),
            "output": tensor(y),
            "grad_input": tensor(x.grad),
            "grad_gamma": tensor(bn.weight.grad),
            "grad_beta": tensor(bn.bias.grad),
            "running_mean": tensor(bn.running_mean),
            "running_var": tensor(bn.running_var),
        })
    bn.eval()
    with torch.no_grad():
        evaluated = bn(inputs[TRAIN_BATCHES])
    case = {
        "num_features": features,
        "batch": batch,
        "eps": eps,
        "momentum": momentum,
        "gamma": tensor(bn.weight),
        "beta": tensor(bn.bias),
        "tolerance": tolerance,
        "train": steps,
        "eval": {"input": tensor(inputs[TRAIN_BATCHES]), "output": tensor(evaluated)},
    }
    return case, inputs


def worst_output_excess(a: dict[str, Any], b: dict[str, Any], tolerance: dict[str, float]) -> float:
    worst = 0.0
    for sa, sb in zip(a["train"] + [a["eval"]], b["train"] + [b["eval"]]):
        x = torch.tensor(sa["output"]["values"], dtype=torch.float64)
        y = torch.tensor(sb["output"]["values"], dtype=torch.float64)
        worst = max(worst, float(((x - y).abs() / (tolerance["atol"] + tolerance["rtol"] * x.abs())).max()))
    return worst


def generate() -> dict[str, Any]:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(SEED)
    definitions = {
        "batch2_f3": dict(features=3, batch=2, eps=1e-5, momentum=0.1, scale=2.0, offset=0.5),
        "batch7_f5": dict(features=5, batch=7, eps=1e-5, momentum=0.1, scale=1.0, offset=-1.0),
        "batch64_f4": dict(features=4, batch=64, eps=1e-5, momentum=0.1, scale=3.0, offset=2.0),
        "eps1e-3_momentum0.3_batch7_f3": dict(features=3, batch=7, eps=1e-3, momentum=0.3, scale=1.0, offset=0.0),
        "variance_near_eps_batch7_f4": dict(features=4, batch=7, eps=1e-5, momentum=0.1, scale=1e-3, offset=0.25),
    }
    cases = {}
    for name, d in definitions.items():
        tolerance = TOLERANCE
        case, inputs = run(generator=generator, tolerance=tolerance, **d)
        if name.startswith("variance_near_eps"):
            case["float32_device_tolerance"] = NEAR_EPS_TOLERANCE
            # eps / 10 or eps x 10 must miss the fixture by far: the case catches a wrong epsilon.
            for wrong in (d["eps"] / 10.0, d["eps"] * 10.0):
                other, _ = run(generator=generator, eps_override=wrong, inputs=inputs, tolerance=tolerance, **d)
                excess = worst_output_excess(case, other, tolerance)
                if excess < 10.0:
                    raise SystemExit(f"{name}: eps {wrong} moves the output by only {excess:.2f} tolerances")
                case[f"eps_{wrong:g}_tolerances"] = excess
        cases[name] = case
    return {
        "schema_version": SCHEMA_VERSION,
        "oracle": {"name": "PyTorch", "version": torch.__version__, "device": "cpu",
                   "operation": "torch.nn.BatchNorm1d"},
        "seed": SEED,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fixture = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for name, case in fixture["cases"].items():
        notes = {k: round(v) for k, v in case.items() if k.endswith("_tolerances")}
        print(name, notes or "")
    print(args.output)


if __name__ == "__main__":
    main()
