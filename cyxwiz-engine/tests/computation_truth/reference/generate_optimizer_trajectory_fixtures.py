"""Long optimizer trajectories in which epsilon matters (TOFIX228 1.4, ORA-03).

The older optimizer fixtures (training_core_pytorch.json) run 3 steps on a
3-element parameter with gradients of 0.1 to 1: there a 1000x wrong epsilon,
or epsilon inside the square root, stays under the tolerance. These cases run
50 steps on a rank-2 parameter with gradients from 1e-8 to 1e-4, and with
exact zeros (whole steps, a column that never gets a gradient), in float32
and float64. Every case with an epsilon also records how far the run moves
when epsilon is 10x; the generator refuses to write a case that a 10x
epsilon would not move by at least 10 tolerances (the test checks the same
on the Engine).

Run with Python 3.12 and torch (CPU):
    py -3.12 generate_optimizer_trajectory_fixtures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch

SCHEMA_VERSION = 1
SEED = 228
STEPS = 50
SHAPE = (4, 5)
CHECKPOINTS = [1, 5, 10, 25, 50]
RESUME_AFTER = 25
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "optimizer_trajectory_pytorch.json"
)

TOLERANCE = {
    "float32": {"atol": 2.0e-6, "rtol": 2.0e-5},
    "float64": {"atol": 1.0e-12, "rtol": 1.0e-9},
}
DTYPES = {"float32": torch.float32, "float64": torch.float64}


def values(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().contiguous().reshape(-1).tolist()


def gradient_stream(regime: str, dtype: torch.dtype, generator: torch.Generator) -> list[torch.Tensor]:
    """The gradients of one regime, drawn in float64 and rounded to dtype."""
    stream = []
    for step in range(STEPS):
        sign = torch.where(torch.rand(SHAPE, generator=generator, dtype=torch.float64) < 0.5, -1.0, 1.0)
        if regime == "tiny":
            # |g| from 1e-8 to 1e-4: sqrt(v) is near epsilon.
            exponent = -8.0 + 4.0 * torch.rand(SHAPE, generator=generator, dtype=torch.float64)
            gradient = sign * torch.pow(10.0, exponent)
        elif regime == "zeros":
            # |g| from 1e-6 to 1: 40 % zeros each step, column 0 never gets a
            # gradient, steps 10-14 get none at all.
            exponent = -6.0 + 6.0 * torch.rand(SHAPE, generator=generator, dtype=torch.float64)
            gradient = sign * torch.pow(10.0, exponent)
            keep = torch.rand(SHAPE, generator=generator, dtype=torch.float64) >= 0.4
            gradient = gradient * keep
            gradient[:, 0] = 0.0
            if 10 <= step < 15:
                gradient = torch.zeros(SHAPE, dtype=torch.float64)
        else:
            raise ValueError(regime)
        stream.append(gradient.to(dtype))
    return stream


# name -> (torch factory, hyperparameters, epsilon key or None)
OPTIMIZERS: dict[str, tuple[Callable[..., torch.optim.Optimizer], dict[str, float], str | None]] = {
    "sgd_momentum": (
        lambda p, h: torch.optim.SGD(p, lr=h["learning_rate"], momentum=h["momentum"], foreach=False),
        {"learning_rate": 0.1, "momentum": 0.9},
        None,
    ),
    "sgd_nesterov_decay": (
        lambda p, h: torch.optim.SGD(p, lr=h["learning_rate"], momentum=h["momentum"],
                                     weight_decay=h["weight_decay"], nesterov=True, foreach=False),
        {"learning_rate": 0.1, "momentum": 0.9, "weight_decay": 0.01},
        None,
    ),
    "adam": (
        lambda p, h: torch.optim.Adam(p, lr=h["learning_rate"], betas=(h["beta1"], h["beta2"]),
                                      eps=h["epsilon"], foreach=False),
        {"learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1.0e-8},
        "epsilon",
    ),
    "adamw": (
        lambda p, h: torch.optim.AdamW(p, lr=h["learning_rate"], betas=(h["beta1"], h["beta2"]),
                                       eps=h["epsilon"], weight_decay=h["weight_decay"], foreach=False),
        {"learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1.0e-8, "weight_decay": 0.01},
        "epsilon",
    ),
    "nadam": (
        lambda p, h: torch.optim.NAdam(p, lr=h["learning_rate"], betas=(h["beta1"], h["beta2"]),
                                       eps=h["epsilon"], momentum_decay=h["momentum_decay"], foreach=False),
        {"learning_rate": 0.002, "beta1": 0.9, "beta2": 0.999, "epsilon": 1.0e-8, "momentum_decay": 0.004},
        "epsilon",
    ),
    "rmsprop": (
        lambda p, h: torch.optim.RMSprop(p, lr=h["learning_rate"], alpha=h["alpha"], eps=h["epsilon"],
                                         momentum=h["momentum"], foreach=False),
        {"learning_rate": 0.001, "alpha": 0.99, "epsilon": 1.0e-8, "momentum": 0.9},
        "epsilon",
    ),
    "adagrad": (
        lambda p, h: torch.optim.Adagrad(p, lr=h["learning_rate"], eps=h["epsilon"], foreach=False),
        {"learning_rate": 0.01, "epsilon": 1.0e-10},
        "epsilon",
    ),
    "adadelta": (
        lambda p, h: torch.optim.Adadelta(p, lr=h["learning_rate"], rho=h["rho"], eps=h["epsilon"],
                                          foreach=False),
        {"learning_rate": 1.0, "rho": 0.9, "epsilon": 1.0e-6},
        "epsilon",
    ),
}


# The regime in which each optimizer's epsilon must show. "tiny" (|g| 1e-8 to
# 1e-4, sqrt(v) near epsilon) for all but Adadelta: its step is
# sqrt(acc_delta + eps) / sqrt(v + eps) * g, and with gradients far below
# sqrt(eps) both roots are sqrt(eps), so epsilon cancels; in "zeros"
# (gradients up to 1, acc_delta starting from 0) it does not. In the other
# regime the case is a plain 50-step trajectory ("zeros": gradients up to 1
# hide epsilon for the Adam family, as they should).
EPSILON_REGIME = {"adadelta": "zeros"}


def run(name: str, hyperparameters: dict[str, float], initial: torch.Tensor,
        gradients: list[torch.Tensor]) -> list[torch.Tensor]:
    """The parameter after every step."""
    factory = OPTIMIZERS[name][0]
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = factory([parameter], hyperparameters)
    after = []
    for gradient in gradients:
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = gradient.clone()
        optimizer.step()
        after.append(parameter.detach().clone())
    return after


def worst_excess(reference: list[torch.Tensor], other: list[torch.Tensor], tolerance: dict[str, float]) -> float:
    """max over steps and elements of |a - b| / (atol + rtol |a|)."""
    worst = 0.0
    for a, b in zip(reference, other):
        a64, b64 = a.double(), b.double()
        excess = (a64 - b64).abs() / (tolerance["atol"] + tolerance["rtol"] * a64.abs())
        worst = max(worst, float(excess.max()))
    return worst


def generate() -> dict[str, Any]:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(SEED)
    initial64 = torch.randn(SHAPE, generator=generator, dtype=torch.float64)

    streams: dict[str, Any] = {}
    cases: dict[str, Any] = {}
    for regime in ("tiny", "zeros"):
        draws = gradient_stream(regime, torch.float64, generator)
        for dtype_name, dtype in DTYPES.items():
            stream_name = f"{regime}_{dtype_name}"
            gradients = [g.to(dtype) for g in draws]
            initial = initial64.to(dtype)
            streams[stream_name] = {
                "dtype": dtype_name,
                "shape": list(SHAPE),
                "initial_parameter": values(initial),
                "gradients": [values(g) for g in gradients],
            }
            tolerance = TOLERANCE[dtype_name]
            for name, (_, hyperparameters, epsilon_key) in OPTIMIZERS.items():
                after = run(name, hyperparameters, initial, gradients)
                case = {
                    "optimizer": name,
                    "operation": f"torch.optim.{OPTIMIZERS[name][0]([torch.nn.Parameter(initial.clone())], hyperparameters).__class__.__name__}",
                    "stream": stream_name,
                    "dtype": dtype_name,
                    "hyperparameters": hyperparameters,
                    "tolerance": tolerance,
                    "resume_after": RESUME_AFTER,
                    "expected": {str(step): values(after[step - 1]) for step in CHECKPOINTS},
                }
                if epsilon_key is not None and EPSILON_REGIME.get(name, "tiny") == regime:
                    wrong = dict(hyperparameters)
                    wrong[epsilon_key] = hyperparameters[epsilon_key] * 10.0
                    excess = worst_excess(after, run(name, wrong, initial, gradients), tolerance)
                    if excess < 10.0:
                        raise SystemExit(
                            f"{name} / {stream_name}: a 10x epsilon moves the run by only "
                            f"{excess:.2f} tolerances; the case cannot catch a wrong epsilon")
                    case["epsilon_x10_tolerances"] = excess
                cases[f"{name}_{stream_name}"] = case
    return {
        "schema_version": SCHEMA_VERSION,
        "oracle": {"name": "PyTorch", "version": torch.__version__, "device": "cpu"},
        "seed": SEED,
        "steps": STEPS,
        "checkpoints": CHECKPOINTS,
        "streams": streams,
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
        note = case.get("epsilon_x10_tolerances")
        print(f"{name}: epsilon x10 = {note:.0f} tolerances" if note is not None else f"{name}: no epsilon")
    print(args.output)


if __name__ == "__main__":
    main()
