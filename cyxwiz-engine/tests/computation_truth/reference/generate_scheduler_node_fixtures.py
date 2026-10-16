#!/usr/bin/env python3
"""PyTorch fixtures for the scheduler nodes (TOFIX140 A3): for each node, its
parameters as the graph stores them and the learning rate torch's scheduler
gives after each epoch (optimizer.step() then scheduler.step(), once per
epoch). The Engine test compiles a graph with the node, trains it with the
TrainingExecutor and compares its learning-rate history with these values.

Reduce LR is not here: it steps on the run's own validation losses, which only
the run knows; the Engine test replays them through the backend scheduler,
which tests/unit/test_scheduler.cpp holds to torch.

    py -3.12 generate_scheduler_node_fixtures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1
LEARNING_RATE = 0.0625  # exact in float32, so the Engine's float config adds no error
EPOCHS = 6
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "fixtures" / "scheduler_node_pytorch.json"
)


def build(node: str, params: dict[str, str], optimizer: torch.optim.Optimizer):
    lr = torch.optim.lr_scheduler
    if node == "StepLR":
        return lr.StepLR(optimizer, step_size=int(params["step_size"]), gamma=float(params["gamma"]))
    if node == "CosineAnnealing":
        return lr.CosineAnnealingLR(optimizer, T_max=int(params["T_max"]), eta_min=float(params["eta_min"]))
    if node == "ExponentialLR":
        return lr.ExponentialLR(optimizer, gamma=float(params["gamma"]))
    if node == "WarmupScheduler":
        return lr.LinearLR(optimizer, start_factor=float(params["start_factor"]), end_factor=1.0,
                           total_iters=int(params["warmup_epochs"]))
    raise ValueError(node)


def case(name: str, node: str, params: dict[str, str]) -> dict[str, Any]:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=LEARNING_RATE)
    scheduler = build(node, params, optimizer)
    first = optimizer.param_groups[0]["lr"]
    history = []
    for _ in range(EPOCHS):
        optimizer.step()
        scheduler.step()
        history.append(optimizer.param_groups[0]["lr"])
    return {"name": name, "node": node, "parameters": params, "first_epoch_learning_rate": first,
            "learning_rate_history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "torch_version": torch.__version__,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "tolerance": 1.0e-9,
        "cases": [
            case("step", "StepLR", {"step_size": "2", "gamma": "0.5"}),
            # T_max below the epoch count: past T_max the cosine rises again.
            case("cosine", "CosineAnnealing", {"T_max": "4", "eta_min": "0.001"}),
            case("exponential", "ExponentialLR", {"gamma": "0.8"}),
            case("warmup", "WarmupScheduler", {"warmup_epochs": "3", "start_factor": "0.25"}),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
