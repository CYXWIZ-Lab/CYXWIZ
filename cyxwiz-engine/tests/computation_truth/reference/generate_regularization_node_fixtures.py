#!/usr/bin/env python3
"""PyTorch fixtures for the regularization nodes (TOFIX140 A4): a Dense(2 -> 1)
model trained with SGD on MSE plus each node's penalty, from the Engine's own
seeded start (INITIAL below, which the Engine test checks against its
initialisation first). For each node: its parameters as the graph stores them
and torch's parameters after training, where every step is

    loss = mse(model(x), y) + penalty(model.parameters())
    loss.backward(); optimizer.step()

The Engine test compiles a graph with the node, trains it with the
TrainingExecutor and compares the trained parameters with these.

    py -3.12 generate_regularization_node_fixtures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1
LEARNING_RATE = 0.0625
BATCH_SIZE = 4
EPOCHS = 3
X0 = [0.0, 0.1, 0.9, 1.0, 0.2, 0.8, 0.4, 0.6]
X1 = [0.0, 0.2, 0.8, 1.0, 0.1, 0.9, 0.5, 0.3]
LABEL = [0.0, 0.3, 1.7, 2.0, 0.3, 1.7, 0.9, 0.9]
# The Engine's Dense(2 -> 1) initialisation for model_seed 52, as the Engine
# test prints it: weight in the Engine's layout ([in, out]), then bias.
INITIAL = {
    "layer0.weight": [-0.941265523, 0.647087157],
    "layer0.bias": [0.0],
}
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "fixtures" / "regularization_node_pytorch.json"
)


def penalty(node: str | None, params: dict[str, str], model: torch.nn.Module) -> torch.Tensor:
    if node is None:
        return torch.zeros(())
    l1 = sum(p.abs().sum() for p in model.parameters())
    l2 = sum(p.pow(2).sum() for p in model.parameters())
    lam = float(params["lambda"])
    if node == "L1Regularization":
        return lam * l1
    if node == "L2Regularization":
        return lam * l2
    if node == "ElasticNet":
        ratio = float(params["l1_ratio"])
        return lam * (ratio * l1 + (1 - ratio) * l2)
    raise ValueError(node)


def sgd(model: torch.nn.Module, settings: dict[str, str]) -> torch.optim.Optimizer:
    """The SGD node's settings as the graph stores them, as torch.optim.SGD."""
    decay = float(settings.get("weight_decay", "0"))
    options = {
        "lr": LEARNING_RATE,
        "momentum": float(settings.get("momentum", "0")),
        "nesterov": settings.get("nesterov", "false") == "true",
    }
    if settings.get("weight_decay_exclude", "none") == "norms_and_biases":
        # Every 1-D tensor (here the bias) takes no decay.
        groups = [
            {"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": decay},
            {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0},
        ]
        return torch.optim.SGD(groups, **options)
    return torch.optim.SGD(model.parameters(), weight_decay=decay, **options)


def case(name: str, node: str | None, params: dict[str, str],
         optimizer_settings: dict[str, str] | None = None) -> dict[str, Any]:
    model = torch.nn.Linear(2, 1)
    weight_name, bias_name = list(INITIAL)
    with torch.no_grad():
        # Engine [in, out] -> torch [out, in]
        model.weight.copy_(torch.tensor(INITIAL[weight_name]).reshape(2, 1).t())
        model.bias.copy_(torch.tensor(INITIAL[bias_name]))
    optimizer = sgd(model, optimizer_settings or {})
    x = torch.tensor([X0, X1]).t()
    y = torch.tensor(LABEL).reshape(-1, 1)
    for _ in range(EPOCHS):
        for start in range(0, len(LABEL), BATCH_SIZE):
            optimizer.zero_grad()
            out = model(x[start:start + BATCH_SIZE])
            loss = torch.nn.functional.mse_loss(out, y[start:start + BATCH_SIZE])
            loss = loss + penalty(node, params, model)
            loss.backward()
            optimizer.step()
    result: dict[str, Any] = {"name": name}
    if node is not None:
        result["node"] = node
        result["parameters"] = params
    if optimizer_settings:
        result["optimizer"] = optimizer_settings
    result["parameters_after"] = {
        weight_name: model.weight.detach().t().reshape(-1).tolist(),
        bias_name: model.bias.detach().tolist(),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "torch_version": torch.__version__,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "tolerance": 1.0e-6,
        "x0": X0,
        "x1": X1,
        "label": LABEL,
        "initial_parameters": INITIAL,
        "cases": [
            # No penalty: holds the harness itself to torch.
            case("none", None, {}),
            case("l1", "L1Regularization", {"lambda": "0.05"}),
            case("l2", "L2Regularization", {"lambda": "0.05"}),
            case("elastic_net", "ElasticNet", {"lambda": "0.1", "l1_ratio": "0.25"}),
            # The SGD node's own settings (TOFIX178 1.7): weight decay, Nesterov, the exclusion.
            case("sgd_weight_decay", None, {}, {"momentum": "0", "weight_decay": "0.1"}),
            case("sgd_nesterov", None, {}, {"momentum": "0.9", "nesterov": "true"}),
            case("sgd_nesterov_decay_no_bias", None, {},
                 {"momentum": "0.9", "nesterov": "true", "weight_decay": "0.05",
                  "weight_decay_exclude": "norms_and_biases"}),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
