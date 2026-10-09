#!/usr/bin/env python3
"""PyTorch fixtures for triplet metric learning (TOFIX140 A5): Data Input ->
Triplet Dataset Builder -> Dense(3 -> 4) -> ReLU -> Dense(4 -> 2) -> Triplet
Loss -> SGD, from the Engine's own seeded start (INITIAL below, which the
Engine test checks against its initialisation first).

Every step is what the Engine does: the Triplet Dataset Builder picks, for every
row of the batch, a positive of the same class and a negative of another class
(SplitMix64 keyed by the epoch's DataLoader seed, the batch index and the row:
select_triplets below mirrors metric_learning_sampling.cpp), the encoder runs
once over the stacked [anchors; positives; negatives], and

    loss = triplet_margin_loss(a, p, n, margin)   # p = 2, mean over triplets
    loss.backward(); optimizer.step()

    py -3.12 generate_triplet_graph_fixtures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1
LEARNING_RATE = 0.1
BATCH_SIZE = 6
EPOCHS = 3
DATALOADER_SEED = 42
MASK = (1 << 64) - 1
# Twelve rows, three features, classes 0/1/2 interleaved so every batch of six
# holds two rows of each class.
X = [
    [0.9, 0.1, 0.0], [0.0, 1.0, 0.2], [0.1, 0.0, 0.8], [0.7, 0.3, 0.1],
    [0.2, 0.8, 0.0], [0.0, 0.2, 1.0], [1.0, 0.0, 0.2], [0.1, 0.9, 0.1],
    [0.3, 0.1, 0.9], [0.8, 0.2, 0.3], [0.0, 0.7, 0.3], [0.2, 0.3, 0.7],
]
LABEL = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
# The Engine's initialisation for model_seed 52, as the Engine test prints it:
# Dense weights in the Engine's layout ([out, in], as torch), then biases.
INITIAL: dict[str, list[float]] = {
    "layer0.weight": [-0.61620295, 0.884442687, -0.0481370613, 0.423617989, 0.599007249, -0.0626468137,
                      0.785875797, -0.120244257, 0.799673915, -0.773547769, -0.119739108, -0.474215657],
    "layer0.bias": [0, 0, 0, 0],
    "layer2.weight": [0.50926125, -0.529384732, -0.299112082, 0.0942151546, 0.19807744, 0.324198008,
                      -0.789388776, 0.342939854],
    "layer2.bias": [0, 0],
}
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "triplet_graph_pytorch.json"


def mix(value: int) -> int:
    z = (value + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def training_epoch_seed(run_seed: int, epoch: int) -> int:
    # training_resume_checkpoint.cpp TrainingEpochSeed
    z = (run_seed + 0x9E3779B97F4A7C15 * (epoch + 1)) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def select_triplets(class_ids: list[int], key: int, batch_index: int) -> list[tuple[int, int, int]]:
    batch_key = mix(key ^ mix(batch_index))
    triplets = []
    for row, cls in enumerate(class_ids):
        positives = [i for i, c in enumerate(class_ids) if c == cls and i != row]
        negatives = [i for i, c in enumerate(class_ids) if c != cls]
        if not positives or not negatives:
            continue
        p = positives[mix(batch_key ^ mix(2 * row)) % len(positives)]
        n = negatives[mix(batch_key ^ mix(2 * row + 1)) % len(negatives)]
        triplets.append((row, p, n))
    return triplets


def engine_names() -> tuple[str, str, str, str]:
    names = list(INITIAL)
    weights = [n for n in names if n.endswith("weight")]
    biases = [n for n in names if n.endswith("bias")]
    if len(weights) != 2 or len(biases) != 2:
        raise SystemExit("fill INITIAL from the Engine test's printout first")
    return weights[0], biases[0], weights[1], biases[1]


def case(name: str, margin: float) -> dict[str, Any]:
    w1, b1, w2, b2 = engine_names()
    first = torch.nn.Linear(3, 4)
    second = torch.nn.Linear(4, 2)
    with torch.no_grad():
        # The Engine's Dense weight is [out, in], as torch's
        first.weight.copy_(torch.tensor(INITIAL[w1]).reshape(4, 3))
        first.bias.copy_(torch.tensor(INITIAL[b1]))
        second.weight.copy_(torch.tensor(INITIAL[w2]).reshape(2, 4))
        second.bias.copy_(torch.tensor(INITIAL[b2]))
    encoder = torch.nn.Sequential(first, torch.nn.ReLU(), second)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=LEARNING_RATE)
    x = torch.tensor(X)
    losses = []
    for epoch in range(1, EPOCHS + 1):
        key = training_epoch_seed(DATALOADER_SEED, epoch)
        for batch_index, start in enumerate(range(0, len(LABEL), BATCH_SIZE)):
            triplets = select_triplets(LABEL[start:start + BATCH_SIZE], key, batch_index)
            rows = [start + t[0] for t in triplets] + [start + t[1] for t in triplets] + \
                   [start + t[2] for t in triplets]
            optimizer.zero_grad()
            embeddings = encoder(x[rows])
            count = len(triplets)
            loss = torch.nn.functional.triplet_margin_loss(
                embeddings[:count], embeddings[count:2 * count], embeddings[2 * count:], margin=margin)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return {
        "name": name,
        "margin": margin,
        "batch_losses": losses,
        "parameters_after": {
            w1: first.weight.detach().reshape(-1).tolist(),
            b1: first.bias.detach().tolist(),
            w2: second.weight.detach().reshape(-1).tolist(),
            b2: second.bias.detach().tolist(),
        },
    }


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
        "dataloader_seed": DATALOADER_SEED,
        "tolerance": 1.0e-5,
        "x": X,
        "label": LABEL,
        "initial_parameters": INITIAL,
        # The sampler's picks for the first batch of epoch 1, which the Engine
        # test checks against SelectBatchTriplets directly.
        "first_batch_triplets": [list(t) for t in select_triplets(
            LABEL[:BATCH_SIZE], training_epoch_seed(DATALOADER_SEED, 1), 0)],
        "cases": [
            case("margin_1", 1.0),
            # A small margin leaves some triplets already satisfied (zero loss).
            case("margin_0_2", 0.2),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
