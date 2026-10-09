#!/usr/bin/env python3
"""PyTorch fixtures for metric learning (TOFIX140 A5): Data Input -> Pair or
Triplet Dataset Builder -> Dense(3 -> 4) -> ReLU -> Dense(4 -> 2) -> Triplet,
Contrastive or Cosine Embedding Loss -> SGD, from the Engine's own seeded start
(INITIAL below, which the Engine test checks against its initialisation first).

Every step is what the Engine does: the builder picks, from each batch, a
positive of the same class and a negative of another class for every row
(SplitMix64 keyed by the epoch's DataLoader seed, the batch index and the row:
select_triplets / select_pairs mirror metric_learning_sampling.cpp), the
encoder runs once over the stacked rows, and

    triplet:     triplet_margin_loss(a, p, n, margin)            # p = 2
    contrastive: mean(d^2 if similar else max(0, margin - d)^2)  # d = |a - b|
    cosine:      cosine_embedding_loss(a, b, +1 / -1, margin)
    loss.backward(); optimizer.step()

    py -3.12 generate_metric_graph_fixtures.py
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
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "metric_graph_pytorch.json"


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


class BatchClasses:
    def __init__(self, class_ids: list[int], key: int, batch_index: int) -> None:
        self.ids = class_ids
        self.batch_key = mix(key ^ mix(batch_index))

    def positives(self, row: int) -> list[int]:
        return [i for i, c in enumerate(self.ids) if c == self.ids[row] and i != row]

    def negatives(self, row: int) -> list[int]:
        return [i for i, c in enumerate(self.ids) if c != self.ids[row]]

    def positive(self, row: int) -> int:
        candidates = self.positives(row)
        return candidates[mix(self.batch_key ^ mix(2 * row)) % len(candidates)]

    def negative(self, row: int) -> int:
        candidates = self.negatives(row)
        return candidates[mix(self.batch_key ^ mix(2 * row + 1)) % len(candidates)]


def select_triplets(class_ids: list[int], key: int, batch_index: int) -> list[tuple[int, int, int]]:
    classes = BatchClasses(class_ids, key, batch_index)
    return [(row, classes.positive(row), classes.negative(row)) for row in range(len(class_ids))
            if classes.positives(row) and classes.negatives(row)]


def select_pairs(class_ids: list[int], key: int, batch_index: int) -> list[tuple[int, int, int]]:
    classes = BatchClasses(class_ids, key, batch_index)
    pairs = []
    for row in range(len(class_ids)):
        has_positive = bool(classes.positives(row))
        has_negative = bool(classes.negatives(row))
        if not has_positive and not has_negative:
            continue
        wants_similar = ((row + classes.batch_key) & 1) == 0
        similar = has_positive and (wants_similar or not has_negative)
        pairs.append((row, classes.positive(row) if similar else classes.negative(row), 1 if similar else 0))
    return pairs


def engine_names() -> tuple[str, str, str, str]:
    names = list(INITIAL)
    weights = [n for n in names if n.endswith("weight")]
    biases = [n for n in names if n.endswith("bias")]
    if len(weights) != 2 or len(biases) != 2:
        raise SystemExit("fill INITIAL from the Engine test's printout first")
    return weights[0], biases[0], weights[1], biases[1]


def batch_loss(kind: str, embeddings: torch.Tensor, picks: list[tuple[int, int, int]], margin: float) -> torch.Tensor:
    count = len(picks)
    if kind == "triplet":
        return torch.nn.functional.triplet_margin_loss(
            embeddings[:count], embeddings[count:2 * count], embeddings[2 * count:], margin=margin)
    a, b = embeddings[:count], embeddings[count:]
    similar = torch.tensor([float(p[2]) for p in picks])
    if kind == "contrastive":
        distance_sq = ((a - b) ** 2).sum(dim=1)
        distance = distance_sq.sqrt()
        return (similar * distance_sq + (1 - similar) * torch.clamp(margin - distance, min=0) ** 2).mean()
    return torch.nn.functional.cosine_embedding_loss(a, b, 2 * similar - 1, margin=margin)


def case(name: str, kind: str, margin: float) -> dict[str, Any]:
    w1, b1, w2, b2 = engine_names()
    first = torch.nn.Linear(3, 4)
    second = torch.nn.Linear(4, 2)
    with torch.no_grad():
        # The Engine's Dense weight is [out, in], as torch's
        first.weight.copy_(torch.tensor(INITIAL[w1]).reshape(4, 3))
        first.bias.copy_(torch.tensor(INITIAL[b1], dtype=torch.float32))
        second.weight.copy_(torch.tensor(INITIAL[w2]).reshape(2, 4))
        second.bias.copy_(torch.tensor(INITIAL[b2], dtype=torch.float32))
    encoder = torch.nn.Sequential(first, torch.nn.ReLU(), second)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=LEARNING_RATE)
    x = torch.tensor(X)
    select = select_triplets if kind == "triplet" else select_pairs
    blocks = 3 if kind == "triplet" else 2
    losses = []
    for epoch in range(1, EPOCHS + 1):
        key = training_epoch_seed(DATALOADER_SEED, epoch)
        for batch_index, start in enumerate(range(0, len(LABEL), BATCH_SIZE)):
            picks = select(LABEL[start:start + BATCH_SIZE], key, batch_index)
            rows = [start + p[block] for block in range(blocks) for p in picks]
            optimizer.zero_grad()
            loss = batch_loss(kind, encoder(x[rows]), picks, margin)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return {
        "name": name,
        "kind": kind,
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
    first_key = training_epoch_seed(DATALOADER_SEED, 1)
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
        # The samplers' picks for the first batch of epoch 1, which the Engine
        # test checks against SelectBatchTriplets / SelectBatchPairs directly.
        "first_batch_triplets": [list(t) for t in select_triplets(LABEL[:BATCH_SIZE], first_key, 0)],
        "first_batch_pairs": [list(p) for p in select_pairs(LABEL[:BATCH_SIZE], first_key, 0)],
        "cases": [
            case("triplet_margin_1", "triplet", 1.0),
            # A small margin leaves some triplets already satisfied (zero loss).
            case("triplet_margin_0_2", "triplet", 0.2),
            case("contrastive_margin_1", "contrastive", 1.0),
            case("contrastive_margin_0_3", "contrastive", 0.3),
            case("cosine_margin_0", "cosine", 0.0),
            case("cosine_margin_0_5", "cosine", 0.5),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
