#!/usr/bin/env python3
"""Level 3 oracle fixtures (TOFIX228 1.6), tabular: what the Engine's data path
must make of one CSV, worked out in Python without the Engine.

Writes fixtures/data_parity/tabular.csv (53 rows: age, income, score, visits,
a text label `segment`) and fixtures/data_parity_tabular_pytorch.json:

  partition   each row's part (0 train, 1 validation, 2 test): the Engine's
              split rule (split_partitioning.h: CountsPerClass, SplitAssigner
              on std::mt19937_64, stratified by first appearance of a class)
              ported to Python; 0.7 / 0.15 / 0.15, seed 7, shuffled;
  unshuffled_partition  scenario B: the same rule unstratified and unshuffled
              (contiguous slices with its largest-remainder counts);
  scaled      age and income through StandardScaler's formula (scikit-learn's:
              population variance) fitted on the Train rows only, every row
              transformed;
  features    each row as the batcher should serve it (age, income scaled;
              score, visits as read), class_ids over the sorted classes
              (scikit-learn's LabelEncoder order);
  epochs      per epoch: the Train rows' order (std::mt19937 seeded with
              TrainingEpochSeed(dataloader seed 11, epoch), Fisher-Yates from
              the back: portable_random.h, ported), its batch sizes, the
              validation loss and every parameter after the epoch;
  steps       per step the epoch's running loss (sample-weighted mean).

The training is PyTorch running the model's own export
(fixtures/data_parity/tabular_mlp.py, written by
`test_oracle_data --write-exports <that folder>`), as level 2 does: its setup,
the lazy layers materialised in eval mode, the starting weights saved
(tabular_mlp_init.pth), then its example loop, uncommented, over a torch
DataLoader whose sampler gives each epoch's order.

    py -3.12 generate_data_parity_fixtures.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from portable_random import MT19937, MT19937_64, below, shuffle

SCHEMA_VERSION = 1
FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
DATA_DIR = FIXTURES / "data_parity"
OUTPUT = FIXTURES / "data_parity_tabular_pytorch.json"
CSV_NAME = "tabular.csv"
MODEL = "tabular_mlp"
ROWS = 53
SPLIT = (0.7, 0.15, 0.15)
SPLIT_SEED = 7
LOADER_SEED = 11
BATCH = 8
EPOCHS = 2
TORCH_SEED = 2286
M64 = (1 << 64) - 1
TOLERANCE = {"atol": 2e-5, "rtol": 2e-4, "data_atol": 1e-6, "data_rtol": 1e-6}
LOOP_MARKER = "    # Example training loop:\n"
SETUP_END = "    # TODO: Add your training data here\n"


# ------------------------------------------------------------------- data --

def write_csv(path: Path) -> list[dict[str, Any]]:
    rng = np.random.default_rng(2286)
    segments = rng.choice(["retail", "online", "partner"], size=ROWS, p=[0.5, 0.32, 0.18])
    rows = []
    for i in range(ROWS):
        rows.append({
            "age": round(float(rng.uniform(18, 80)), 1),
            "income": round(float(rng.lognormal(10.5, 0.6)), 2),
            "score": round(float(rng.normal(0.0, 1.0)), 4),
            "visits": int(rng.integers(0, 21)),
            "segment": str(segments[i]),
        })
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["age", "income", "score", "visits", "segment"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


# ------------------------------------------------------------- the split --

def f32(x: float) -> np.float32:
    return np.float32(x)


def counts_per_class(class_rows: list[int], shares: tuple[float, float, float], seed: int) -> list[list[int]]:
    """split_partition_detail::CountsPerClass."""
    total_share = f32(shares[0]) + f32(shares[1]) + f32(shares[2])
    share = [float(f32(s) / total_share) for s in shares]  # float32 division, read as double
    total = sum(class_rows)
    target = [0, 0, 0]
    fraction = [0.0, 0.0, 0.0]
    assigned = 0
    for p in range(3):
        exact = float(total) * share[p]
        target[p] = int(exact)
        fraction[p] = exact - target[p]
        assigned += target[p]
    order = sorted(range(3), key=lambda p: -fraction[p])  # stable
    i = 0
    while assigned < total:
        target[order[i % 3]] += 1
        i += 1
        assigned += 1
    if total > 0 and target[0] == 0 and share[0] > 0.0:
        src = 1 if target[1] >= target[2] else 2
        target[src] -= 1
        target[0] = 1

    counts = [[0, 0, 0] for _ in class_rows]
    left = [0] * len(class_rows)
    need = list(target)
    remainders = []
    for c, rows in enumerate(class_rows):
        floors = 0
        for p in range(3):
            exact = float(rows) * share[p]
            counts[c][p] = int(exact)
            floors += counts[c][p]
            need[p] -= counts[c][p]
            remainders.append((exact - counts[c][p], c, p))
        left[c] = rows - floors

    def tie_key(cls: int) -> int:
        z = ((seed << 32) + cls + 0x9E3779B97F4A7C15) & M64
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    remainders.sort(key=lambda r: (-r[0], tie_key(r[1])))  # stable: equal keys keep their order
    for _, c, p in remainders:
        if left[c] > 0 and need[p] > 0:
            counts[c][p] += 1
            left[c] -= 1
            need[p] -= 1
    for c in range(len(class_rows)):
        while left[c] > 0:
            p = max(range(3), key=lambda q: (need[q], -q))  # max_element: the first largest
            counts[c][p] += 1
            left[c] -= 1
            need[p] -= 1
    if total > 0 and share[0] > 0.0 and sum(c[0] for c in counts) == 0:
        best, best_c, best_p = 0, 0, 0
        for c in range(len(counts)):
            for p in (1, 2):
                if counts[c][p] > best:
                    best, best_c, best_p = counts[c][p], c, p
        if best > 0:
            counts[best_c][best_p] -= 1
            counts[best_c][0] += 1
    return counts


def assign_split(class_of_row: list[int], shares, shuffled: bool, seed: int) -> list[int]:
    """AssignSplitPartitions / SplitAssigner."""
    class_rows = [0] * (max(class_of_row) + 1)
    for c in class_of_row:
        class_rows[c] += 1
    remaining = counts_per_class(class_rows, shares, seed)
    rng = MT19937_64(seed)
    out = []
    for c in class_of_row:
        left = remaining[c]
        total = sum(left)
        draw = below(rng, total) if shuffled and total > 0 else 0
        part = 0
        while part < 2 and draw >= left[part]:
            draw -= left[part]
            part += 1
        if left[part] > 0:
            left[part] -= 1
        out.append(part)
    return out


def training_epoch_seed(run_seed: int, epoch: int) -> int:
    """TrainingEpochSeed (training_resume_checkpoint.cpp)."""
    z = (run_seed + 0x9E3779B97F4A7C15 * (epoch + 1)) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def standard_scaled(values: np.ndarray, train: list[int]) -> np.ndarray:
    """scikit-learn's StandardScaler fitted on the Train rows, every row
    transformed: (x - mean) / sqrt(population variance), a zero scale read as 1."""
    mean = values[train].mean(axis=0)
    scale = np.sqrt(values[train].var(axis=0))
    scale[scale == 0.0] = 1.0
    return (values - mean) / scale


# ---------------------------------------------------------- the training --

def plain(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().to(torch.float32).contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def split_export(code: str) -> tuple[str, str]:
    setup = code[: code.index(SETUP_END)]
    loop = []
    for line in code[code.index(LOOP_MARKER) + len(LOOP_MARKER):].splitlines():
        if not line.strip():
            continue
        match = re.match(r"^    # ?(.*)$", line)
        if not match:
            raise ValueError("unexpected line in the example loop: " + line)
        loop.append(match.group(1))
    return setup, "\n".join(loop) + "\n"


class EpochOrderSampler(torch.utils.data.Sampler):
    """Each pass gives the next epoch's order."""

    def __init__(self, orders: list[list[int]]):
        self.orders, self.epoch = orders, 0

    def __iter__(self):
        order = self.orders[self.epoch]
        self.epoch += 1
        return iter(order)

    def __len__(self):
        return len(self.orders[0])


class RecordingLoader:
    """The example loop's train_loader: a torch DataLoader over the rows with
    the epoch orders; records each step's running loss and, after each epoch,
    the validation loss and every parameter."""

    def __init__(self, loader, namespace, val_x, val_y):
        self.loader, self.ns, self.val_x, self.val_y = loader, namespace, val_x, val_y
        self.steps: list[dict[str, Any]] = []
        self.epochs: list[dict[str, Any]] = []

    def __iter__(self):
        weighted, seen, sizes, pending = 0.0, 0, [], None
        for data, target in self.loader:
            if pending is not None:
                weighted, seen = self.step(pending, weighted, seen)
            pending = len(target)
            sizes.append(pending)
            yield data, target
        weighted, seen = self.step(pending, weighted, seen)
        model, criterion = self.ns["model"], self.ns["criterion"]
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(self.val_x), self.val_y))
        model.train()
        self.epochs.append({"batch_sizes": sizes, "val_loss": val_loss,
                            "parameters": {k: plain(v) for k, v in model.state_dict().items()
                                           if v.is_floating_point()}})

    def step(self, size, weighted, seen):
        weighted += float(self.ns["loss"].detach()) * size
        seen += size
        self.steps.append({"step": len(self.steps) + 1, "running_loss": weighted / seen})
        return weighted, seen


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = write_csv(DATA_DIR / CSV_NAME)

    first_seen: dict[str, int] = {}
    class_of_row = [first_seen.setdefault(r["segment"], len(first_seen)) for r in rows]
    partition = assign_split(class_of_row, SPLIT, True, SPLIT_SEED)
    # Scenario B: no stratification, no shuffle (one class, contiguous slices).
    unshuffled = assign_split([0] * ROWS, SPLIT, False, SPLIT_SEED)
    train = [i for i, p in enumerate(partition) if p == 0]
    val = [i for i, p in enumerate(partition) if p == 1]

    raw = np.array([[r["age"], r["income"]] for r in rows], dtype=np.float64)
    scaled = standard_scaled(raw, train)
    classes = sorted(first_seen)
    class_ids = [classes.index(r["segment"]) for r in rows]
    features = [[float(scaled[i, 0]), float(scaled[i, 1]), r["score"], float(r["visits"])] for i, r in enumerate(rows)]

    orders = []
    for epoch in range(1, EPOCHS + 1):
        rng = MT19937(training_epoch_seed(LOADER_SEED, epoch) & 0xFFFFFFFF)
        orders.append(shuffle(list(train), rng))

    code = (DATA_DIR / f"{MODEL}.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    setup, loop = split_export(code)
    torch.manual_seed(TORCH_SEED)
    ns: dict[str, Any] = {"__name__": "__main__"}
    exec(compile(setup, f"{MODEL}.py (setup)", "exec"), ns)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(class_ids, dtype=torch.long)
    model = ns["model"]
    model.eval()
    with torch.no_grad():
        model(x[:BATCH])
    model.train()
    torch.save({k: v.detach().clone() for k, v in model.state_dict().items()}, DATA_DIR / f"{MODEL}_init.pth")

    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, sampler=EpochOrderSampler(orders),
                                         drop_last=False)
    recorder = RecordingLoader(loader, ns, x[val], y[val])
    ns.update({"train_loader": recorder, "num_epochs": EPOCHS})
    exec(compile(loop, f"{MODEL}.py (example loop)", "exec"), ns)
    for epoch, order in zip(recorder.epochs, orders):
        epoch["order"] = order

    counts = [partition.count(p) for p in range(3)]
    print(f"{ROWS} rows: {counts[0]} train / {counts[1]} val / {counts[2]} test; classes {classes}; "
          f"unshuffled {[unshuffled.count(p) for p in range(3)]}")
    print(f"loss {recorder.steps[0]['running_loss']:.6f} -> {recorder.steps[-1]['running_loss']:.6f}, "
          f"val {[round(e['val_loss'], 6) for e in recorder.epochs]}")
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "tolerance": TOLERANCE,
        "csv": CSV_NAME,
        "rows": ROWS,
        "export_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "init_weights": f"{MODEL}_init.pth",
        "classes": classes,
        "partition": partition,
        "unshuffled_partition": unshuffled,
        "scaled": {"age": scaled[:, 0].tolist(), "income": scaled[:, 1].tolist()},
        "features": features,
        "class_ids": class_ids,
        "epochs": recorder.epochs,
        "steps": recorder.steps,
    }
    OUTPUT.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
