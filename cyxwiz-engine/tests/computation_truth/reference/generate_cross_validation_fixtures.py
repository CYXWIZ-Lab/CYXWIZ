"""Truth for Results > Cross-validation and Learning curve (TOFIX129 D4): PyTorch trains the same
graph the C++ test trains, on the same folds, with several seeds, so the Engine's fold scores can be
checked against the spread of real training.

Data: the Palmer penguins rows with all four measurements (bill length / depth, flipper length, body
mass), standardised (z-scores), species as class 0 / 1 / 2 (Adelie, Chinstrap, Gentoo).
Folds: stratified, 5 folds, fixed here and written to the fixture (the test hands them to the
runner), so both sides train on exactly the same rows.
Graph: Dense 4 -> 16 (Xavier uniform, zero bias, as the Engine's Dense), ReLU, Dense 16 -> 3, cross
entropy, Adam lr 0.01, batch 32, 20 epochs, shuffled each epoch.
Per fold: validation accuracy and loss over SEEDS seeds (mean, sd). Learning curve: for shares 0.2,
0.5 and 1.0 of each fold's training rows (a random subset per seed), training and validation
accuracy, mean and sd over folds x seeds.

Run (PyTorch + numpy; the Python with torch on this machine is `py -3.12`):
  py -3.12 generate_cross_validation_fixtures.py path/to/penguins.csv
"""
import csv
import json
import os
import sys

import numpy as np
import torch

FOLDS = 5
SEEDS = 8
EPOCHS = 20
BATCH = 32
LR = 0.01
SHARES = [0.2, 0.5, 1.0]
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

rows = []
with open(sys.argv[1], newline="") as f:
    for r in csv.DictReader(f):
        try:
            values = [float(r[c]) for c in FEATURES]
        except ValueError:
            continue
        rows.append((values, r["species"]))
classes = sorted({s for _, s in rows})
X = np.array([v for v, _ in rows], dtype=np.float64)
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = np.array([classes.index(s) for _, s in rows], dtype=np.int64)

# Stratified folds: each class shuffled, then dealt round-robin.
rng = np.random.default_rng(7)
fold = np.zeros(len(y), dtype=np.int64)
for c in range(len(classes)):
    members = rng.permutation(np.flatnonzero(y == c))
    fold[members] = np.arange(len(members)) % FOLDS


def train(train_idx, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    first, second = torch.nn.Linear(4, 16).double(), torch.nn.Linear(16, 3).double()
    for layer in (first, second):
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
    model = torch.nn.Sequential(first, torch.nn.ReLU(), second)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    xt, yt = torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx])
    for _ in range(EPOCHS):
        order = torch.randperm(len(train_idx), generator=g)
        for start in range(0, len(train_idx), BATCH):
            batch = order[start:start + BATCH]
            opt.zero_grad()
            loss_fn(model(xt[batch]), yt[batch]).backward()
            opt.step()
    return model


def score(model, idx):
    with torch.no_grad():
        logits = model(torch.from_numpy(X[idx]))
        loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(y[idx])).item()
        accuracy = (logits.argmax(dim=1).numpy() == y[idx]).mean()
    return float(accuracy), float(loss)


per_fold = []
for k in range(FOLDS):
    tr, va = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    accs, losses = zip(*(score(train(tr, s), va) for s in range(SEEDS)))
    per_fold.append({"fold": k + 1, "train_rows": int(len(tr)), "val_rows": int(len(va)),
                     "accuracy_mean": float(np.mean(accs)), "accuracy_sd": float(np.std(accs)),
                     "loss_mean": float(np.mean(losses)), "loss_sd": float(np.std(losses))})

curve = []
for share in SHARES:
    tr_acc, va_acc, sizes = [], [], []
    for k in range(FOLDS):
        tr, va = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        n = max(1, int(np.floor(share * len(tr) + 0.5)))  # half up, as the runner's lround
        sizes.append(n)
        for s in range(SEEDS):
            subset = np.random.default_rng(100 + s).permutation(tr)[:n]
            model = train(subset, s)
            tr_acc.append(score(model, subset)[0])
            va_acc.append(score(model, va)[0])
    curve.append({"share": share, "train_rows": sizes, "train_accuracy_mean": float(np.mean(tr_acc)),
                  "train_accuracy_sd": float(np.std(tr_acc)), "val_accuracy_mean": float(np.mean(va_acc)),
                  "val_accuracy_sd": float(np.std(va_acc))})

out = {"schema_version": 1, "oracle": {"name": "PyTorch", "version": torch.__version__},
       "graph": {"hidden": 16, "classes": len(classes), "learning_rate": LR, "batch_size": BATCH, "epochs": EPOCHS},
       "features": FEATURES, "class_names": classes, "seeds": SEEDS,
       "x": [[round(v, 10) for v in r] for r in X.tolist()], "y": y.tolist(), "fold": (fold + 1).tolist(),
       "folds": per_fold, "learning_curve": curve}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "cross_validation_pytorch.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", os.path.normpath(path))
for p in per_fold:
    print("fold %d: %d / %d rows, accuracy %.4f +- %.4f, loss %.4f" % (p["fold"], p["train_rows"], p["val_rows"],
                                                                    p["accuracy_mean"], p["accuracy_sd"], p["loss_mean"]))
for c in curve:
    print("share %.1f: train %.4f +- %.4f, validation %.4f +- %.4f" % (c["share"], c["train_accuracy_mean"],
                                                                     c["train_accuracy_sd"], c["val_accuracy_mean"],
                                                                     c["val_accuracy_sd"]))
