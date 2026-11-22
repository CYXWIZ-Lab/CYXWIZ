#!/usr/bin/env python3
"""Level 3 oracle fixtures (TOFIX228 1.6), batch level for the other data
kinds: what each kind's batchers must serve for one small committed file,
worked out in Python without the Engine (no third-party package beyond
numpy and Pillow).

For each kind the fixture holds: the Train rows of each epoch in order (the
epoch shuffle: std::mt19937 seeded with TrainingEpochSeed(dataloader seed 11,
epoch), Fisher-Yates from the back, ported in portable_random.py), the
Validation and Test rows, and every row's values and label as the batcher
should serve them. test_oracle_data reads the batchers as a training run
does (the epoch seed, then Reset) and compares.

  text    fixtures/data_parity/text.csv through TextTokenizer (Word tokens:
          runs of letters and digits, lowercased, punctuation dropped; the
          vocabulary from the Train rows: count >= min_word_freq, by count
          then token, after the 4 special ids, max_vocab_size in all; ids
          truncated / padded with 0 to max_length), the training split
          stratified on the label (the split rule, ported in
          generate_data_parity_fixtures.py), class ids by first appearance;
  series  fixtures/data_parity/series.csv through TimeSeriesWindow (one window
          per start row: the value window, then each feature's window, the
          target `shift` steps after the window) and TimeSeriesSplit
          (chronological: floor(windows x ratio) in float32, Validation takes
          a row from Train when it would be empty).

    py -3.12 generate_data_parity_kinds.py
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from generate_data_parity_fixtures import (EpochOrderSampler, RecordingLoader, assign_split, split_export,
                                           training_epoch_seed)
from portable_random import MT19937, shuffle

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
DATA_DIR = FIXTURES / "data_parity"
LOADER_SEED = 11
EPOCHS = 2


def epoch_orders(train_rows: list[int]) -> list[list[int]]:
    return [shuffle(list(train_rows), MT19937(training_epoch_seed(LOADER_SEED, e) & 0xFFFFFFFF))
            for e in range(1, EPOCHS + 1)]


def write_fixture(name: str, fixture: dict) -> None:
    path = FIXTURES / f"data_parity_{name}.json"
    path.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


# ------------------------------------------------------------------- text --

WORDS = ["great", "film", "bad", "plot", "acting", "music", "slow", "fun", "boring", "loved", "hated", "ending",
         "story", "cast", "2024", "ok"]


def write_text_csv(path: Path) -> list[dict]:
    rng = np.random.default_rng(22861)
    labels = rng.choice(["pos", "neg", "mixed"], size=40, p=[0.45, 0.35, 0.2])
    rows = []
    for i in range(40):
        n = int(rng.integers(2, 10))
        words = [WORDS[int(k)] for k in rng.integers(0, len(WORDS), size=n)]
        # Capitals and punctuation the tokenizer must undo.
        words = [w.upper() if rng.random() < 0.15 else (w.capitalize() if rng.random() < 0.2 else w) for w in words]
        text = " ".join(w + ("," if rng.random() < 0.15 else "") for w in words) + rng.choice([".", "!", "", "?"])
        rows.append({"text": text, "label": str(labels[i])})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"], lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def text_kind() -> None:
    max_length, min_freq, max_vocab = 6, 2, 12
    rows = write_text_csv(DATA_DIR / "text.csv")
    first_seen: dict[str, int] = {}
    class_of_row = [first_seen.setdefault(r["label"], len(first_seen)) for r in rows]
    partition = assign_split(class_of_row, (0.7, 0.15, 0.15), True, 7)
    tokens = [re.findall(r"[a-z0-9]+", r["text"].lower()) for r in rows]
    counts: dict[str, int] = {}
    for i, p in enumerate(partition):
        if p == 0:
            for tok in tokens[i]:
                counts[tok] = counts.get(tok, 0) + 1
    kept = sorted(((tok, c) for tok, c in counts.items() if c >= min_freq), key=lambda tc: (-tc[1], tc[0]))
    kept = kept[: max(0, max_vocab - 4)]
    vocab = {tok: 4 + i for i, (tok, _) in enumerate(kept)}  # 0 [PAD], 1 [UNK], 2 [BOS], 3 [EOS]
    values, labels = {}, {}
    for i, toks in enumerate(tokens):
        ids = [vocab.get(t, 1) for t in toks][:max_length]
        values[str(i)] = [float(v) for v in ids + [0] * (max_length - len(ids))]
        labels[str(i)] = float(class_of_row[i])
    train = [i for i, p in enumerate(partition) if p == 0]
    print(f"text: {len(rows)} rows, {partition.count(0)} / {partition.count(1)} / {partition.count(2)}, "
          f"vocabulary {[t for t, _ in kept]}")
    write_fixture("text", {
        "summary": f"{len(rows)} reviews, Word tokens, vocabulary of {max_vocab} from the Train rows, length {max_length}",
        "tokenizer": {"text_col": "text", "label_col": "label", "max_length": str(max_length),
                      "min_word_freq": str(min_freq), "max_vocab_size": str(max_vocab), "tokenizer_type": "1"},
        "training_split": {"label_column": "label", "train": 0.7, "val": 0.15, "test": 0.15, "seed": 7,
                           "stratified": True, "shuffle": True},
        "batch_size": 4,
        "classes": len(first_seen),
        "tolerance": {"atol": 0.0, "rtol": 0.0},
        "vocabulary": [t for t, _ in kept],
        "partition": partition,
        "epochs": epoch_orders(train),
        "val_rows": [i for i, p in enumerate(partition) if p == 1],
        "test_rows": [i for i, p in enumerate(partition) if p == 2],
        "values": values,
        "labels": labels,
    })


# ----------------------------------------------------------------- series --

def series_kind() -> None:
    n, width, shift, label_width = 30, 4, 1, 1
    train_ratio, val_ratio = 0.6, 0.2
    value = [round(math.sin(0.5 * t) + 0.05 * t, 4) for t in range(n)]
    temp = [round(20.0 + 3.0 * math.cos(0.3 * t), 3) for t in range(n)]
    with (DATA_DIR / "series.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["t", "value", "temp"])
        for t in range(n):
            writer.writerow([t, value[t], temp[t]])
    windows = n - (width + shift + label_width - 1) + 1
    values, labels = {}, {}
    for w in range(windows):
        values[str(w)] = value[w:w + width] + temp[w:w + width]  # channel-major: each column's window in turn
        labels[str(w)] = value[w + width + shift - 1]
    train_count = int(math.floor(float(np.float32(windows) * np.float32(train_ratio))))
    val_count = int(math.floor(float(np.float32(windows) * np.float32(val_ratio))))
    if train_count + val_count > windows:
        val_count = windows - train_count
    if val_count == 0 and train_count > 1:
        train_count, val_count = train_count - 1, 1
    partition = [0] * train_count + [1] * val_count + [2] * (windows - train_count - val_count)
    train = [w for w, p in enumerate(partition) if p == 0]
    print(f"series: {windows} windows, {partition.count(0)} / {partition.count(1)} / {partition.count(2)}")
    write_fixture("series", {
        "summary": f"{n} readings, windows of {width} over value and temp, target {shift} step on, chronological "
                   f"{train_ratio} / {val_ratio}",
        "window": {"value_col": "value", "feature_cols": "temp", "time_col": "t", "input_width": str(width),
                   "label_width": str(label_width), "shift": str(shift)},
        "split": {"train_ratio": str(train_ratio), "val_ratio": str(val_ratio),
                  "test_ratio": str(round(1.0 - train_ratio - val_ratio, 6))},
        "batch_size": 4,
        "tolerance": {"atol": 1e-6, "rtol": 1e-6},
        "partition": partition,
        "epochs": epoch_orders(train),
        "val_rows": [w for w, p in enumerate(partition) if p == 1],
        "test_rows": [w for w, p in enumerate(partition) if p == 2],
        "values": values,
        "labels": labels,
    })


# ------------------------------------------------------------------ image --

def image_kind() -> None:
    """Class folders (sorted names are the class ids), files sorted by path.
    Each image is decoded to RGB (grey repeated, alpha dropped) as u8 / 255
    in [H, W, C] order; an 8 x 8 image is shrunk to 4 x 4 by its 2 x 2 block
    means (OpenCV's area resize), built from blocks of one value so the mean
    is exact. The split is the rule shuffled, stratified on the class id."""
    from PIL import Image

    size = 4
    root = DATA_DIR / "images"
    rng = np.random.default_rng(22862)
    files: list[tuple[str, str, np.ndarray]] = []  # (class, name, expected [H, W, 3] u8)
    for cls in ("cat", "dog"):
        (root / cls).mkdir(parents=True, exist_ok=True)
    rgb = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    Image.fromarray(rgb, "RGB").save(root / "cat" / "c0.png")
    files.append(("cat", "c0.png", rgb))
    grey = rng.integers(0, 256, (size, size), dtype=np.uint8)
    Image.fromarray(grey, "L").save(root / "cat" / "c1.png")
    files.append(("cat", "c1.png", np.repeat(grey[:, :, None], 3, axis=2)))
    base = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    big = np.repeat(np.repeat(base, 2, axis=0), 2, axis=1)  # 8 x 8, each 2 x 2 block one value
    Image.fromarray(big, "RGB").save(root / "cat" / "c2.png")
    files.append(("cat", "c2.png", base))
    rgba = rng.integers(0, 256, (size, size, 4), dtype=np.uint8)
    Image.fromarray(rgba, "RGBA").save(root / "dog" / "d0.png")
    files.append(("dog", "d0.png", rgba[:, :, :3]))
    for name in ("d1.png", "d2.png", "d3.png"):
        img = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        Image.fromarray(img, "RGB").save(root / "dog" / name)
        files.append(("dog", name, img))
    files.sort(key=lambda f: (f[0], f[1]))
    classes = sorted({f[0] for f in files})
    labels_of = [classes.index(f[0]) for f in files]
    split = {"train": 0.5, "val": 0.25, "test": 0.25, "stratified": True, "seed": 7}
    partition = assign_split(labels_of, (split["train"], split["val"], split["test"]), True, split["seed"])
    values = {str(i): (f[2].astype(np.float32) / np.float32(255.0)).reshape(-1).tolist() for i, f in enumerate(files)}
    labels = {str(i): float(labels_of[i]) for i in range(len(files))}
    train = [i for i, p in enumerate(partition) if p == 0]
    print(f"image: {len(files)} images, {partition.count(0)} / {partition.count(1)} / {partition.count(2)}")
    write_fixture("image", {
        "summary": f"{len(files)} PNG in 2 class folders (RGB, grey, RGBA, one 8 x 8), at {size} x {size}",
        "rows": len(files), "classes": len(classes), "size": size, "batch_size": 2, "split": split,
        "tolerance": {"atol": 1e-7, "rtol": 0.0},
        "files": [f"{f[0]}/{f[1]}" for f in files],
        "partition": partition,
        "epochs": epoch_orders(train),
        "val_rows": [i for i, p in enumerate(partition) if p == 1],
        "test_rows": [i for i, p in enumerate(partition) if p == 2],
        "values": values,
        "labels": labels,
    })


# ------------------------------------------------------------------ audio --

def audio_kind() -> None:
    """Class folders of 16-bit mono WAVs at the target rate, each clip read
    as int16 / 32768 and padded / cut to rate x seconds; the magnitude
    spectrogram: zeros of n_fft / 2 on each side, a symmetric Hann window,
    |rfft| per frame, laid out [bins, frames]; then each clip z-scored
    (population variance, its square terms in float32 as the Engine sums
    them)."""
    from scipy.io import wavfile

    rate, seconds, n_fft, hop = 8000, 0.25, 64, 32
    length = int(rate * seconds)
    root = DATA_DIR / "audio"
    t = np.arange(length) / rate
    clips = []
    for cls, freqs in (("hum", (110.0, 165.0, 220.0)), ("tone", (880.0, 1320.0, 1760.0))):
        (root / cls).mkdir(parents=True, exist_ok=True)
        for k, f0 in enumerate(freqs):
            n = length - 400 * k  # shorter clips are padded
            wave = 0.4 * np.sin(2 * np.pi * f0 * t[:n]) + 0.05 * np.sin(2 * np.pi * 3 * f0 * t[:n])
            pcm = np.round(wave * 32767).astype(np.int16)
            name = f"{cls[0]}{k}.wav"
            wavfile.write(root / cls / name, rate, pcm)
            clips.append((cls, name, pcm))
    clips.sort(key=lambda c: (c[0], c[1]))
    classes = sorted({c[0] for c in clips})
    window = np.array([0.5 * (1 - math.cos(2 * math.pi * i / (n_fft - 1))) for i in range(n_fft)])
    values, labels = {}, {}
    for i, (cls, _, pcm) in enumerate(clips):
        x = np.zeros(length)
        x[: len(pcm)] = pcm.astype(np.float64) / 32768.0
        padded = np.concatenate([np.zeros(n_fft // 2), x, np.zeros(n_fft // 2)])
        frames = (len(padded) - n_fft) // hop + 1
        spec = np.stack([np.abs(np.fft.rfft(padded[f * hop: f * hop + n_fft] * window)) for f in range(frames)], axis=1)
        feat = spec.astype(np.float32).reshape(-1)  # [bins, frames] row-major
        squares = (feat * feat).astype(np.float64)
        mean = float(np.sum(feat.astype(np.float64)) / feat.size)
        var = float(np.sum(squares) / feat.size - mean * mean)
        std = np.float32(math.sqrt(var)) if var > 1e-8 else np.float32(1.0)
        values[str(i)] = ((feat - np.float32(mean)) / std).tolist()
        labels[str(i)] = float(classes.index(cls))
    labels_of = [classes.index(c[0]) for c in clips]
    split = {"train": 0.5, "val": 0.25, "test": 0.25, "stratified": True, "seed": 7}
    partition = assign_split(labels_of, (split["train"], split["val"], split["test"]), True, split["seed"])
    train = [i for i, p in enumerate(partition) if p == 0]
    print(f"audio: {len(clips)} clips, {partition.count(0)} / {partition.count(1)} / {partition.count(2)}")
    write_fixture("audio", {
        "summary": f"{len(clips)} WAV in 2 class folders, {rate} Hz, {seconds} s, spectrogram n_fft {n_fft} hop {hop}",
        "rows": len(clips), "classes": len(classes), "sample_rate": rate, "seconds": seconds, "n_fft": n_fft,
        "hop": hop, "batch_size": 2, "split": split,
        "tolerance": {"atol": 2e-4, "rtol": 2e-4},
        "files": [f"{c[0]}/{c[1]}" for c in clips],
        "partition": partition,
        "epochs": epoch_orders(train),
        "val_rows": [i for i, p in enumerate(partition) if p == 1],
        "test_rows": [i for i, p in enumerate(partition) if p == 2],
        "values": values,
        "labels": labels,
    })


# ----------------------------------------------------------------- sparse --

def sparse_kind() -> None:
    """text.csv through TF-IDF fitted on the Train rows: terms are runs of
    letters and digits of 2 or more, lowercased (no stop words); the cap of
    max_features keeps terms by (corpus count desc, document count desc,
    term), then the features are in term order; idf = ln((n + 1) / (df + 1))
    + 1 with n the Train documents; each value float32(count x idf), then
    each row scaled to unit l2 length. Served dense, one row per document."""
    max_features = 10
    with (DATA_DIR / "text.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    first_seen: dict[str, int] = {}
    class_of_row = [first_seen.setdefault(r["label"], len(first_seen)) for r in rows]
    partition = assign_split(class_of_row, (0.7, 0.15, 0.15), True, 7)
    terms = [[t for t in re.findall(r"[a-z0-9]+", r["text"].lower()) if len(t) >= 2] for r in rows]
    train = [i for i, p in enumerate(partition) if p == 0]
    count: dict[str, int] = {}
    df: dict[str, int] = {}
    for i in train:
        for t in terms[i]:
            count[t] = count.get(t, 0) + 1
        for t in set(terms[i]):
            df[t] = df.get(t, 0) + 1
    kept = sorted(count, key=lambda t: (-count[t], -df[t], t))[:max_features]
    features = sorted(kept)
    n = len(train)
    idf = {t: math.log((n + 1) / (df[t] + 1)) + 1.0 for t in features}
    values, labels = {}, {}
    for i, doc in enumerate(terms):
        v = np.array([np.float32(doc.count(t) * idf[t]) for t in features], dtype=np.float32)
        norm = np.float32(math.sqrt(float(np.sum(v.astype(np.float64) ** 2))))
        values[str(i)] = (v / norm if norm > 0 else v).tolist()
        labels[str(i)] = float(class_of_row[i])
    training = sparse_training(values, class_of_row, partition, epoch_orders(train))
    print(f"sparse: {len(rows)} rows, features {features}")
    write_fixture("sparse", {
        "summary": f"the text CSV as TF-IDF, {max_features} terms from the Train rows, l2 rows",
        "vectorizer": {"text_col": "text", "label_col": "label", "output_format": "sparse", "norm": "l2",
                       "stop_words": "none", "min_df": "1", "max_features": str(max_features)},
        "training_split": {"label_column": "label", "train": 0.7, "val": 0.15, "test": 0.15, "seed": 7,
                           "stratified": True, "shuffle": True},
        "batch_size": 4, "classes": len(first_seen),
        "tolerance": {"atol": 1e-6, "rtol": 1e-5},
        "features": features,
        "partition": partition,
        "epochs": epoch_orders(train),
        "val_rows": [i for i, p in enumerate(partition) if p == 1],
        "test_rows": [i for i, p in enumerate(partition) if p == 2],
        "values": values,
        "labels": labels,
        "training": training,
    })


def sparse_training(values: dict, class_of_row: list[int], partition: list[int], orders: list[list[int]]) -> dict:
    """The sparse model's export (sparse_linear.py, test_oracle_data
    --write-exports) trained by PyTorch on the dense TF-IDF rows, in the
    epochs' orders, as level 2 runs an export."""
    import hashlib

    import torch

    code = (DATA_DIR / "sparse_linear.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    setup, loop = split_export(code)
    torch.manual_seed(22863)
    ns = {"__name__": "__main__"}
    exec(compile(setup, "sparse_linear.py (setup)", "exec"), ns)
    x = torch.tensor([values[str(i)] for i in range(len(values))], dtype=torch.float32)
    y = torch.tensor(class_of_row, dtype=torch.long)
    model = ns["model"]
    model.eval()
    with torch.no_grad():
        model(x[:4])
    model.train()
    torch.save({k: v.detach().clone() for k, v in model.state_dict().items()}, DATA_DIR / "sparse_linear_init.pth")
    val = [i for i, p in enumerate(partition) if p == 1]
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=4,
                                         sampler=EpochOrderSampler(orders), drop_last=False)
    recorder = RecordingLoader(loader, ns, x[val], y[val])
    ns.update({"train_loader": recorder, "num_epochs": EPOCHS})
    exec(compile(loop, "sparse_linear.py (example loop)", "exec"), ns)
    print(f"sparse training: loss {recorder.steps[0]['running_loss']:.6f} -> {recorder.steps[-1]['running_loss']:.6f}")
    return {
        "export_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "init_weights": "sparse_linear_init.pth",
        "tolerance": {"atol": 2e-5, "rtol": 2e-4},
        "steps": recorder.steps,
        "epochs": recorder.epochs,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    text_kind()
    series_kind()
    image_kind()
    audio_kind()
    sparse_kind()


if __name__ == "__main__":
    main()
