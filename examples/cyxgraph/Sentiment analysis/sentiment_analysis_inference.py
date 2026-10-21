#!/usr/bin/env python3
"""
Run embedded inference for the sentiment analysis demo.

Primary mode:
- load metadata from the sentiment prep script
- load the generated vocab file
- encode text the same way the graph expects

Fallback mode:
- if metadata is absent, rebuild a local vocab from the CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


DEFAULT_DATASET = r"D:\demo\mrcj\datasets\sentiment analysis\sentiment_mental_health.csv"
DEFAULT_METADATA = r"D:\demo\mrcj\datasets\sentiment analysis\sentiment_analysis_metadata.json"
DEFAULT_ENDPOINT = "http://localhost:8080/v1/predict"
DEFAULT_HEALTH_URL = "http://localhost:8080/health"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run embedded inference for the sentiment analysis demo."
    )
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA,
        help="Metadata JSON produced by the sentiment prep script",
    )
    parser.add_argument(
        "--csv-file",
        default=DEFAULT_DATASET,
        help="CSV used for fallback vocab building and label discovery",
    )
    parser.add_argument(
        "--text-column",
        default="statement",
        help="CSV column containing the input text",
    )
    parser.add_argument(
        "--label-column",
        default="status",
        help="CSV column containing the label",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Embedded inference endpoint",
    )
    parser.add_argument(
        "--health-url",
        default=DEFAULT_HEALTH_URL,
        help="Health endpoint",
    )
    parser.add_argument(
        "--tokenizer-type",
        choices=["word", "whitespace", "character"],
        default="word",
        help="Tokenizer mode used by the graph",
    )
    parser.add_argument(
        "--lowercase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lowercase the input before tokenization",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Token sequence length expected by the model",
    )
    parser.add_argument(
        "--min-word-freq",
        type=int,
        default=5,
        help="Minimum frequency for tokens to enter the fallback vocabulary",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Vocabulary cap used by the fallback vocabulary builder",
    )
    parser.add_argument("--text", help="Single text sample to classify")
    parser.add_argument(
        "--row-index",
        type=int,
        help="Classify one row from the CSV by 0-based index",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="How many CSV rows to evaluate in batch mode",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many class scores to print",
    )
    return parser.parse_args()


def http_json(url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(v - max_value) for v in values]
    total = sum(exps)
    if total == 0.0:
        return [0.0 for _ in values]
    return [v / total for v in exps]


def tokenize(text: str, tokenizer_type: str, lowercase: bool) -> list[str]:
    text = text.strip()
    if lowercase:
        text = text.lower()
    if tokenizer_type == "character":
        return list(text)
    if tokenizer_type == "whitespace":
        return [token for token in text.split() if token]
    return re.findall(r"[A-Za-z0-9_]+|[^\w\s]", text)


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_label_order(rows: list[dict[str, str]], label_column: str) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for row in rows:
        label = (row.get(label_column) or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def build_vocab(
    rows: list[dict[str, str]],
    text_column: str,
    tokenizer_type: str,
    lowercase: bool,
    min_word_freq: int,
    max_vocab_size: int,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        text = (row.get(text_column) or "").strip()
        if not text:
            continue
        counts.update(tokenize(text, tokenizer_type, lowercase))

    vocab: dict[str, int] = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
    tokens = [
        token
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_word_freq and token not in vocab
    ]
    if max_vocab_size > 0:
        capacity = max(0, max_vocab_size - len(vocab))
        tokens = tokens[:capacity]
    for token in tokens:
        vocab[token] = len(vocab)
    return vocab


def load_vocab(path: Path) -> dict[str, int]:
    vocab: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            token = line.rstrip("\n")
            if token:
                vocab[token] = idx
    return vocab


def encode_text(
    text: str,
    vocab: dict[str, int],
    tokenizer_type: str,
    lowercase: bool,
    max_length: int,
) -> list[float]:
    pad_idx = vocab.get("[PAD]", 0)
    unk_idx = vocab.get("[UNK]", 1)
    ids = [float(vocab.get(token, unk_idx)) for token in tokenize(text, tokenizer_type, lowercase)]
    if len(ids) > max_length:
        ids = ids[:max_length]
    if len(ids) < max_length:
        ids.extend([float(pad_idx)] * (max_length - len(ids)))
    return ids


def label_name(index: int, labels: list[str]) -> str:
    if 0 <= index < len(labels):
        return labels[index]
    return f"class_{index}"


def ensure_server_ready(health_url: str) -> dict:
    try:
        health = http_json(health_url)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach embedded server at {health_url}: {exc}") from exc
    if health.get("status") != "healthy":
        raise RuntimeError(f"Server is not healthy: {health}")
    if not health.get("model_loaded", False):
        raise RuntimeError("Embedded server is running but no model is loaded.")
    return health


def predict_one(
    endpoint: str,
    text: str,
    vocab: dict[str, int],
    tokenizer_type: str,
    lowercase: bool,
    max_length: int,
    labels: list[str],
    top_k: int,
) -> tuple[str, float, list[tuple[int, float]], float]:
    payload = {"input": encode_text(text, vocab, tokenizer_type, lowercase, max_length)}
    start = time.perf_counter()
    result = http_json(endpoint, payload)
    latency_ms = float(result.get("latency_ms", (time.perf_counter() - start) * 1000.0))

    output = result.get("output", [])
    if not isinstance(output, list) or not output:
        raise RuntimeError(f"Unexpected inference response: {result}")

    logits = [float(v) for v in output]
    probs = softmax(logits)
    best_index = max(range(len(probs)), key=probs.__getitem__)
    predicted = label_name(best_index, labels)
    ranked = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[: max(1, top_k)]
    return predicted, probs[best_index], ranked, latency_ms


def main() -> int:
    args = parse_args()
    metadata_path = Path(args.metadata)
    csv_path = Path(args.csv_file)

    try:
        health = ensure_server_ready(args.health_url)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    meta: dict = {}
    vocab: dict[str, int]
    labels: list[str] = []
    tokenizer_type = args.tokenizer_type
    lowercase = args.lowercase
    max_length = args.max_length

    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        meta_vocab = meta.get("vocab_file")
        if isinstance(meta_vocab, str) and meta_vocab and Path(meta_vocab).exists():
            vocab = load_vocab(Path(meta_vocab))
        else:
            print("Metadata file exists but vocab_file is missing; falling back to CSV build.", file=sys.stderr)
            rows = read_rows(csv_path)
            vocab = build_vocab(
                rows,
                str(meta.get("text_column", args.text_column)),
                str(meta.get("tokenizer_type", tokenizer_type)),
                bool(meta.get("lowercase", lowercase)),
                int(meta.get("min_word_freq", args.min_word_freq)),
                int(meta.get("max_vocab_size", args.vocab_size)),
            )

        label_order = meta.get("label_order")
        if isinstance(label_order, list) and all(isinstance(item, str) for item in label_order):
            labels = [item for item in label_order if item]

        tokenizer_type = str(meta.get("tokenizer_type", tokenizer_type))
        lowercase = bool(meta.get("lowercase", lowercase))
        max_length = int(meta.get("max_length", max_length))
    else:
        if not csv_path.exists():
            print(f"CSV file not found: {csv_path}", file=sys.stderr)
            return 1
        rows = read_rows(csv_path)
        if not rows:
            print(f"CSV file is empty: {csv_path}", file=sys.stderr)
            return 1
        vocab = build_vocab(
            rows,
            args.text_column,
            tokenizer_type,
            lowercase,
            args.min_word_freq,
            args.vocab_size,
        )
        labels = load_label_order(rows, args.label_column)

    if not labels and csv_path.exists():
        rows = read_rows(csv_path)
        labels = load_label_order(rows, args.label_column)

    if not labels:
        print("Could not determine label order.", file=sys.stderr)
        return 4

    print(f"Model: {health.get('model_name', 'unknown')}")
    print(f"Metadata: {metadata_path if metadata_path.exists() else 'n/a'}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Labels: {', '.join(labels)}")

    if args.row_index is not None:
        if not csv_path.exists():
            print(f"CSV file not found: {csv_path}", file=sys.stderr)
            return 1
        rows = read_rows(csv_path)
        if args.row_index < 0 or args.row_index >= len(rows):
            print(f"row-index out of range: {args.row_index}", file=sys.stderr)
            return 5
        row = rows[args.row_index]
        text = (row.get(args.text_column) or "").strip()
        actual = (row.get(args.label_column) or "").strip()
        if not text:
            print(f"Row {args.row_index} has empty text", file=sys.stderr)
            return 5
        try:
            predicted, conf, ranked, latency_ms = predict_one(
                args.endpoint,
                text,
                vocab,
                tokenizer_type,
                lowercase,
                max_length,
                labels,
                args.top_k,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"Inference request failed: HTTP {exc.code}\n{body}", file=sys.stderr)
            return 6
        except Exception as exc:
            print(f"Inference failed: {exc}", file=sys.stderr)
            return 6

        print(f"Row: {args.row_index}")
        print(f"Actual: {actual}")
        print(f"Predicted: {predicted}")
        print(f"Confidence: {conf:.4f}")
        print(f"Latency (ms): {latency_ms:.1f}")
        for index, prob in ranked:
            print(f"  {label_name(index, labels)}: prob={prob:.4f}")
        return 0

    if args.text:
        try:
            predicted, conf, ranked, latency_ms = predict_one(
                args.endpoint,
                args.text,
                vocab,
                tokenizer_type,
                lowercase,
                max_length,
                labels,
                args.top_k,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"Inference request failed: HTTP {exc.code}\n{body}", file=sys.stderr)
            return 6
        except Exception as exc:
            print(f"Inference failed: {exc}", file=sys.stderr)
            return 6

        print(f"Predicted label: {predicted}")
        print(f"Confidence: {conf:.4f}")
        print(f"Latency (ms): {latency_ms:.1f}")
        for index, prob in ranked:
            print(f"  {label_name(index, labels)}: prob={prob:.4f}")
        return 0

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    rows = read_rows(csv_path)
    correct = 0
    total = 0
    for row in rows:
        if total >= args.samples:
            break
        text = (row.get(args.text_column) or "").strip()
        actual = (row.get(args.label_column) or "").strip()
        if not text:
            continue
        try:
            predicted, conf, ranked, latency_ms = predict_one(
                args.endpoint,
                text,
                vocab,
                tokenizer_type,
                lowercase,
                max_length,
                labels,
                args.top_k,
            )
        except Exception as exc:
            print(f"Sample {total + 1}: error={exc}", file=sys.stderr)
            continue

        total += 1
        correct += int(predicted == actual)
        snippet = text.replace("\n", " ")[:100]
        print(
            f"{total:03d} actual={actual} predicted={predicted} "
            f"conf={conf:.4f} latency_ms={latency_ms:.1f} ok={predicted == actual} "
            f"text={snippet!r}"
        )
        print(
            "     top-k:",
            ", ".join(f"{label_name(index, labels)}={prob:.4f}" for index, prob in ranked),
        )

    if total > 0:
        print(f"Accuracy: {correct}/{total} = {correct / total:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
