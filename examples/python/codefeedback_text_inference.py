#!/usr/bin/env python3
"""
Run a CyxWiz text-classification model against the embedded inference server.

The script tokenizes with the same tokenizer + vocab-file convention used by
the demo prep script. It can run a single prompt or evaluate rows from the
prepared CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run embedded inference for the CodeFeedback text demo."
    )
    parser.add_argument(
        "--metadata",
        default=r"D:\Dev\DataSet_List\datasets--m-a-p--CodeFeedback-Filtered-Instruction\data\codefeedback_metadata.json",
        help="Metadata JSON produced by the prep script",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8080/v1/predict",
        help="Embedded inference endpoint",
    )
    parser.add_argument(
        "--health-url",
        default="http://localhost:8080/health",
        help="Health endpoint",
    )
    parser.add_argument("--text", help="Raw text to classify")
    parser.add_argument("--query", help="Query text to combine with --answer")
    parser.add_argument("--answer", help="Answer text to combine with --query")
    parser.add_argument("--expected-label", help="Optional expected label for single-sample mode")
    parser.add_argument(
        "--csv-file",
        help="Prepared CSV to evaluate in batch mode",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of CSV rows to test in batch mode",
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


def load_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_vocab(path: Path) -> dict[str, int]:
    vocab: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            token = line.rstrip("\n")
            if token:
                vocab[token] = idx
    return vocab


def build_input_text(query: str, answer: str, text_mode: str) -> str:
    query = query.strip()
    answer = answer.strip()

    if text_mode == "answer_only":
        return answer or query
    if text_mode == "query_only":
        return query or answer

    parts = ["[q]", query]
    if answer:
        parts.extend(["[a]", answer])
    return " ".join(part for part in parts if part)


def load_label_order(csv_path: Path, label_column: str = "label") -> list[str]:
    """
    Reconstruct the class order the engine will use when it loads the CSV.

    TextDataset::LoadCSV assigns string labels on first appearance, so a shuffled
    CSV does not imply alphabetical or frequency order. This helper mirrors that
    behavior so inference decodes logits with the correct label names.
    """
    label_order: list[str] = []
    seen: set[str] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = (row.get(label_column) or "").strip()
            if not label or label in seen:
                continue
            seen.add(label)
            label_order.append(label)

    return label_order


def encode_text(
    text: str,
    vocab: dict[str, int],
    max_length: int,
    lowercase: bool = True,
    tokenizer_type: str = "whitespace",
) -> list[float]:
    if lowercase:
        text = text.lower()
    if tokenizer_type == "character":
        tokens = list(text)
    else:
        tokens = text.split()
    pad_idx = vocab.get("[PAD]", 0)
    unk_idx = vocab.get("[UNK]", 1)

    ids = [float(vocab.get(token, unk_idx)) for token in tokens]
    if len(ids) > max_length:
        ids = ids[:max_length]
    if len(ids) < max_length:
        ids.extend([float(pad_idx)] * (max_length - len(ids)))
    return ids


def label_name(index: int, labels: list[str]) -> str:
    if 0 <= index < len(labels):
        return labels[index]
    return f"class_{index}"


def infer_one(endpoint: str, text: str, vocab: dict[str, int], max_length: int, labels: list[str],
              top_k: int, tokenizer_type: str = "whitespace") -> tuple[str, float, list[float], float]:
    payload = {"input": encode_text(text, vocab, max_length, tokenizer_type=tokenizer_type)}
    start = time.perf_counter()
    result = http_json(endpoint, payload)
    latency_ms = float(result.get("latency_ms", (time.perf_counter() - start) * 1000.0))

    output = result.get("output", [])
    if not isinstance(output, list) or not output:
        raise RuntimeError(f"Unexpected inference response: {result}")

    logits = [float(v) for v in output]
    probs = softmax(logits)
    best_index = max(range(len(probs)), key=probs.__getitem__)
    pred = label_name(best_index, labels)

    ranked = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[: max(1, top_k)]
    return pred, probs[best_index], [logits[i] for i, _ in ranked], latency_ms


def main() -> int:
    args = parse_args()
    metadata_path = Path(args.metadata)
    meta = load_metadata(metadata_path)
    vocab_path = Path(meta["vocab_file"])
    vocab = load_vocab(vocab_path)
    labels: list[str] = []
    meta_label_order = meta.get("label_order")
    if isinstance(meta_label_order, list) and all(isinstance(item, str) for item in meta_label_order):
        labels = [item for item in meta_label_order if item]
    csv_label_source = meta.get("csv_file")
    if not labels and isinstance(csv_label_source, str) and csv_label_source:
        csv_path = Path(csv_label_source)
        if csv_path.exists():
            labels = load_label_order(csv_path)
    if not labels:
        labels = list(meta.get("selected_labels", []))
    max_length = int(meta.get("max_length", 192))
    lowercase = bool(meta.get("lowercase", True))
    tokenizer_type = str(meta.get("tokenizer_type", "whitespace"))
    text_mode = str(meta.get("text_mode", "query_answer"))

    try:
        health = http_json(args.health_url)
    except urllib.error.URLError as exc:
        print(f"Failed to reach embedded server at {args.health_url}: {exc}", file=sys.stderr)
        return 2
    if health.get("status") != "healthy":
        print(f"Server is not healthy: {health}", file=sys.stderr)
        return 3
    if not health.get("model_loaded", False):
        print("Embedded server is running but no model is loaded.", file=sys.stderr)
        return 4

    if args.csv_file:
        csv_path = Path(args.csv_file)
        if not csv_path.exists():
            print(f"CSV file not found: {csv_path}", file=sys.stderr)
            return 1
        correct = 0
        total = 0
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if total >= args.num_samples:
                    break
                text = row.get("text", "")
                actual = row.get("label", "")
                if not text:
                    continue
                try:
                    text = build_input_text(query, answer, text_mode)
                    pred, conf, _, latency_ms = infer_one(
                        args.endpoint, text, vocab, max_length, labels, args.top_k, tokenizer_type
                    )
                except Exception as exc:
                    print(f"Sample {total + 1}: error={exc}", file=sys.stderr)
                    continue
                ok = (pred == actual)
                correct += int(ok)
                total += 1
                snippet = text[:80].replace("\n", " ")
                print(
                    f"{total:03d} actual={actual} predicted={pred} "
                    f"conf={conf:.4f} latency_ms={latency_ms:.1f} "
                    f"ok={ok} text={snippet!r}"
                )
        if total > 0:
            print(f"Accuracy: {correct}/{total} = {correct / total:.2%}")
        return 0

    if args.text:
        text = args.text
    elif args.query is not None or args.answer is not None:
        query = args.query or ""
        answer = args.answer or ""
        text = build_input_text(query, answer, text_mode)
    else:
        print("Provide either --text or --csv-file.", file=sys.stderr)
        return 1

    try:
        pred, conf, _, latency_ms = infer_one(
            args.endpoint, text, vocab, max_length, labels, args.top_k, tokenizer_type
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"Inference request failed: HTTP {exc.code}\n{body}", file=sys.stderr)
        return 5
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 5

    print(f"Model: {health.get('model_name', 'unknown')}")
    print(f"Predicted label: {pred}")
    if args.expected_label:
        print(f"Expected label: {args.expected_label}")
        print(f"Correct: {pred == args.expected_label}")
    print(f"Confidence: {conf:.4f}")
    print(f"Latency (ms): {latency_ms:.1f}")
    if tokenizer_type == "character":
        token_count = len(text.lower() if lowercase else text)
    else:
        token_count = len(text.lower().split()) if lowercase else len(text.split())
    print(f"Input token count: {token_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
