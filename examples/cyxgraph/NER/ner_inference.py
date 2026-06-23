#!/usr/bin/env python3
"""
Run inference for the CyxWiz NER sequence-tagging demo.

This helper mirrors the preprocessing from prepare_ner_demo.py:
- loads word, POS, and tag vocabularies from metadata
- encodes one sentence into padded word/POS ID sequences
- sends the payload to the embedded inference endpoint
- decodes per-token BIO tag predictions

Use --dry-run to inspect the encoded payload without requiring a running
embedded server. This is useful while the sequence-tagging backend nodes
are still being implemented.
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


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_METADATA = EXAMPLE_DIR / "generated" / "ner_metadata.json"
DEFAULT_ENDPOINT = "http://localhost:8080/v1/predict"
DEFAULT_HEALTH_URL = "http://localhost:8080/health"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NER sequence-tagging inference.")
    parser.add_argument("--metadata", default=DEFAULT_METADATA, help="NER metadata JSON")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Embedded inference endpoint")
    parser.add_argument("--health-url", default=DEFAULT_HEALTH_URL, help="Embedded health endpoint")
    parser.add_argument("--sentence", help="Raw sentence to tag")
    parser.add_argument(
        "--tokens",
        nargs="+",
        help="Pre-tokenized sentence. Overrides --sentence when provided.",
    )
    parser.add_argument(
        "--pos-tags",
        nargs="+",
        help="Optional POS tags aligned with --tokens. Unknown/missing POS tags map to [UNK].",
    )
    parser.add_argument(
        "--csv-file",
        help="Prepared ner_sentences.csv for batch evaluation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Rows to evaluate from --csv-file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print encoded payload instead of calling the server.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Print top-k tags per token when logits are available.",
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_vocab(path: Path) -> dict[str, int]:
    vocab: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            token = line.rstrip("\n")
            if token:
                vocab[token] = idx
    return vocab


def resolve_metadata_path(metadata_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else metadata_path.parent / path


def invert_vocab(vocab: dict[str, int]) -> list[str]:
    labels = [""] * (max(vocab.values()) + 1 if vocab else 0)
    for token, idx in vocab.items():
        if idx >= 0:
            labels[idx] = token
    return labels


def simple_tokenize(sentence: str) -> list[str]:
    # The NER dataset is whitespace-tokenized after preparation.
    return [token for token in sentence.strip().split() if token]


def pad(values: list[int], max_length: int, pad_value: int) -> list[int]:
    values = values[:max_length]
    if len(values) < max_length:
        values = values + [pad_value] * (max_length - len(values))
    return values


def encode_sequence(
    tokens: list[str],
    pos_tags: list[str],
    word_vocab: dict[str, int],
    pos_vocab: dict[str, int],
    max_length: int,
) -> tuple[list[int], list[int], list[int], list[str]]:
    pad_word = word_vocab.get("[PAD]", 0)
    unk_word = word_vocab.get("[UNK]", 1)
    pad_pos = pos_vocab.get("[PAD]", 0)
    unk_pos = pos_vocab.get("[UNK]", 1)

    visible_tokens = tokens[:max_length]
    word_ids = [word_vocab.get(token, unk_word) for token in visible_tokens]
    pos_ids = [
        pos_vocab.get(pos_tags[i], unk_pos) if i < len(pos_tags) else unk_pos
        for i in range(len(visible_tokens))
    ]
    attention_mask = [1] * len(visible_tokens)

    return (
        pad(word_ids, max_length, pad_word),
        pad(pos_ids, max_length, pad_pos),
        pad(attention_mask, max_length, 0),
        visible_tokens,
    )


def build_payload(
    word_ids: list[int],
    pos_ids: list[int],
    attention_mask: list[int],
    sequence_length: int,
) -> dict:
    # Sequence-based NER inference expects named tensors in `input`.
    return {
        "input": {
            "word_ids": word_ids,
            "pos_ids": pos_ids,
            "attention_mask": attention_mask,
            # Optional length metadata helps confirm batch/sequence framing.
            "sequence_lengths": [sequence_length],
        }
    }


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps] if total else [0.0 for _ in values]


def tag_name(index: int, labels: list[str]) -> str:
    if 0 <= index < len(labels) and labels[index]:
        return labels[index]
    return f"tag_{index}"


def normalize_logits(output: object, max_length: int, num_tags: int) -> list[list[float]]:
    if isinstance(output, dict):
        output = output.get("logits", output.get("output", []))

    if not isinstance(output, list):
        raise RuntimeError(f"Unexpected model output type: {type(output).__name__}")

    if output and all(isinstance(row, list) for row in output):
        return [[float(value) for value in row] for row in output[:max_length]]

    flat = [float(value) for value in output]
    if num_tags <= 0 or len(flat) % num_tags != 0:
        raise RuntimeError(
            f"Cannot reshape output of length {len(flat)} into token logits with {num_tags} tags"
        )
    rows = len(flat) // num_tags
    return [flat[i * num_tags:(i + 1) * num_tags] for i in range(min(rows, max_length))]


def decode_predictions(
    result: dict,
    tokens: list[str],
    labels: list[str],
    max_length: int,
    top_k: int,
) -> list[dict[str, object]]:
    sequence_payload = result.get("sequence")
    if isinstance(sequence_payload, dict) and "tag_labels" in sequence_payload:
        tag_labels = sequence_payload["tag_labels"]
        if isinstance(tag_labels, list) and tag_labels:
            if all(isinstance(row, list) for row in tag_labels):
                # Sequence responses are [batch, seq]. Prefer first decoded row.
                if isinstance(tag_labels[0], list) and len(tag_labels[0]) > 0:
                    tag_labels = tag_labels[0]
            decoded: list[dict[str, object]] = []
            for i, token in enumerate(tokens):
                if i >= len(tag_labels) or i >= max_length:
                    break
                decoded.append(
                    {
                        "token": token,
                        "tag": tag_labels[i] or tag_name(0, labels),
                        "confidence": 1.0,
                        "top": [],
                    }
                )
            if decoded:
                return decoded

    output = result.get("output", result.get("logits", []))
    logits_by_token = normalize_logits(output, max_length=max_length, num_tags=len(labels))

    decoded: list[dict[str, object]] = []
    for i, token in enumerate(tokens):
        if i >= len(logits_by_token):
            break
        logits = logits_by_token[i]
        probs = softmax(logits)
        ranked = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)
        best_idx, best_prob = ranked[0]
        decoded.append(
            {
                "token": token,
                "tag": tag_name(best_idx, labels),
                "confidence": best_prob,
                "top": [
                    {"tag": tag_name(idx, labels), "probability": prob}
                    for idx, prob in ranked[: max(1, top_k)]
                ],
            }
        )
    return decoded


def ensure_server_ready(health_url: str) -> None:
    try:
        health = http_json(health_url)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach embedded server at {health_url}: {exc}") from exc
    if health.get("status") != "healthy":
        raise RuntimeError(f"Server is not healthy: {health}")
    if not health.get("model_loaded", False):
        raise RuntimeError("Embedded server is running but no model is loaded.")


def run_one(
    endpoint: str,
    tokens: list[str],
    pos_tags: list[str],
    word_vocab: dict[str, int],
    pos_vocab: dict[str, int],
    tag_labels: list[str],
    max_length: int,
    top_k: int,
    dry_run: bool,
) -> list[dict[str, object]]:
    word_ids, pos_ids, attention_mask, visible_tokens = encode_sequence(
        tokens=tokens,
        pos_tags=pos_tags,
        word_vocab=word_vocab,
        pos_vocab=pos_vocab,
        max_length=max_length,
    )
    payload = build_payload(
        word_ids=word_ids,
        pos_ids=pos_ids,
        attention_mask=attention_mask,
        sequence_length=len(visible_tokens),
    )

    if dry_run:
        print(json.dumps({"tokens": visible_tokens, "payload": payload}, indent=2))
        return []

    start = time.perf_counter()
    result = http_json(endpoint, payload)
    latency_ms = float(result.get("latency_ms", (time.perf_counter() - start) * 1000.0))
    decoded = decode_predictions(result, visible_tokens, tag_labels, max_length, top_k)
    print(f"latency_ms={latency_ms:.1f}")
    return decoded


def print_decoded(decoded: list[dict[str, object]], expected_tags: list[str] | None = None) -> int:
    correct = 0
    total = 0
    for i, item in enumerate(decoded):
        expected = expected_tags[i] if expected_tags and i < len(expected_tags) else ""
        ok = expected == item["tag"] if expected else None
        if expected:
            total += 1
            correct += int(bool(ok))
        suffix = f" expected={expected} ok={ok}" if expected else ""
        print(
            f"{i:03d} token={item['token']} tag={item['tag']} "
            f"conf={float(item['confidence']):.4f}{suffix}"
        )
    return correct if total else -1


def main() -> int:
    args = parse_args()
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"Metadata not found: {metadata_path}", file=sys.stderr)
        print("Run prepare_ner_demo.py first.", file=sys.stderr)
        return 1

    metadata = load_json(metadata_path)
    word_vocab = load_vocab(resolve_metadata_path(metadata_path, metadata["word_vocab_file"]))
    pos_vocab = load_vocab(resolve_metadata_path(metadata_path, metadata["pos_vocab_file"]))
    tag_vocab = load_vocab(resolve_metadata_path(metadata_path, metadata["tag_vocab_file"]))
    tag_labels = invert_vocab(tag_vocab)
    max_length = int(metadata.get("configured_max_length", 96))

    if not args.dry_run:
        try:
            ensure_server_ready(args.health_url)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if args.csv_file:
        csv_path = Path(args.csv_file)
        if not csv_path.exists():
            print(f"CSV file not found: {csv_path}", file=sys.stderr)
            return 1
        total = 0
        correct = 0
        evaluated_tokens = 0
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if total >= args.num_samples:
                    break
                tokens = simple_tokenize(row.get("tokens", ""))
                pos_tags = simple_tokenize(row.get("pos_tags", ""))
                expected = simple_tokenize(row.get("ner_tags", ""))
                if not tokens:
                    continue
                print(f"\nSentence {row.get('sentence_id', total + 1)}")
                decoded = run_one(
                    args.endpoint,
                    tokens,
                    pos_tags,
                    word_vocab,
                    pos_vocab,
                    tag_labels,
                    max_length,
                    args.top_k,
                    args.dry_run,
                )
                if decoded:
                    sentence_correct = print_decoded(decoded, expected)
                    if sentence_correct >= 0:
                        correct += sentence_correct
                        evaluated_tokens += min(len(decoded), len(expected))
                total += 1
        if evaluated_tokens:
            print(f"\nToken accuracy: {correct}/{evaluated_tokens} = {correct / evaluated_tokens:.4f}")
        return 0

    if args.tokens:
        tokens = args.tokens
    elif args.sentence:
        tokens = simple_tokenize(args.sentence)
    else:
        print("Provide --sentence, --tokens, or --csv-file.", file=sys.stderr)
        return 1

    pos_tags = args.pos_tags or []
    decoded = run_one(
        args.endpoint,
        tokens,
        pos_tags,
        word_vocab,
        pos_vocab,
        tag_labels,
        max_length,
        args.top_k,
        args.dry_run,
    )
    if decoded:
        print_decoded(decoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
