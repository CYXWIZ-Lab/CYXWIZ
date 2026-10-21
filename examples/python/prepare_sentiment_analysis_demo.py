#!/usr/bin/env python3
"""
Prepare a CyxWiz-friendly sentiment analysis vocab for the mental-health CSV.

This mirrors the CodeFeedback prep flow:
- read the source CSV
- build a vocabulary from the training text
- write a vocab file compatible with cyxwiz::Vocabulary::LoadFromFile
- write metadata that the inference helper can consume

The engine graph can then point its TextVocabulary node at the generated
vocab file so the backend reads the same token-to-index mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
DEFAULT_SOURCE = r"D:\demo\mrcj\datasets\sentiment analysis\sentiment_mental_health.csv"
DEFAULT_OUT_DIR = r"D:\demo\mrcj\datasets\sentiment analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a vocab file and metadata for the sentiment demo."
    )
    parser.add_argument("--csv-file", default=DEFAULT_SOURCE, help="Source CSV file")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory for vocab and metadata",
    )
    parser.add_argument("--text-column", default="statement", help="Text column name")
    parser.add_argument("--label-column", default="status", help="Label column name")
    parser.add_argument(
        "--tokenizer-type",
        choices=["word", "whitespace", "character"],
        default="word",
        help="Tokenizer mode to mirror in the graph",
    )
    parser.add_argument(
        "--lowercase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lowercase text before tokenization",
    )
    parser.add_argument(
        "--min-word-freq",
        type=int,
        default=5,
        help="Minimum token frequency to keep",
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=10000,
        help="Maximum vocab size including special tokens",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Sequence length used by the graph",
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def tokenize(text: str, tokenizer_type: str, lowercase: bool) -> list[str]:
    text = text.strip()
    if lowercase:
        text = text.lower()
    if tokenizer_type == "character":
        return list(text)
    if tokenizer_type == "whitespace":
        return [token for token in text.split() if token]
    return re.findall(r"[A-Za-z0-9_]+|[^\w\s]", text)


def build_vocab(
    rows: list[dict[str, str]],
    text_column: str,
    tokenizer_type: str,
    lowercase: bool,
    min_word_freq: int,
    max_vocab_size: int,
) -> list[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        text = (row.get(text_column) or "").strip()
        if not text:
            continue
        counts.update(tokenize(text, tokenizer_type, lowercase))

    tokens = [
        token
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_word_freq and token not in SPECIAL_TOKENS
    ]

    if max_vocab_size > 0:
        capacity = max(0, max_vocab_size - len(SPECIAL_TOKENS))
        tokens = tokens[:capacity]

    return SPECIAL_TOKENS + tokens


def label_order(rows: list[dict[str, str]], label_column: str) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for row in rows:
        label = (row.get(label_column) or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def main() -> int:
    args = parse_args()
    source = Path(args.csv_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise SystemExit(f"CSV file not found: {source}")

    rows = read_rows(source)
    if not rows:
        raise SystemExit(f"No rows found in {source}")

    labels = label_order(rows, args.label_column)
    if not labels:
        raise SystemExit(f"No labels found in column '{args.label_column}'")

    vocab_tokens = build_vocab(
        rows,
        args.text_column,
        args.tokenizer_type,
        args.lowercase,
        args.min_word_freq,
        args.max_vocab_size,
    )

    vocab_path = out_dir / "sentiment_analysis_vocab.txt"
    metadata_path = out_dir / "sentiment_analysis_metadata.json"

    with vocab_path.open("w", encoding="utf-8", newline="\n") as handle:
        for token in vocab_tokens:
            handle.write(token + "\n")

    metadata = {
        "csv_file": str(source.resolve()),
        "vocab_file": str(vocab_path.resolve()),
        "text_column": args.text_column,
        "label_column": args.label_column,
        "label_order": labels,
        "num_classes": len(labels),
        "tokenizer_type": args.tokenizer_type,
        "lowercase": args.lowercase,
        "min_word_freq": args.min_word_freq,
        "max_vocab_size": args.max_vocab_size,
        "max_length": args.max_length,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Source: {source}")
    print(f"Wrote: {vocab_path}")
    print(f"Wrote: {metadata_path}")
    print(f"Labels: {', '.join(labels)}")
    print(f"Vocab size: {len(vocab_tokens)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
