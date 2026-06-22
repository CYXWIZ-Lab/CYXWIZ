#!/usr/bin/env python3
"""
Prepare the CoNLL-style NER CSV for the CyxWiz NER graph.

Input rows:
    Sentence #,Word,POS,Tag
    Sentence: 1,Thousands,NNS,O
    ,of,IN,O

Output rows:
    sentence_id,tokens,pos_tags,ner_tags
    1,"Thousands of ...","NNS IN ...","O O ..."
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path


SPECIAL_TOKENS = ["[PAD]", "[UNK]"]
EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = EXAMPLE_DIR / "sample_ner.csv"
DEFAULT_OUT_DIR = EXAMPLE_DIR / "generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CyxWiz NER demo files.")
    parser.add_argument("--csv-file", default=DEFAULT_SOURCE, help="Source NER CSV file")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--min-word-freq", type=int, default=2, help="Minimum word frequency")
    parser.add_argument("--max-word-vocab-size", type=int, default=30000, help="Word vocab cap")
    parser.add_argument("--max-length", type=int, default=96, help="Max tokens per sentence")
    parser.add_argument(
        "--encoding",
        default="cp1252",
        help="Source CSV encoding. The provided dataset contains Windows-encoded bytes.",
    )
    parser.add_argument(
        "--lowercase",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Lowercase tokens for the word vocabulary",
    )
    return parser.parse_args()


def sentence_number(value: str, fallback: int) -> str:
    value = (value or "").strip()
    if not value:
        return str(fallback)
    match = re.search(r"(\d+)", value)
    return match.group(1) if match else value


def read_sentences(csv_path: Path, encoding: str) -> list[dict[str, list[str] | str]]:
    sentences: list[dict[str, list[str] | str]] = []
    current: dict[str, list[str] | str] | None = None
    sentence_count = 0

    with csv_path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Sentence #", "Word", "POS", "Tag"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            marker = (row.get("Sentence #") or "").strip()
            word = (row.get("Word") or "").strip()
            pos = (row.get("POS") or "").strip()
            tag = (row.get("Tag") or "").strip()

            if marker:
                if current and current["tokens"]:
                    sentences.append(current)
                sentence_count += 1
                current = {
                    "sentence_id": sentence_number(marker, sentence_count),
                    "tokens": [],
                    "pos_tags": [],
                    "ner_tags": [],
                }

            if current is None:
                sentence_count += 1
                current = {
                    "sentence_id": str(sentence_count),
                    "tokens": [],
                    "pos_tags": [],
                    "ner_tags": [],
                }

            if not word:
                continue

            current["tokens"].append(word)
            current["pos_tags"].append(pos or "[UNK]")
            current["ner_tags"].append(tag or "O")

    if current and current["tokens"]:
        sentences.append(current)

    return sentences


def write_sentence_csv(sentences: list[dict[str, list[str] | str]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sentence_id", "tokens", "pos_tags", "ner_tags"],
        )
        writer.writeheader()
        for sentence in sentences:
            writer.writerow(
                {
                    "sentence_id": sentence["sentence_id"],
                    "tokens": " ".join(sentence["tokens"]),
                    "pos_tags": " ".join(sentence["pos_tags"]),
                    "ner_tags": " ".join(sentence["ner_tags"]),
                }
            )


def write_vocab(tokens: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for token in tokens:
            handle.write(token + "\n")


def metadata_relative_path(path: Path, base_dir: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), base_dir.resolve())).as_posix()
    except ValueError:
        return str(path.resolve())


def build_word_vocab(
    sentences: list[dict[str, list[str] | str]],
    min_freq: int,
    max_size: int,
    lowercase: bool,
) -> list[str]:
    counts: Counter[str] = Counter()
    for sentence in sentences:
        for token in sentence["tokens"]:
            counts[token.lower() if lowercase else token] += 1

    words = [
        token
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_freq and token not in SPECIAL_TOKENS
    ]

    capacity = max(0, max_size - len(SPECIAL_TOKENS))
    return SPECIAL_TOKENS + words[:capacity]


def build_simple_vocab(sentences: list[dict[str, list[str] | str]], key: str) -> list[str]:
    seen: set[str] = set(SPECIAL_TOKENS)
    values: list[str] = list(SPECIAL_TOKENS)
    for sentence in sentences:
        for token in sentence[key]:
            if token not in seen:
                seen.add(token)
                values.append(token)
    return values


def tag_counts(sentences: list[dict[str, list[str] | str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for sentence in sentences:
        counts.update(sentence["ner_tags"])
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def main() -> int:
    args = parse_args()
    source = Path(args.csv_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise SystemExit(f"CSV file not found: {source}")

    sentences = read_sentences(source, args.encoding)
    if not sentences:
        raise SystemExit(f"No sentences found in {source}")

    sentence_csv = out_dir / "ner_sentences.csv"
    word_vocab_path = out_dir / "ner_word_vocab.txt"
    pos_vocab_path = out_dir / "ner_pos_vocab.txt"
    tag_vocab_path = out_dir / "ner_tag_vocab.txt"
    metadata_path = out_dir / "ner_metadata.json"

    word_vocab = build_word_vocab(
        sentences,
        min_freq=args.min_word_freq,
        max_size=args.max_word_vocab_size,
        lowercase=args.lowercase,
    )
    pos_vocab = build_simple_vocab(sentences, "pos_tags")
    tag_vocab = build_simple_vocab(sentences, "ner_tags")

    write_sentence_csv(sentences, sentence_csv)
    write_vocab(word_vocab, word_vocab_path)
    write_vocab(pos_vocab, pos_vocab_path)
    write_vocab(tag_vocab, tag_vocab_path)

    lengths = [len(sentence["tokens"]) for sentence in sentences]
    metadata = {
        "source_csv": metadata_relative_path(source, out_dir),
        "sentence_csv": metadata_relative_path(sentence_csv, out_dir),
        "word_vocab_file": metadata_relative_path(word_vocab_path, out_dir),
        "pos_vocab_file": metadata_relative_path(pos_vocab_path, out_dir),
        "tag_vocab_file": metadata_relative_path(tag_vocab_path, out_dir),
        "num_sentences": len(sentences),
        "num_tokens": sum(lengths),
        "max_sentence_length_in_source": max(lengths),
        "configured_max_length": args.max_length,
        "word_vocab_size": len(word_vocab),
        "pos_vocab_size": len(pos_vocab),
        "tag_vocab_size": len(tag_vocab),
        "tag_order": tag_vocab,
        "tag_counts": tag_counts(sentences),
        "lowercase": args.lowercase,
        "source_encoding": args.encoding,
        "min_word_freq": args.min_word_freq,
        "max_word_vocab_size": args.max_word_vocab_size,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Source: {source}")
    print(f"Sentences: {metadata['num_sentences']}")
    print(f"Tokens: {metadata['num_tokens']}")
    print(f"Max sentence length: {metadata['max_sentence_length_in_source']}")
    print(f"Word vocab size: {metadata['word_vocab_size']}")
    print(f"POS vocab size: {metadata['pos_vocab_size']}")
    print(f"Tag vocab size: {metadata['tag_vocab_size']}")
    print(f"Wrote: {sentence_csv}")
    print(f"Wrote: {word_vocab_path}")
    print(f"Wrote: {pos_vocab_path}")
    print(f"Wrote: {tag_vocab_path}")
    print(f"Wrote: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
