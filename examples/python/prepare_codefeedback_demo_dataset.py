#!/usr/bin/env python3
"""
Prepare a small CyxWiz text-classification demo from the CodeFeedback blobs.

This does not create a true LLM dataset. It builds a reproducible supervised
demo that CyxWiz can train today:
- input text: code-bearing answer text by default
- label: the `lang` field
- tokenizer: character by default
- vocab file: one token per line, compatible with CyxWiz Vocabulary::LoadFromFile

The output files are:
- codefeedback_lang_multiclass.csv
- codefeedback_vocab.txt
- codefeedback_metadata.json
- codefeedback_lang_classifier_multi.cyxgraph
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import random
from typing import Iterable


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
LABEL_ALIASES = {
    "asm": "assembly",
    "as3": "actionscript",
    "bat": "batch",
    "batchfile": "batch",
    "cmd": "batch",
    "cs": "csharp",
    "docker": "dockerfile",
    "f#": "fsharp",
    "html5": "html",
    "hs": "haskell",
    "javasript": "javascript",
    "js": "javascript",
    "jsx": "javascript",
    "mongo": "mongodb",
    "objc": "objective-c",
    "plain": "plaintext",
    "powershellscript": "powershell",
    "proto": "protobuf",
    "py": "python",
    "rb": "ruby",
    "rs": "rust",
    "shellscript": "shell",
    "text": "plaintext",
    "tsx": "typescript",
    "txt": "plaintext",
    "vbnet": "vb",
    "vuejs": "vue",
    "xmlhtml": "xml",
    "yml": "yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a CyxWiz-friendly text demo from CodeFeedback blobs."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Blob directory or a single JSONL file",
    )
    parser.add_argument(
        "--out-dir",
        default=r"D:\Dev\CyxWiz_Claude\examples\data\codefeedback_demo",
        help="Output directory for CSV, vocab, and metadata",
    )
    parser.add_argument("--query-field", default="query", help="JSON field for the prompt")
    parser.add_argument("--answer-field", default="answer", help="JSON field for the answer")
    parser.add_argument("--label-field", default="lang", help="JSON field for the label")
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=5000,
        help="Cap rows per selected label",
    )
    parser.add_argument(
        "--top-k-labels",
        type=int,
        default=12,
        help="Keep only the most common normalized labels (0 keeps all labels)",
    )
    parser.add_argument(
        "--min-label-count",
        type=int,
        default=200,
        help="Drop normalized labels with fewer than this many raw examples",
    )
    parser.add_argument(
        "--balance-mode",
        choices=["none", "oversample", "downsample"],
        default="downsample",
        help=(
            "How to rebalance the output CSV. 'oversample' repeats minority "
            "classes up to --balance-target, 'downsample' trims majority "
            "classes down to --balance-target, and 'none' keeps the raw cap."
        ),
    )
    parser.add_argument(
        "--balance-target",
        type=int,
        default=1200,
        help="Target rows per label when balance-mode is oversample or downsample",
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=12000,
        help="Total vocab size including special tokens",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Training sequence length to record in metadata",
    )
    parser.add_argument(
        "--tokenizer-type",
        choices=["whitespace", "character"],
        default="character",
        help="Tokenizer to build into the demo graph and metadata",
    )
    parser.add_argument(
        "--text-mode",
        choices=["answer_only", "query_answer", "query_only"],
        default="answer_only",
        help="Which text fields to concatenate into each training sample",
    )
    return parser.parse_args()


def normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    return LABEL_ALIASES.get(normalized, normalized)


def iter_json_records(source: Path) -> Iterable[dict]:
    files: list[Path]
    if source.is_file():
        files = [source]
    else:
        files = [p for p in source.rglob("*") if p.is_file()]

    for path in files:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line[0] not in "{[":
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        except OSError:
            continue


def build_text(query: str, answer: str, text_mode: str) -> str:
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


def tokenize_text(text: str, tokenizer_type: str, lowercase: bool = True) -> list[str]:
    if lowercase:
        text = text.lower()
    if tokenizer_type == "character":
        return list(text)
    return text.split()


def build_graph_template(
    csv_path: str,
    vocab_path: str,
    num_classes: int,
    max_length: int,
    max_vocab_size: int,
    tokenizer_type: str,
) -> dict:
    tokenizer_id = "2" if tokenizer_type == "character" else "0"
    tokenizer_name = "Tokenizer (character)" if tokenizer_type == "character" else "Tokenizer (whitespace)"
    return {
        "id": "codefeedback_lang_classifier_text",
        "name": "CodeFeedback Lang Classifier - Multi-Class Text Demo",
        "description": (
            "Supported CyxWiz text demo for the CodeFeedback blobs. "
            "This is not a transformer LLM. It is a supervised text "
            f"classifier using {tokenizer_type} tokenization, a fixed vocab file, "
            "and Embedding -> GRU -> Dense. The demo is configured as a "
            "multi-class lang classifier over all selected labels."
        ),
        "category": "Training Pipeline",
        "tags": ["text", "nlp", "code", "classification", "demo", "gru"],
        "parameters": [],
        "template": {
            "nodes": [
                {
                    "id": "data_input",
                    "type": "DataInput",
                    "name": "Data Input",
                    "pos": [0, 300],
                    "params": {
                        "source_type": "file",
                        "file_category": "text",
                        "file_path": csv_path,
                        "text_column": "text",
                        "text_label_column": "label",
                        "text_tokenizer_type": tokenizer_id,
                        "text_max_length": str(max_length),
                        "text_lowercase": "true",
                        "text_min_freq": "1",
                        "text_max_vocab_size": str(max_vocab_size)
                    }
                },
                {
                    "id": "text_tokenizer",
                    "type": "TextTokenizer",
                    "name": tokenizer_name,
                    "pos": [250, 300],
                    "params": {
                        "tokenizer_type": tokenizer_id,
                        "max_length": str(max_length),
                        "lowercase": "true",
                        "min_freq": "1",
                        "max_vocab_size": str(max_vocab_size)
                    }
                },
                {
                    "id": "text_vocab",
                    "type": "TextVocabulary",
                    "name": "Vocabulary",
                    "pos": [500, 300],
                    "params": {
                        "min_freq": "1",
                        "max_vocab_size": str(max_vocab_size),
                        "vocab_file": vocab_path
                    }
                },
                {
                    "id": "text_padding",
                    "type": "TextPadding",
                    "name": "Padding",
                    "pos": [750, 300],
                    "params": {
                        "max_length": str(max_length),
                        "pad_value": "0"
                    }
                },
                {
                    "id": "data_split",
                    "type": "DataSplit",
                    "name": "Split 80/10/10",
                    "pos": [1000, 300],
                    "params": {
                        "train_ratio": "0.8",
                        "val_ratio": "0.1",
                        "test_ratio": "0.1",
                        "stratified": "true",
                        "seed": "42"
                    }
                },
                {
                    "id": "data_loader",
                    "type": "DataLoader",
                    "name": "DataLoader",
                    "pos": [1250, 300],
                    "params": {
                        "batch_size": "64",
                        "epochs": "8",
                        "shuffle": "true",
                        "seed": "42",
                        "validation_freq": "1"
                    }
                },
                {
                    "id": "embedding",
                    "type": "Embedding",
                    "name": "Embedding",
                    "pos": [1500, 300],
                    "params": {
                        "num_embeddings": str(max_vocab_size),
                        "embedding_dim": "96",
                        "padding_idx": "0"
                    }
                },
                {
                    "id": "gru",
                    "type": "GRU",
                    "name": "GRU",
                    "pos": [1750, 300],
                    "params": {
                        "hidden_size": "128",
                        "num_layers": "1",
                        "bidirectional": "false",
                        "return_sequences": "false"
                    }
                },
                {
                    "id": "fc1",
                    "type": "Dense",
                    "name": "Dense 64",
                    "pos": [2000, 300],
                    "params": {
                        "units": "64"
                    }
                },
                {
                    "id": "relu1",
                    "type": "ReLU",
                    "name": "ReLU",
                    "pos": [2250, 300]
                },
                {
                    "id": "drop1",
                    "type": "Dropout",
                    "name": "Dropout",
                    "pos": [2500, 300],
                    "params": {
                        "rate": "0.2"
                    }
                },
                {
                    "id": "fc_out",
                    "type": "Dense",
                    "name": "Dense N",
                    "pos": [2750, 300],
                    "params": {
                        "units": str(num_classes)
                    }
                },
                {
                    "id": "loss",
                    "type": "CrossEntropyLoss",
                    "name": "CrossEntropy",
                    "pos": [3000, 300]
                },
                {
                    "id": "optimizer",
                    "type": "Adam",
                    "name": "Adam",
                    "pos": [3250, 300],
                    "params": {
                        "learning_rate": "0.0007"
                    }
                },
                {
                    "id": "output",
                    "type": "Output",
                    "name": "Output",
                    "pos": [3500, 300],
                    "params": {
                        "classes": str(num_classes)
                    }
                }
            ],
            "links": [
                {"from": "data_input", "to": "text_tokenizer"},
                {"from": "text_tokenizer", "to": "text_vocab"},
                {"from": "text_vocab", "to": "text_padding"},
                {"from": "text_padding", "to": "data_split"},
                {"from": "data_split", "to": "data_loader"},
                {"from": "data_loader", "to": "embedding"},
                {"from": "embedding", "to": "gru"},
                {"from": "gru", "to": "fc1"},
                {"from": "fc1", "to": "relu1"},
                {"from": "relu1", "to": "drop1"},
                {"from": "drop1", "to": "fc_out"},
                {"from": "fc_out", "to": "loss"},
                {"from": "loss", "to": "optimizer"},
                {"from": "optimizer", "to": "output"}
            ]
        }
    }


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_abs = source.resolve()
    out_dir_abs = out_dir.resolve()

    label_counts = Counter()
    for record in iter_json_records(source):
        label = record.get(args.label_field)
        query = record.get(args.query_field)
        if isinstance(label, str) and isinstance(query, str) and query.strip():
            label_counts[normalize_label(label)] += 1

    if not label_counts:
        raise SystemExit(f"No usable records found in {source}")

    selected_labels = [
        label
        for label, count in label_counts.most_common()
        if count >= args.min_label_count
    ]
    if args.top_k_labels > 0:
        selected_labels = selected_labels[: args.top_k_labels]
    if not selected_labels:
        raise SystemExit(
            "No labels left after filtering. Lower --min-label-count or increase --top-k-labels."
        )

    csv_path = out_dir_abs / "codefeedback_lang_multiclass.csv"
    vocab_path = out_dir_abs / "codefeedback_vocab.txt"
    meta_path = out_dir_abs / "codefeedback_metadata.json"
    graph_path = out_dir_abs / "codefeedback_lang_classifier_multi.cyxgraph"

    per_label_written = Counter()
    token_counts = Counter()
    rows_by_label: dict[str, list[dict[str, str]]] = {label: [] for label in selected_labels}

    for record in iter_json_records(source):
        label = record.get(args.label_field)
        query = record.get(args.query_field)
        answer = record.get(args.answer_field, "")

        if not isinstance(label, str):
            continue
        label = normalize_label(label)
        if label not in selected_labels:
            continue
        if not isinstance(query, str) or not query.strip():
            continue
        if per_label_written[label] >= args.max_per_label:
            continue

        if not isinstance(answer, str):
            answer = ""

        text = build_text(query, answer, args.text_mode)
        token_counts.update(tokenize_text(text, args.tokenizer_type, lowercase=True))

        rows_by_label[label].append(
            {
                "text": text,
                "label": label,
                "query": query,
                "answer": answer,
            }
        )
        per_label_written[label] += 1

    rng = random.Random(42)
    balanced_rows: list[dict[str, str]] = []
    target_rows_per_label: dict[str, int] = {}

    if args.balance_mode == "none":
        for label in selected_labels:
            target_rows_per_label[label] = len(rows_by_label[label])
            balanced_rows.extend(rows_by_label[label])
    else:
        if args.balance_mode == "downsample":
            available = [len(rows_by_label[label]) for label in selected_labels if rows_by_label[label]]
            balance_target = min(args.balance_target, min(available) if available else 0)
        else:
            balance_target = args.balance_target

        for label in selected_labels:
            rows = rows_by_label[label]
            if not rows:
                target_rows_per_label[label] = 0
                continue
            if args.balance_mode == "downsample":
                count = min(len(rows), balance_target)
                sampled = rng.sample(rows, count) if count < len(rows) else list(rows)
            else:
                count = balance_target
                if len(rows) >= count:
                    sampled = rng.sample(rows, count)
                else:
                    sampled = list(rows)
                    while len(sampled) < count:
                        sampled.append(rng.choice(rows))
            target_rows_per_label[label] = len(sampled)
            balanced_rows.extend(sampled)

    rng.shuffle(balanced_rows)
    rows_written = len(balanced_rows)

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["text", "label", "query", "answer"],
        )
        writer.writeheader()
        for row in balanced_rows:
            writer.writerow(row)

    max_vocab_tokens = max(0, args.max_vocab_size - len(SPECIAL_TOKENS))
    vocab_tokens = [
        token
        for token, _ in sorted(
            token_counts.items(), key=lambda item: (-item[1], item[0])
        )[:max_vocab_tokens]
    ]

    with vocab_path.open("w", encoding="utf-8", newline="\n") as vocab_file:
        for token in SPECIAL_TOKENS:
            vocab_file.write(token + "\n")
        for token in vocab_tokens:
            vocab_file.write(token + "\n")

    metadata = {
        "source": str(source_abs),
        "csv_file": str(csv_path),
        "vocab_file": str(vocab_path),
        "label_field": args.label_field,
        "query_field": args.query_field,
        "answer_field": args.answer_field,
        "selected_labels": selected_labels,
        "label_counts": {label: label_counts[label] for label in selected_labels},
        "label_order": selected_labels,
        "label_aliases_applied": LABEL_ALIASES,
        "top_k_labels": args.top_k_labels,
        "min_label_count": args.min_label_count,
        "rows_per_label_target": target_rows_per_label,
        "rows_written": rows_written,
        "max_per_label": args.max_per_label,
        "balance_mode": args.balance_mode,
        "balance_target": args.balance_target,
        "max_vocab_size": args.max_vocab_size,
        "max_length": args.max_length,
        "num_classes": len(selected_labels),
        "tokenizer_type": args.tokenizer_type,
        "tokenizer_type_id": 2 if args.tokenizer_type == "character" else 0,
        "text_mode": args.text_mode,
        "lowercase": True,
        "text_template": (
            "{answer}" if args.text_mode == "answer_only"
            else "{query}" if args.text_mode == "query_only"
            else "[q] {query} [a] {answer}"
        ),
    }

    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    graph_path.write_text(
        json.dumps(
                build_graph_template(
                    str(csv_path).replace("\\", "/"),
                    str(vocab_path).replace("\\", "/"),
                    len(selected_labels),
                    args.max_length,
                    args.max_vocab_size,
                    args.tokenizer_type,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"Source: {source_abs}")
    print(f"Selected labels: {', '.join(selected_labels)}")
    print(f"Balance mode: {args.balance_mode} (target={args.balance_target})")
    print(f"Text mode: {args.text_mode}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {vocab_path}")
    print(f"Wrote: {meta_path}")
    print(f"Wrote: {graph_path}")
    print(f"Rows written: {rows_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
