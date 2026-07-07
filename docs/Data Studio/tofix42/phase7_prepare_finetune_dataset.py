#!/usr/bin/env python3
"""Prepare reviewed tofix42 corrections for a small fine-tuning experiment.

This is a local dataset-prep step only. It does not launch training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import phase1b_answer as answer_adapter
import phase6_eval_capture


PREP_SCHEMA = "cyxwiz.tofix42.phase7.dataset_prep.v1"
EXPORT_SCHEMA = "cyxwiz.tofix42.phase7.dataset_export.v1"
DEFAULT_INPUT = Path(__file__).with_name("phase6_reviewed_corrections.jsonl")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("phase7_dataset")

SYSTEM_PROMPT = (
    "You are the local CyxWiz source-aware assistant.\n"
    "Answer from retrieved CyxWiz evidence.\n"
    "Copy cited file paths exactly when you name them.\n"
    "Do not invent unsupported engine behavior.\n"
    "Return ONLY this structure in order:\n"
    "Answer: ...\n"
    "Evidence: ...\n"
    "Unknowns: ...\n"
    "Unsupported or not implemented: ..."
)


def load_reviewed_examples(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Reviewed corrections file not found: {path}")

    records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            invalid.append({"line": line_number, "error": str(exc)})
            continue
        if not isinstance(value, dict):
            invalid.append({"line": line_number, "error": "record is not a JSON object"})
            continue
        records.append(value)
    return records, invalid


def load_jsonl_objects(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            invalid.append({"line": line_number, "error": str(exc)})
            continue
        if not isinstance(value, dict):
            invalid.append({"line": line_number, "error": "record is not a JSON object"})
            continue
        records.append(value)
    return records, invalid


def case_family(case_id: str) -> str:
    parts = [part for part in case_id.split(".") if part]
    if len(parts) >= 2 and parts[0] == "phase2":
        return ".".join(parts[:2])
    if len(parts) >= 2 and parts[0] == "phase1a":
        return ".".join(parts[:2])
    return parts[0] if parts else "unknown"


def user_prompt(record: dict[str, Any]) -> str:
    query = str(record.get("query", "")).strip()
    expected = str(record.get("expected_citation", "")).strip()
    failure_mode = str(record.get("failure_mode", "")).strip()
    lines = [
        "Use the current CyxWiz retrieval context for this question.",
        "",
        "Question:",
        query,
        "",
        "Expected citation path:",
        expected or "unknown",
    ]
    if failure_mode:
        lines.extend(["", "Previous failure mode:", failure_mode])
    lines.extend(
        [
            "",
            "Return exactly:",
            "Answer: ...",
            "Evidence: ...",
            "Unknowns: ...",
            "Unsupported or not implemented: ...",
        ]
    )
    return "\n".join(lines).strip()


def normalize_corrected_output(text: str) -> str:
    out = text.strip()
    if "\\n" in out and "\n" not in out:
        out = out.replace("\\n", "\n")
    return out.strip()


def validate_target(record: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    corrected = normalize_corrected_output(str(record.get("corrected_output", "")))
    parsed = answer_adapter.parse_structured_answer(corrected)
    missing = [
        key
        for key in answer_adapter.ANSWER_SECTION_KEYS.values()
        if not str(parsed.get(key, "")).strip()
    ]
    return bool(parsed.get("parsed")) and not missing, missing, parsed


def prepare_example(record: dict[str, Any]) -> dict[str, Any]:
    corrected = normalize_corrected_output(str(record.get("corrected_output", "")))
    return {
        "schema": "cyxwiz.tofix42.phase7.chat_training_example.v1",
        "case_id": str(record.get("case_id", "")).strip(),
        "family": case_family(str(record.get("case_id", "")).strip()),
        "expected_citation": str(record.get("expected_citation", "")).strip(),
        "failure_mode": str(record.get("failure_mode", "")).strip(),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(record)},
            {"role": "assistant", "content": corrected},
        ],
        "assistant_output": corrected,
    }


def group_by_case_id(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        case_id = str(record.get("case_id", "")).strip()
        groups.setdefault(case_id, []).append(record)
    return groups


def split_case_ids(groups: dict[str, list[dict[str, Any]]], validation_groups_per_family: int) -> tuple[set[str], set[str], dict[str, list[str]]]:
    family_to_case_ids: dict[str, list[str]] = {}
    for case_id, items in groups.items():
        family = case_family(case_id)
        family_to_case_ids.setdefault(family, []).append(case_id)

    train_ids: set[str] = set()
    val_ids: set[str] = set()
    split_notes: dict[str, list[str]] = {}
    for family, case_ids in sorted(family_to_case_ids.items()):
        ordered = sorted(case_ids)
        if len(ordered) <= 1:
            train_ids.update(ordered)
            split_notes[family] = ["singleton_family_kept_in_train"]
            continue
        holdout_count = min(validation_groups_per_family, len(ordered) - 1)
        held_out = ordered[-holdout_count:]
        kept = ordered[:-holdout_count]
        val_ids.update(held_out)
        train_ids.update(kept)
        split_notes[family] = [
            "validation_case_ids=" + ",".join(held_out),
            "train_case_ids=" + ",".join(kept),
        ]
    return train_ids, val_ids, split_notes


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def export_messages_example(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": example.get("messages", []),
    }


def export_instruction_example(example: dict[str, Any]) -> dict[str, Any]:
    messages = example.get("messages", [])
    system = ""
    user = ""
    assistant = ""
    if isinstance(messages, list):
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", ""))
            if role == "system" and not system:
                system = content
            elif role == "user" and not user:
                user = content
            elif role == "assistant" and not assistant:
                assistant = content
    instruction = system.strip()
    if user.strip():
        instruction = (instruction + "\n\n" + user.strip()).strip() if instruction else user.strip()
    return {
        "instruction": instruction,
        "input": "",
        "output": assistant.strip(),
    }


def export_records(
    examples: list[dict[str, Any]],
    export_format: str,
) -> list[dict[str, Any]]:
    if export_format == "messages":
        return [export_messages_example(example) for example in examples]
    if export_format == "instruction":
        return [export_instruction_example(example) for example in examples]
    raise ValueError(f"Unsupported export format: {export_format}")


def cmd_prepare(args: argparse.Namespace) -> int:
    records, invalid = load_reviewed_examples(args.input)
    if invalid:
        print(
            json.dumps(
                {
                    "schema": PREP_SCHEMA,
                    "ok": False,
                    "error": "invalid_input_jsonl",
                    "invalid_records": invalid,
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 1

    reviewed = [
        record
        for record in records
        if str(record.get("corrected_output", "")).strip()
    ]
    validation_failures: list[dict[str, Any]] = []
    for idx, record in enumerate(reviewed, start=1):
        ok, missing, parsed = validate_target(record)
        if not ok:
            validation_failures.append(
                {
                    "record_number": idx,
                    "case_id": str(record.get("case_id", "")),
                    "missing_sections": missing,
                    "parsed": parsed.get("parsed", False),
                }
            )
    if validation_failures:
        print(
            json.dumps(
                {
                    "schema": PREP_SCHEMA,
                    "ok": False,
                    "error": "invalid_corrected_output",
                    "validation_failures": validation_failures,
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 1

    groups = group_by_case_id(reviewed)
    train_case_ids, val_case_ids, split_notes = split_case_ids(
        groups,
        args.validation_groups_per_family,
    )

    train_examples: list[dict[str, Any]] = []
    val_examples: list[dict[str, Any]] = []
    for record in reviewed:
        case_id = str(record.get("case_id", "")).strip()
        example = prepare_example(record)
        if case_id in val_case_ids:
            val_examples.append(example)
        else:
            train_examples.append(example)

    output_dir = args.output_dir
    train_path = output_dir / "train.chat.jsonl"
    val_path = output_dir / "validation.chat.jsonl"
    manifest_path = output_dir / "manifest.json"
    write_jsonl(train_path, train_examples)
    write_jsonl(val_path, val_examples)

    manifest = {
        "schema": PREP_SCHEMA,
        "ok": True,
        "input": str(args.input),
        "output_dir": str(output_dir),
        "train_path": str(train_path),
        "validation_path": str(val_path),
        "input_record_count": len(records),
        "reviewed_record_count": len(reviewed),
        "train_record_count": len(train_examples),
        "validation_record_count": len(val_examples),
        "unique_case_id_count": len(groups),
        "validation_groups_per_family": args.validation_groups_per_family,
        "families": sorted({case_family(str(record.get("case_id", "")).strip()) for record in reviewed}),
        "split_notes": split_notes,
        "system_prompt": SYSTEM_PROMPT,
        "format": "openai_chat_jsonl",
        "guardrails": [
            "retrieval_remains_mandatory",
            "citations_remain_mandatory",
            "no_unreviewed_outputs",
            "no_graph_or_source_mutation_examples",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(manifest, indent=2, ensure_ascii=True))
    else:
        print(f"Input: {args.input}")
        print(f"Output dir: {output_dir}")
        print(f"Train: {train_path}")
        print(f"Validation: {val_path}")
        print(f"Reviewed records: {len(reviewed)}")
        print(f"Unique case ids: {len(groups)}")
        print(f"Train records: {len(train_examples)}")
        print(f"Validation records: {len(val_examples)}")
    return 0


def cmd_export_format(args: argparse.Namespace) -> int:
    input_dir = args.input_dir
    train_src = input_dir / "train.chat.jsonl"
    val_src = input_dir / "validation.chat.jsonl"
    manifest_src = input_dir / "manifest.json"

    train_examples, train_invalid = load_jsonl_objects(train_src)
    val_examples, val_invalid = load_jsonl_objects(val_src)
    invalid = {
        "train": train_invalid,
        "validation": val_invalid,
    }
    if train_invalid or val_invalid:
        print(
            json.dumps(
                {
                    "schema": EXPORT_SCHEMA,
                    "ok": False,
                    "error": "invalid_source_jsonl",
                    "input_dir": str(input_dir),
                    "invalid_records": invalid,
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 1

    manifest = {}
    if manifest_src.exists():
        manifest = json.loads(manifest_src.read_text(encoding="utf-8-sig"))

    export_dir = args.output_dir or input_dir
    suffix = "messages" if args.format == "messages" else "instruction"
    train_out = export_dir / f"train.{suffix}.jsonl"
    val_out = export_dir / f"validation.{suffix}.jsonl"
    export_manifest = export_dir / f"manifest.{suffix}.json"

    train_export = export_records(train_examples, args.format)
    val_export = export_records(val_examples, args.format)
    write_jsonl(train_out, train_export)
    write_jsonl(val_out, val_export)

    report = {
        "schema": EXPORT_SCHEMA,
        "ok": True,
        "input_dir": str(input_dir),
        "source_manifest": str(manifest_src),
        "format": args.format,
        "output_dir": str(export_dir),
        "train_output": str(train_out),
        "validation_output": str(val_out),
        "train_record_count": len(train_export),
        "validation_record_count": len(val_export),
        "source_train_record_count": len(train_examples),
        "source_validation_record_count": len(val_examples),
        "source_format": manifest.get("format", "openai_chat_jsonl"),
    }
    export_manifest.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print(f"Input dir: {input_dir}")
        print(f"Format: {args.format}")
        print(f"Train output: {train_out}")
        print(f"Validation output: {val_out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="prepare train/validation JSONL from reviewed corrections")
    prepare.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare.add_argument("--validation-groups-per-family", type=int, default=1)
    prepare.add_argument("--json", action="store_true")
    prepare.set_defaults(func=cmd_prepare)

    export_format = sub.add_parser("export-format", help="export canonical chat dataset to a trainer-specific format")
    export_format.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    export_format.add_argument("--output-dir", type=Path, default=None)
    export_format.add_argument("--format", choices=["messages", "instruction"], required=True)
    export_format.add_argument("--json", action="store_true")
    export_format.set_defaults(func=cmd_export_format)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
