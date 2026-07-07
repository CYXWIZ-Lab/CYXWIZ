#!/usr/bin/env python3
"""Phase 4 training trace explanation harness for tofix42.

This is deterministic and local-only. It reads a TrainingTraceCollector JSON
snapshot, support-bundle training_trace JSON, or stdin, then explains the latest
training terminal state from trace fields plus Phase 1A source citations. It
does not start a model, edit graphs, mutate source, or depend on Studio UI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import phase1a_retrieval as retrieval


CONTEXT_SCHEMA = "cyxwiz.tofix42.phase4.training_trace_context.v1"
EXPLANATION_SCHEMA = "cyxwiz.tofix42.phase4.training_trace_explanation.v1"
PACKET_SCHEMA = "cyxwiz.tofix42.phase1a.answer_packet.v1"


def read_text_any(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_json(path: Path | None) -> dict[str, Any]:
    raw = sys.stdin.read() if path is None or str(path) == "-" else read_text_any(path)
    raw = raw.lstrip("\ufeff")
    raw = raw.removeprefix(chr(239) + chr(187) + chr(191))
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def normalize_trace(value: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value.get("training_trace"), dict):
        value = value["training_trace"]

    events = value.get("events")
    if events is None:
        events = value.get("recent_events", [])
    materialization = value.get("materialization_events", [])
    if not isinstance(events, list):
        events = []
    if not isinstance(materialization, list):
        materialization = []

    latest = events[-1] if events and isinstance(events[-1], dict) else {}
    return {
        "available": bool(value.get("available", True)),
        "run_id": value.get("run_id", latest.get("run_id", "")),
        "status": value.get("status", ""),
        "latest_stage": value.get("latest_stage", latest.get("stage", "")),
        "latest_timestamp": value.get("latest_timestamp", latest.get("timestamp", "")),
        "latest_epoch": value.get("latest_epoch", latest.get("epoch", 0)),
        "latest_batch": value.get("latest_batch", latest.get("batch", 0)),
        "latest_total_batches": value.get(
            "latest_total_batches",
            latest.get("total_batches", 0),
        ),
        "latest_loss": value.get("latest_loss", latest.get("loss", 0.0)),
        "latest_accuracy": value.get("latest_accuracy", latest.get("accuracy", 0.0)),
        "events": events,
        "materialization_events": materialization,
        "warnings": value.get("warnings", []) if isinstance(value.get("warnings", []), list) else [],
    }


def terminal_event(events: list[Any]) -> dict[str, Any]:
    terminal_statuses = {"completed", "early_stopped", "cancelled", "failed"}
    terminal_stages = {"TrainingTerminal", "EarlyStopped", "Complete", "Cancelled", "Failed"}
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        if str(event.get("terminal_reason", "")).strip():
            return event
        if str(event.get("status", "")).strip() in terminal_statuses:
            return event
        if str(event.get("stage", "")).strip() in terminal_stages:
            return event
    return events[-1] if events and isinstance(events[-1], dict) else {}


def classify_terminal_state(summary: dict[str, Any], event: dict[str, Any]) -> str:
    status = str(event.get("status") or summary.get("status") or "").lower()
    stage = str(event.get("stage", "")).lower()
    reason = str(event.get("terminal_reason", "")).lower()
    if status == "completed" or reason == "completed_all_epochs" or stage == "complete":
        return "completed"
    if status == "early_stopped" or stage == "earlystopped":
        return "early_stopped"
    if status == "cancelled" or reason == "user_cancelled" or stage == "cancelled":
        return "cancelled"
    if status == "failed" or stage == "failed":
        return "failed"
    return status or "unknown"


def source_citations(index: dict[str, Any], top: int) -> list[str]:
    queries = [
        "TrainingTraceEvent terminal_reason field",
        "RecordTerminalEvent terminal_reason message",
        "completed_all_epochs terminal_reason user_cancelled",
    ]
    citations: list[str] = []
    seen: set[str] = set()
    for query in queries:
        for item in retrieval.search(index, query, top=2):
            chunk = item["chunk"]
            key = f"{chunk['path']}:{chunk['line_start']}:{chunk['line_end']}"
            if key in seen:
                continue
            seen.add(key)
            citations.append(key)
            if len(citations) >= top:
                return citations
    return citations


def source_evidence(index: dict[str, Any], top: int) -> list[dict[str, Any]]:
    queries = [
        "TrainingTraceEvent terminal_reason field",
        "RecordTerminalEvent terminal_reason message",
        "completed_all_epochs terminal_reason user_cancelled",
    ]
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in queries:
        for item in retrieval.search(index, query, top=2):
            chunk = item["chunk"]
            key = f"{chunk['path']}:{chunk['line_start']}:{chunk['line_end']}:{chunk['title']}"
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                {
                    "rank": len(evidence) + 1,
                    "score": item["score"],
                    "citation": retrieval.citation_for(chunk),
                    "text": chunk["text"],
                }
            )
            if len(evidence) >= top:
                return evidence
    return evidence


def build_context(summary: dict[str, Any]) -> dict[str, Any]:
    event = terminal_event(summary["events"])
    state = classify_terminal_state(summary, event)
    reason = str(event.get("terminal_reason") or event.get("message") or "").strip()
    return {
        "schema": CONTEXT_SCHEMA,
        "run_summary": {
            "available": summary["available"],
            "run_id": summary["run_id"],
            "status": summary["status"],
            "latest_stage": summary["latest_stage"],
            "latest_timestamp": summary["latest_timestamp"],
            "latest_epoch": summary["latest_epoch"],
            "latest_batch": summary["latest_batch"],
            "latest_total_batches": summary["latest_total_batches"],
            "latest_loss": summary["latest_loss"],
            "latest_accuracy": summary["latest_accuracy"],
            "event_count": len(summary["events"]),
            "materialization_event_count": len(summary["materialization_events"]),
            "warning_count": len(summary["warnings"]),
        },
        "terminal_event": event,
        "terminal_state": state,
        "terminal_reason": reason,
        "warnings": summary["warnings"][-10:],
        "recent_events": summary["events"][-10:],
        "materialization_events": summary["materialization_events"][-10:],
    }


def trace_evidence(context: dict[str, Any]) -> list[str]:
    summary = context["run_summary"]
    event = context["terminal_event"]
    evidence = [
        f"run_summary.run_id={summary.get('run_id', '')}",
        f"run_summary.status={summary.get('status', '')}",
        f"run_summary.latest_stage={summary.get('latest_stage', '')}",
        f"run_summary.latest_epoch={summary.get('latest_epoch', 0)}",
        f"run_summary.latest_batch={summary.get('latest_batch', 0)}",
        f"terminal_event.stage={event.get('stage', '')}",
        f"terminal_event.status={event.get('status', '')}",
        f"terminal_event.terminal_reason={event.get('terminal_reason', '')}",
        f"terminal_event.message={event.get('message', '')}",
    ]
    for idx, warning in enumerate(context.get("warnings", [])[:3]):
        evidence.append(f"warnings[{idx}]={warning}")
    return evidence


def packet_question(context: dict[str, Any]) -> str:
    lines = [
        "Explain the latest CyxWiz training trace terminal state.",
        "",
        "Training trace facts:",
    ]
    for item in trace_evidence(context):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "Answer why training stopped, whether it completed, early-stopped,",
            "cancelled, failed, or is unknown, and what to inspect next. Use",
            "the training trace facts and cited source evidence only.",
        ]
    )
    return "\n".join(lines)


def make_answer_packet(context: dict[str, Any], index: dict[str, Any], top: int) -> dict[str, Any]:
    evidence = source_evidence(index, top)
    missing_notes = []
    if not evidence:
        missing_notes.append("No matching local evidence was found in the Phase 1A index.")
    if len(evidence) < top:
        missing_notes.append("Fewer evidence chunks were available than requested.")
    return {
        "schema": PACKET_SCHEMA,
        "question": packet_question(context),
        "answer_contract": {
            "mode": "training_trace_context_plus_retrieval",
            "model_runtime": "not_used",
            "rules": [
                "answer only from training trace facts and cited evidence",
                "separate fact from inference",
                "state when evidence is missing",
                "do not treat warnings as proven root causes without terminal-event support",
                "do not mutate graphs, source, checkpoints, or training state",
            ],
        },
        "training_context": context,
        "evidence": evidence,
        "missing_evidence_notes": missing_notes,
    }


def make_explanation(context: dict[str, Any], source_evidence: list[str]) -> dict[str, Any]:
    summary = context["run_summary"]
    state = context["terminal_state"]
    reason = context["terminal_reason"]
    run_id = summary.get("run_id", "") or "unknown"
    latest_stage = summary.get("latest_stage", "") or "unknown"
    answer = f"Training run {run_id} is classified as {state}."
    if reason:
        answer += f" The terminal reason is {reason}."

    if state == "completed":
        likely = "Fact: the terminal state indicates training completed."
    elif state == "early_stopped":
        likely = "Fact: the terminal state indicates an early-stop path."
    elif state == "cancelled":
        likely = "Fact: the terminal state indicates user or caller cancellation."
    elif state == "failed":
        likely = "Fact: the terminal state indicates failure; inspect warnings and terminal event message first."
    else:
        likely = "Unknown: the trace does not include a recognized terminal state."

    inspect = [
        "Inspect terminal_event.terminal_reason and terminal_event.message.",
        "Inspect the latest recent_events entries around the terminal event.",
    ]
    if context.get("warnings"):
        inspect.append("Review warnings; treat them as signals, not proven root causes.")
    if context.get("materialization_events"):
        inspect.append("Review materialization_events for node-specific task failures.")

    return {
        "schema": EXPLANATION_SCHEMA,
        "answer": answer,
        "where": (
            f"Latest stage is {latest_stage}, epoch {summary.get('latest_epoch', 0)}, "
            f"batch {summary.get('latest_batch', 0)} of {summary.get('latest_total_batches', 0)}."
        ),
        "likely_why": likely,
        "inspect_next": inspect,
        "trace_evidence": trace_evidence(context),
        "source_evidence": source_evidence,
        "evidence": trace_evidence(context) + source_evidence,
        "unknowns": [
            "No model inference was used.",
            "Warnings are not treated as proven root causes without terminal-event support.",
        ],
        "unsupported_or_not_implemented": [
            "No graph, source, checkpoint, or training mutation is performed.",
            "Curve diagnosis and broad metric analytics are not implemented in this slice.",
        ],
        "training_context": context,
    }


def print_explanation(explanation: dict[str, Any]) -> None:
    for label, key in (
        ("Answer", "answer"),
        ("Where", "where"),
        ("Likely why", "likely_why"),
    ):
        print(f"{label}:")
        print(explanation[key])
        print()
    print("What to inspect next:")
    for item in explanation["inspect_next"]:
        print(f"- {item}")
    print("\nEvidence:")
    print("Trace fields:")
    for item in explanation["trace_evidence"]:
        print(f"- {item}")
    print("Source citations:")
    for item in explanation["source_evidence"] or ["No source evidence was retrieved."]:
        print(f"- {item}")


def sample_trace() -> dict[str, Any]:
    return {
        "run_id": "train-run-1",
        "status": "early_stopped",
        "events": [
            {
                "timestamp": "2026-07-01 20:00:00",
                "run_id": "train-run-1",
                "stage": "Forward",
                "epoch": 2,
                "batch": 3,
                "total_batches": 10,
                "loss": 0.42,
                "accuracy": 0.75,
                "status": "ok",
                "message": "",
            },
            {
                "timestamp": "2026-07-01 20:01:00",
                "run_id": "train-run-1",
                "stage": "EarlyStopped",
                "epoch": 2,
                "batch": 0,
                "total_batches": 10,
                "loss": 0.39,
                "accuracy": 0.78,
                "status": "early_stopped",
                "message": "validation_loss_plateau",
                "terminal_reason": "validation_loss_plateau",
            },
        ],
        "materialization_events": [
            {
                "stage": "MaterializeGraph",
                "status": "ok",
                "node_id": 12,
                "node_name": "Classifier",
                "message": "materialized",
            }
        ],
        "warnings": ["Validation loss did not improve for patience window."],
    }


def cmd_context(args: argparse.Namespace) -> int:
    context = build_context(normalize_trace(load_json(args.trace)))
    print(json.dumps(context, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}", file=sys.stderr)
        print("Run phase1a_retrieval.py build first.", file=sys.stderr)
        return 2
    context = build_context(normalize_trace(load_json(args.trace)))
    explanation = make_explanation(context, source_citations(retrieval.load_index(args.index), args.top))
    if args.json:
        print(json.dumps(explanation, indent=2 if args.pretty else None, sort_keys=args.pretty))
    else:
        print_explanation(explanation)
    return 0


def cmd_packet(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}", file=sys.stderr)
        print("Run phase1a_retrieval.py build first.", file=sys.stderr)
        return 2
    context = build_context(normalize_trace(load_json(args.trace)))
    packet = make_answer_packet(context, retrieval.load_index(args.index), args.top)
    print(json.dumps(packet, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    context = build_context(normalize_trace(sample_trace()))
    index = retrieval.load_index(args.index) if args.index.exists() else {"chunks": []}
    explanation = make_explanation(context, source_citations(index, args.top))
    packet = make_answer_packet(context, index, args.top)
    checks = [
        context["schema"] == CONTEXT_SCHEMA,
        context["terminal_state"] == "early_stopped",
        context["terminal_reason"] == "validation_loss_plateau",
        explanation["schema"] == EXPLANATION_SCHEMA,
        "terminal_event.terminal_reason=validation_loss_plateau" in explanation["trace_evidence"],
        bool(explanation["source_evidence"]) if args.index.exists() else True,
        packet["schema"] == PACKET_SCHEMA,
        "Training trace facts:" in packet["question"],
        bool(packet["evidence"]) if args.index.exists() else True,
        packet["training_context"]["terminal_state"] == "early_stopped",
    ]
    if not all(checks):
        print(json.dumps({"context": context, "explanation": explanation}, indent=2, sort_keys=True))
        print("Phase 4 training trace check failed", file=sys.stderr)
        return 1
    print("All Phase 4 training trace checks passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    context = sub.add_parser("context", help="build training trace context")
    context.add_argument("--trace", type=Path, default=None, help="trace JSON path, or stdin")
    context.add_argument("--pretty", action="store_true")
    context.set_defaults(func=cmd_context)

    explain = sub.add_parser("explain", help="explain training terminal state")
    explain.add_argument("--trace", type=Path, default=None, help="trace JSON path, or stdin")
    explain.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    explain.add_argument("--top", type=int, default=5)
    explain.add_argument("--json", action="store_true")
    explain.add_argument("--pretty", action="store_true")
    explain.set_defaults(func=cmd_explain)

    packet = sub.add_parser("packet", help="build a retrieval-backed training answer packet")
    packet.add_argument("--trace", type=Path, default=None, help="trace JSON path, or stdin")
    packet.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    packet.add_argument("--top", type=int, default=5)
    packet.add_argument("--pretty", action="store_true")
    packet.set_defaults(func=cmd_packet)

    check = sub.add_parser("check", help="run deterministic Phase 4 checks")
    check.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    check.add_argument("--top", type=int, default=5)
    check.set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
