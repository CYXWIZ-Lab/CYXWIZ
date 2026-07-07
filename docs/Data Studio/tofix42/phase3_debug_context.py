#!/usr/bin/env python3
"""Phase 3 debugger trace context packet builder for tofix42.

This is intentionally a deterministic harness only. It reads an existing
DebugRunStore-style JSON record or a single DebugTraceRecord-style JSON object
and emits the selected-trace context that a later model/runtime integration can
use. It does not start a model, edit graphs, mutate source, or depend on UI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import phase1a_retrieval as retrieval


CONTEXT_SCHEMA = "cyxwiz.tofix42.phase3.debug_trace_context.v1"
SENSITIVE_KEY_MARKERS = (
    "path",
    "file",
    "dataset",
    "raw",
    "preview",
    "token",
    "password",
    "secret",
    "credential",
)
ANSWER_QUESTION = "Explain the selected Studio Debugger trace."
PACKET_SCHEMA = "cyxwiz.tofix42.phase1a.answer_packet.v1"
EXPLANATION_SCHEMA = "cyxwiz.tofix42.phase3.debug_trace_explanation.v1"


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


def is_sensitive_key(key: str) -> bool:
    lower = key.lower()
    return any(marker in lower for marker in SENSITIVE_KEY_MARKERS)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if is_sensitive_key(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        lowered = value.lower()
        for marker in ("token=", "password=", "secret=", "credential="):
            pos = lowered.find(marker)
            if pos >= 0:
                return value[: pos + len(marker)] + "[REDACTED]"
    return value


def normalize_record(value: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value.get("debug_run"), dict):
        value = value["debug_run"]

    if isinstance(value.get("summary"), dict) and isinstance(value.get("traces"), list):
        summary = dict(value["summary"])
        return {
            "run_id": summary.get("run_id", ""),
            "timestamp": summary.get("timestamp", ""),
            "graph_hash": summary.get("graph_hash", 0),
            "success": summary.get("success", False),
            "summary": summary.get("summary", ""),
            "issues": value.get("issues", []),
            "traces": value.get("traces", []),
            "studio_events": value.get("studio_events", []),
            "recommendations": value.get("recommendations", []),
        }

    if isinstance(value.get("traces"), list):
        return {
            "run_id": value.get("run_id", ""),
            "timestamp": value.get("timestamp", ""),
            "graph_hash": value.get("graph_hash", 0),
            "success": value.get("success", False),
            "summary": value.get("summary", ""),
            "issues": value.get("issues", []),
            "traces": value.get("traces", []),
            "studio_events": value.get("studio_events", []),
            "recommendations": value.get("recommendations", []),
        }

    if "run_id" in value or "node_id" in value:
        return {
            "run_id": value.get("run_id", ""),
            "timestamp": "",
            "graph_hash": 0,
            "success": False,
            "summary": "",
            "issues": value.get("issues", []),
            "traces": [value],
            "studio_events": [],
            "recommendations": [],
        }

    raise ValueError("Expected DebugRunStore, support-bundle debug_run, or DebugTraceRecord JSON")


def issue_level(issue: dict[str, Any]) -> str:
    return str(issue.get("level", "")).lower()


def trace_has_problem(trace: dict[str, Any]) -> bool:
    status = str(trace.get("status", "")).lower()
    role = str(trace.get("role", "")).lower()
    issues = trace.get("issues", [])
    if status in {"failed", "failure", "error", "warning", "missing"}:
        return True
    if role in {"error", "warning"}:
        return True
    return isinstance(issues, list) and any(
        isinstance(issue, dict) and issue_level(issue) in {"error", "warning"}
        for issue in issues
    )


def select_trace(
    traces: list[Any],
    *,
    trace_index: int | None,
    node_id: int | None,
    role: str,
) -> tuple[int, dict[str, Any]]:
    normalized = [trace for trace in traces if isinstance(trace, dict)]
    if not normalized:
        raise ValueError("No trace objects found")

    indexed = list(enumerate(normalized))
    if node_id is not None:
        indexed = [
            (index, trace)
            for index, trace in indexed
            if int_value(trace.get("node_id"), -1) == node_id
        ]
    if role:
        wanted = role.lower()
        indexed = [
            (index, trace)
            for index, trace in indexed
            if str(trace.get("role", "")).lower() == wanted
        ]

    if not indexed:
        raise ValueError("No traces matched the selection filters")

    if trace_index is not None:
        for index, trace in indexed:
            if index == trace_index:
                return index, trace
        raise ValueError(f"No trace exists at index {trace_index}")

    for index, trace in indexed:
        if trace_has_problem(trace):
            return index, trace
    return indexed[0]


def int_value(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def shape_text(value: Any) -> str:
    if not isinstance(value, list):
        return "[]"
    return "[" + ", ".join(str(item) for item in value) + "]"


def compact_trace(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": trace.get("run_id", ""),
        "node_id": int_value(trace.get("node_id"), -1),
        "node_name": trace.get("node_name", ""),
        "node_type": trace.get("node_type", ""),
        "phase": trace.get("phase", ""),
        "role": trace.get("role", ""),
        "input_shape": trace.get("input_shape", []),
        "output_shape": trace.get("output_shape", []),
        "dtype": trace.get("dtype", ""),
        "duration_ms": trace.get("duration_ms", 0.0),
        "status": trace.get("status", ""),
        "issues": redact(trace.get("issues", [])),
        "payload": redact(trace.get("payload", {})),
    }


def related_items(items: list[Any], node_id: int, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_node = int_value(item.get("node_id", item.get("selected_node_id")), -1)
        if node_id >= 0 and item_node not in {-1, node_id}:
            continue
        out.append(redact(item))
        if len(out) >= limit:
            break
    return out


def build_facts(trace: dict[str, Any], related_traces: list[dict[str, Any]]) -> list[str]:
    node = f"node {trace.get('node_id', -1)}"
    name = str(trace.get("node_name", "")).strip()
    node_type = str(trace.get("node_type", "")).strip()
    if name:
        node += f" ({name})"
    if node_type:
        node += f" of type {node_type}"

    facts = [
        (
            f"Selected trace is {node}, phase {trace.get('phase', '')}, "
            f"role {trace.get('role', '')}, status {trace.get('status', '')}."
        ),
        (
            f"Input shape is {shape_text(trace.get('input_shape'))}; output shape is "
            f"{shape_text(trace.get('output_shape'))}; dtype is {trace.get('dtype', '')}."
        ),
    ]

    issues = trace.get("issues", [])
    if isinstance(issues, list) and issues:
        facts.append(f"Selected trace has {len(issues)} issue(s).")
        for issue in issues[:5]:
            if isinstance(issue, dict):
                code = str(issue.get("error_code", "")).strip()
                message = str(issue.get("message", "")).strip()
                level = str(issue.get("level", "")).strip()
                label = " ".join(part for part in (level, code) if part)
                facts.append(f"Issue: {label}: {message}".strip())
    else:
        facts.append("Selected trace has no attached issues.")

    if related_traces:
        facts.append(f"{len(related_traces)} related trace(s) from the same node were included.")
    return facts


def build_context(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    index, selected = select_trace(
        record["traces"],
        trace_index=args.trace_index,
        node_id=args.node_id,
        role=args.role,
    )
    selected_trace = compact_trace(selected)
    node_id = int_value(selected_trace.get("node_id"), -1)

    same_node_traces = []
    for item in record["traces"]:
        if not isinstance(item, dict) or item is selected:
            continue
        if int_value(item.get("node_id"), -1) == node_id:
            same_node_traces.append(compact_trace(item))
        if len(same_node_traces) >= args.related_limit:
            break

    return {
        "schema": CONTEXT_SCHEMA,
        "question": args.question,
        "selection": {
            "trace_index": index,
            "node_id": node_id,
            "role": selected_trace.get("role", ""),
        },
        "run_summary": redact(
            {
                "run_id": record.get("run_id", ""),
                "timestamp": record.get("timestamp", ""),
                "graph_hash": record.get("graph_hash", 0),
                "success": record.get("success", False),
                "summary": record.get("summary", ""),
                "issue_count": len(record.get("issues", [])),
                "trace_count": len(record.get("traces", [])),
                "event_count": len(record.get("studio_events", [])),
                "recommendation_count": len(record.get("recommendations", [])),
            }
        ),
        "selected_trace": selected_trace,
        "related_traces": same_node_traces,
        "related_issues": related_items(record.get("issues", []), node_id, args.related_limit),
        "related_studio_events": related_items(
            record.get("studio_events", []),
            node_id,
            args.related_limit,
        ),
        "related_recommendations": related_items(
            record.get("recommendations", []),
            node_id,
            args.related_limit,
        ),
        "facts": build_facts(selected_trace, same_node_traces),
        "evidence_queries": [
            "What source file defines DebugTraceRecord",
            f"{selected_trace.get('node_type', '')} {selected_trace.get('role', '')} debugger trace",
            f"{selected_trace.get('phase', '')} {selected_trace.get('status', '')} validation issue",
        ],
        "answer_contract": {
            "must_answer": [
                "what happened",
                "where it happened",
                "likely why, clearly marked as inference",
                "what to inspect next",
            ],
            "must_not": [
                "claim unsupported behavior without evidence",
                "mutate source or graphs",
                "expose redacted dataset paths or raw data previews",
            ],
        },
    }


def packet_question(context: dict[str, Any]) -> str:
    lines = [
        context.get("question", ANSWER_QUESTION),
        "",
        "Selected debugger context facts:",
    ]
    for fact in context.get("facts", []):
        lines.append(f"- {fact}")
    lines.extend(
        [
            "",
            "Answer what happened, where it happened, likely why as inference,",
            "and what to inspect next. Use the selected trace facts and cited",
            "source evidence only.",
        ]
    )
    return "\n".join(lines)


def add_packet_evidence(
    evidence: list[dict[str, Any]],
    seen: set[str],
    results: list[dict[str, Any]],
) -> None:
    for item in results:
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


def make_debug_answer_packet(context: dict[str, Any], index: dict[str, Any], top: int) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    queries = [
        "What source file defines DebugTraceRecord",
        "DebugRunStore DebugTraceRecord StudioEventRecord",
    ]
    queries.extend(
        query
        for query in context.get("evidence_queries", [])
        if isinstance(query, str) and query.strip()
    )

    per_query_top = max(1, min(top, 3))
    for query in queries:
        add_packet_evidence(evidence, seen, retrieval.search(index, query, per_query_top))
        if len(evidence) >= top:
            evidence = evidence[:top]
            break

    missing_notes = []
    if not evidence:
        missing_notes.append("No matching local evidence was found in the Phase 1A index.")
    if len(evidence) < top:
        missing_notes.append("Fewer evidence chunks were available than requested.")

    return {
        "schema": PACKET_SCHEMA,
        "question": packet_question(context),
        "answer_contract": {
            "mode": "debug_trace_context_plus_retrieval",
            "model_runtime": "not_used",
            "rules": [
                "answer only from selected trace facts and cited evidence",
                "separate fact from inference",
                "state when evidence is missing",
                "do not claim unsupported CyxWiz behavior",
                "do not suggest graph or source mutation unless explicitly approved",
                "do not expose redacted local paths, raw data previews, or secrets",
            ],
        },
        "debug_context": context,
        "evidence": evidence,
        "missing_evidence_notes": missing_notes,
    }


def citation_text(citation: dict[str, Any]) -> str:
    return (
        f"{citation.get('path', '')}:"
        f"{citation.get('line_start', '')}-{citation.get('line_end', '')}"
    )


def evidence_citations(packet: dict[str, Any], limit: int = 4) -> list[str]:
    out: list[str] = []
    for item in packet.get("evidence", [])[:limit]:
        citation = item.get("citation", {})
        if isinstance(citation, dict):
            out.append(citation_text(citation))
    return out


def first_issue(trace: dict[str, Any]) -> dict[str, Any] | None:
    issues = trace.get("issues", [])
    if not isinstance(issues, list):
        return None
    for issue in issues:
        if isinstance(issue, dict):
            return issue
    return None


def first_recommendation(context: dict[str, Any]) -> dict[str, Any] | None:
    recommendations = context.get("related_recommendations", [])
    if not isinstance(recommendations, list):
        return None
    for recommendation in recommendations:
        if isinstance(recommendation, dict):
            return recommendation
    return None


def trace_evidence_items(trace: dict[str, Any], issue: dict[str, Any] | None) -> list[str]:
    items = [
        f"selected_trace.run_id={trace.get('run_id', '')}",
        f"selected_trace.node_id={trace.get('node_id', -1)}",
        f"selected_trace.node_name={trace.get('node_name', '')}",
        f"selected_trace.node_type={trace.get('node_type', '')}",
        f"selected_trace.phase={trace.get('phase', '')}",
        f"selected_trace.role={trace.get('role', '')}",
        f"selected_trace.status={trace.get('status', '')}",
        f"selected_trace.input_shape={shape_text(trace.get('input_shape'))}",
        f"selected_trace.output_shape={shape_text(trace.get('output_shape'))}",
        f"selected_trace.dtype={trace.get('dtype', '')}",
    ]
    if issue:
        items.append(f"selected_trace.issues[0].level={issue.get('level', '')}")
        items.append(f"selected_trace.issues[0].error_code={issue.get('error_code', '')}")
        items.append(f"selected_trace.issues[0].message={issue.get('message', '')}")
    return items


def make_deterministic_explanation(packet: dict[str, Any]) -> dict[str, Any]:
    context = packet.get("debug_context", {})
    trace = context.get("selected_trace", {}) if isinstance(context, dict) else {}
    if not isinstance(trace, dict):
        trace = {}

    issue = first_issue(trace)
    recommendation = first_recommendation(context if isinstance(context, dict) else {})
    node_id = trace.get("node_id", -1)
    node_name = str(trace.get("node_name", "")).strip()
    node_type = str(trace.get("node_type", "")).strip()
    phase = str(trace.get("phase", "")).strip()
    role = str(trace.get("role", "")).strip()
    status = str(trace.get("status", "")).strip()
    node_label = f"node {node_id}"
    if node_name:
        node_label += f" ({node_name})"
    if node_type:
        node_label += f" of type {node_type}"

    answer = (
        f"The selected debugger trace reports {status or 'an unspecified status'} "
        f"for {node_label} during phase {phase or 'unknown'} with role "
        f"{role or 'unknown'}."
    )
    where = (
        f"It is attached to run {trace.get('run_id', '') or 'unknown'}, "
        f"node_id {node_id}, phase {phase or 'unknown'}, role {role or 'unknown'}."
    )

    if issue:
        code = str(issue.get("error_code", "")).strip()
        message = str(issue.get("message", "")).strip()
        likely_why = (
            "Inference: the attached validation issue is the most likely reason "
            f"for the trace status: {code + ' ' if code else ''}{message}."
        )
    else:
        likely_why = (
            "Inference: no issue is attached to this selected trace, so the "
            "trace context alone is not enough to identify a root cause."
        )

    inspect_next = [
        "Inspect the selected trace payload and issue list.",
        "Compare input_shape, output_shape, dtype, phase, role, and status with the active graph node.",
    ]
    if recommendation:
        action = str(recommendation.get("action", "")).strip()
        title = str(recommendation.get("title", "")).strip()
        if action:
            inspect_next.append(f"Review recommendation {title or 'for this node'}: {action}")

    trace_evidence = trace_evidence_items(trace, issue)
    source_evidence = evidence_citations(packet)

    return {
        "schema": EXPLANATION_SCHEMA,
        "answer": answer,
        "where": where,
        "likely_why": likely_why,
        "inspect_next": inspect_next,
        "trace_evidence": trace_evidence,
        "source_evidence": source_evidence,
        "evidence": trace_evidence + source_evidence,
        "unknowns": [
            "No real model inference was used.",
            "Active graph contents are not included unless present in the debug run JSON.",
        ],
        "unsupported_or_not_implemented": [
            "No graph or source mutation is performed.",
            "This deterministic explanation does not replace real local model validation.",
        ],
        "debug_context": context,
    }


def print_explanation_markdown(explanation: dict[str, Any]) -> None:
    print("Answer:")
    print(explanation.get("answer", ""))
    print("\nWhere:")
    print(explanation.get("where", ""))
    print("\nLikely why:")
    print(explanation.get("likely_why", ""))
    print("\nWhat to inspect next:")
    for item in explanation.get("inspect_next", []):
        print(f"- {item}")
    print("\nEvidence:")
    trace_evidence = explanation.get("trace_evidence", [])
    source_evidence = explanation.get("source_evidence", [])
    if trace_evidence:
        print("Trace fields:")
        for item in trace_evidence:
            print(f"- {item}")
    if source_evidence:
        print("Source citations:")
        for citation in source_evidence:
            print(f"- {citation}")
    if not trace_evidence and not source_evidence:
        print("- No trace or source evidence was available.")
    else:
        if not source_evidence:
            print("Source citations:")
            print("- No source evidence was retrieved.")
    print("\nUnknowns:")
    for item in explanation.get("unknowns", []):
        print(f"- {item}")
    print("\nUnsupported or not implemented:")
    for item in explanation.get("unsupported_or_not_implemented", []):
        print(f"- {item}")


def cmd_context(args: argparse.Namespace) -> int:
    record = normalize_record(load_json(args.run))
    context = build_context(record, args)
    print(json.dumps(context, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def cmd_packet(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}", file=sys.stderr)
        print("Run phase1a_retrieval.py build first.", file=sys.stderr)
        return 2

    record = normalize_record(load_json(args.run))
    context = build_context(record, args)
    index = retrieval.load_index(args.index)
    packet = make_debug_answer_packet(context, index, args.top)
    print(json.dumps(packet, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}", file=sys.stderr)
        print("Run phase1a_retrieval.py build first.", file=sys.stderr)
        return 2

    record = normalize_record(load_json(args.run))
    context = build_context(record, args)
    index = retrieval.load_index(args.index)
    packet = make_debug_answer_packet(context, index, args.top)
    explanation = make_deterministic_explanation(packet)
    if args.json:
        print(json.dumps(explanation, indent=2 if args.pretty else None, sort_keys=args.pretty))
    else:
        print_explanation_markdown(explanation)
    return 0


def sample_record() -> dict[str, Any]:
    return {
        "run_id": "support-run",
        "timestamp": "2026-06-18T13:50:00",
        "graph_hash": 65261,
        "success": False,
        "summary": "Support bundle contract",
        "issues": [
            {
                "level": "Error",
                "node_id": 5,
                "node_name": "Tokenizer",
                "error_code": "CW-D-0101",
                "message": "[CW-D-0101] required column missing",
            }
        ],
        "traces": [
            {
                "run_id": "support-run",
                "node_id": 5,
                "node_name": "Tokenizer",
                "node_type": "TextTokenizer",
                "phase": "TextTokenizer",
                "role": "PreprocessingOutput",
                "input_shape": [2],
                "output_shape": [2, 4],
                "dtype": "int64",
                "duration_ms": 1.5,
                "status": "failed",
                "issues": [
                    {
                        "level": "Error",
                        "node_id": 5,
                        "node_name": "Tokenizer",
                        "error_code": "CW-D-0101",
                        "message": "required column missing",
                    }
                ],
                "payload": {
                    "schema": "cyxwiz.debug.node_trace.v1",
                    "backend": "CPU",
                    "raw_text_preview": "private dataset row",
                    "source_path": "C:/Users/private/data.csv",
                    "error_code": "CW-D-0101",
                },
            }
        ],
        "studio_events": [
            {
                "run_id": "support-run",
                "timestamp": "2026-06-18T13:50:01",
                "graph_hash": 65261,
                "selected_node_id": 5,
                "action": "StudioDebugger.SelectTrace",
                "status": "ok",
                "message": "Selected failing trace",
            }
        ],
        "recommendations": [
            {
                "node_id": 5,
                "category": "Data",
                "title": "Missing required column",
                "detail": "The text column was not found.",
                "action": "Select a dataset with the configured text column.",
            }
        ],
    }


def cmd_check(args: argparse.Namespace) -> int:
    context = build_context(
        normalize_record(sample_record()),
        argparse.Namespace(
            trace_index=None,
            node_id=None,
            role="",
            question=ANSWER_QUESTION,
            related_limit=5,
        ),
    )
    checks = [
        context.get("schema") == CONTEXT_SCHEMA,
        context["selection"]["node_id"] == 5,
        context["selected_trace"]["payload"]["raw_text_preview"] == "[REDACTED]",
        context["selected_trace"]["payload"]["source_path"] == "[REDACTED]",
        bool(context["facts"]),
        bool(context["related_studio_events"]),
        bool(context["related_recommendations"]),
    ]
    if not all(checks):
        print(json.dumps(context, indent=2, sort_keys=True))
        print("Phase 3 debug context check failed", file=sys.stderr)
        return 1

    if args.index.exists():
        packet = make_debug_answer_packet(context, retrieval.load_index(args.index), top=5)
        explanation = make_deterministic_explanation(packet)
        packet_checks = [
            packet.get("schema") == PACKET_SCHEMA,
            "Selected debugger context facts:" in packet.get("question", ""),
            bool(packet.get("evidence")),
            packet["evidence"][0]["citation"]["path"]
            == "cyxwiz-engine/src/core/debug_trace_record.h",
            packet["debug_context"]["selected_trace"]["payload"]["source_path"] == "[REDACTED]",
            explanation.get("schema") == EXPLANATION_SCHEMA,
            "required column missing" in explanation.get("likely_why", ""),
            "selected_trace.status=failed" in explanation.get("trace_evidence", []),
            "selected_trace.issues[0].error_code=CW-D-0101"
            in explanation.get("trace_evidence", []),
            bool(explanation.get("source_evidence")),
        ]
        if not all(packet_checks):
            print(json.dumps({"packet": packet, "explanation": explanation}, indent=2, sort_keys=True))
            print("Phase 3 debug answer packet/explanation check failed", file=sys.stderr)
            return 1

    print("All Phase 3 debug context checks passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    context = sub.add_parser("context", help="build a selected-trace context packet")
    context.add_argument("--run", type=Path, default=None, help="DebugRunStore JSON path, or stdin")
    context.add_argument("--trace-index", type=int, default=None)
    context.add_argument("--node-id", type=int, default=None)
    context.add_argument("--role", default="")
    context.add_argument("--question", default=ANSWER_QUESTION)
    context.add_argument("--related-limit", type=int, default=5)
    context.add_argument("--pretty", action="store_true")
    context.set_defaults(func=cmd_context)

    packet = sub.add_parser("packet", help="build a retrieval-backed debug answer packet")
    packet.add_argument("--run", type=Path, default=None, help="DebugRunStore JSON path, or stdin")
    packet.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    packet.add_argument("--trace-index", type=int, default=None)
    packet.add_argument("--node-id", type=int, default=None)
    packet.add_argument("--role", default="")
    packet.add_argument("--question", default=ANSWER_QUESTION)
    packet.add_argument("--related-limit", type=int, default=5)
    packet.add_argument("--top", type=int, default=5)
    packet.add_argument("--pretty", action="store_true")
    packet.set_defaults(func=cmd_packet)

    explain = sub.add_parser("explain", help="build a deterministic selected-trace explanation")
    explain.add_argument("--run", type=Path, default=None, help="DebugRunStore JSON path, or stdin")
    explain.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    explain.add_argument("--trace-index", type=int, default=None)
    explain.add_argument("--node-id", type=int, default=None)
    explain.add_argument("--role", default="")
    explain.add_argument("--question", default=ANSWER_QUESTION)
    explain.add_argument("--related-limit", type=int, default=5)
    explain.add_argument("--top", type=int, default=5)
    explain.add_argument("--json", action="store_true")
    explain.add_argument("--pretty", action="store_true")
    explain.set_defaults(func=cmd_explain)

    check = sub.add_parser("check", help="run deterministic context checks")
    check.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
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
