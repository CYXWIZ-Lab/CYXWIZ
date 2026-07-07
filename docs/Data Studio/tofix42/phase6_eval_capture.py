#!/usr/bin/env python3
"""Phase 6 evaluation and QA capture harness for tofix42.

This is deterministic and local. It evaluates the retrieval, debugger,
training trace, and graph draft-plan fixtures that already exist. It does not
call a model server, start Studio, mutate graphs, or grade subjective answer
quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import phase1a_retrieval as retrieval
import phase2_probe_suite
import phase3_debug_context
import phase4_training_context
import phase5_graph_context


REPORT_SCHEMA = "cyxwiz.tofix42.phase6.eval_report.v1"
BAD_ANSWER_SCHEMA = "cyxwiz.tofix42.phase6.bad_answer.v1"
BAD_ANSWER_REVIEW_SCHEMA = "cyxwiz.tofix42.phase6.bad_answer_review.v1"
BAD_ANSWER_REVIEW_QUEUE_SCHEMA = "cyxwiz.tofix42.phase6.bad_answer_review_queue.v1"
PROBE_FAILURE_CAPTURE_SCHEMA = "cyxwiz.tofix42.phase6.probe_failure_capture.v1"
REVIEWED_EXPORT_SCHEMA = "cyxwiz.tofix42.phase6.reviewed_correction_example.v1"
DEFAULT_BAD_ANSWERS = Path(__file__).with_name("phase6_bad_answers.jsonl")
DEFAULT_REVIEWED_EXPORT = Path(__file__).with_name("phase6_reviewed_corrections.jsonl")


def case_result(
    case_id: str,
    phase: str,
    ok: bool,
    expected: dict[str, Any],
    observed: dict[str, Any],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "phase": phase,
        "ok": ok,
        "expected": expected,
        "observed": observed,
        "notes": notes or [],
    }


def evaluate_retrieval(index: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for probe in phase2_probe_suite.PROBE_CASES:
        packet = retrieval.make_answer_packet(index, probe.query, top=1)
        evidence = packet.get("evidence", [])
        citation = evidence[0].get("citation", {}) if evidence else {}
        top_path = citation.get("path", "")
        results.append(
            case_result(
                f"retrieval.{probe.name}",
                "1A",
                top_path == probe.expected_path,
                {"query": probe.query, "top_path": probe.expected_path},
                {
                    "top_path": top_path,
                    "line_start": citation.get("line_start"),
                    "line_end": citation.get("line_end"),
                    "title": citation.get("title"),
                    "evidence_count": len(evidence),
                },
            )
        )
    return results


def evaluate_debugger(index: dict[str, Any], top: int) -> list[dict[str, Any]]:
    context = phase3_debug_context.build_context(
        phase3_debug_context.normalize_record(phase3_debug_context.sample_record()),
        argparse.Namespace(
            trace_index=None,
            node_id=None,
            role="",
            question=phase3_debug_context.ANSWER_QUESTION,
            related_limit=5,
        ),
    )
    packet = phase3_debug_context.make_debug_answer_packet(context, index, top)
    explanation = phase3_debug_context.make_deterministic_explanation(packet)
    return [
        case_result(
            "debugger.selected_trace_context",
            "3",
            (
                context.get("schema") == phase3_debug_context.CONTEXT_SCHEMA
                and context["selection"]["node_id"] == 5
                and context["selected_trace"]["payload"]["source_path"] == "[REDACTED]"
            ),
            {"schema": phase3_debug_context.CONTEXT_SCHEMA, "node_id": 5, "source_path": "[REDACTED]"},
            {
                "schema": context.get("schema"),
                "node_id": context.get("selection", {}).get("node_id"),
                "source_path": context.get("selected_trace", {}).get("payload", {}).get("source_path"),
            },
        ),
        case_result(
            "debugger.explanation_required_column_missing",
            "3",
            (
                explanation.get("schema") == phase3_debug_context.EXPLANATION_SCHEMA
                and "required column missing" in explanation.get("likely_why", "")
                and "selected_trace.status=failed" in explanation.get("trace_evidence", [])
            ),
            {
                "schema": phase3_debug_context.EXPLANATION_SCHEMA,
                "trace_evidence": "selected_trace.status=failed",
                "likely_why_contains": "required column missing",
            },
            {
                "schema": explanation.get("schema"),
                "likely_why": explanation.get("likely_why"),
                "trace_evidence": explanation.get("trace_evidence", []),
                "source_evidence_count": len(explanation.get("source_evidence", [])),
            },
        ),
    ]


def evaluate_training(index: dict[str, Any], top: int) -> list[dict[str, Any]]:
    context = phase4_training_context.build_context(
        phase4_training_context.normalize_trace(phase4_training_context.sample_trace())
    )
    explanation = phase4_training_context.make_explanation(
        context,
        phase4_training_context.source_citations(index, top),
    )
    packet = phase4_training_context.make_answer_packet(context, index, top)
    return [
        case_result(
            "training.terminal_reason",
            "4",
            (
                context.get("schema") == phase4_training_context.CONTEXT_SCHEMA
                and context.get("terminal_state") == "early_stopped"
                and context.get("terminal_reason") == "validation_loss_plateau"
            ),
            {
                "schema": phase4_training_context.CONTEXT_SCHEMA,
                "terminal_state": "early_stopped",
                "terminal_reason": "validation_loss_plateau",
            },
            {
                "schema": context.get("schema"),
                "terminal_state": context.get("terminal_state"),
                "terminal_reason": context.get("terminal_reason"),
            },
        ),
        case_result(
            "training.packet_and_explanation",
            "4",
            (
                explanation.get("schema") == phase4_training_context.EXPLANATION_SCHEMA
                and "terminal_event.terminal_reason=validation_loss_plateau"
                in explanation.get("trace_evidence", [])
                and packet.get("schema") == phase4_training_context.PACKET_SCHEMA
                and "Training trace facts:" in packet.get("question", "")
            ),
            {
                "explanation_schema": phase4_training_context.EXPLANATION_SCHEMA,
                "packet_schema": phase4_training_context.PACKET_SCHEMA,
                "question_contains": "Training trace facts:",
            },
            {
                "explanation_schema": explanation.get("schema"),
                "packet_schema": packet.get("schema"),
                "question": packet.get("question", "")[:120],
                "source_evidence_count": len(explanation.get("source_evidence", [])),
            },
        ),
    ]


def evaluate_graph(index: dict[str, Any], top: int) -> list[dict[str, Any]]:
    graph = phase5_graph_context.sample_graph()
    evidence = phase5_graph_context.suggestion_source_evidence(index, top)
    audit = phase5_graph_context.make_graph_audit(
        graph,
        Path("examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph"),
        evidence,
    )
    draft_goal = "Generate a draft text-classification graph using supported CyxWiz nodes."
    draft_evidence = phase5_graph_context.draft_source_evidence(
        index,
        draft_goal,
        "text-classification-tfidf-mlp",
        top,
    )
    draft_plan = phase5_graph_context.make_graph_draft_plan(
        "text-classification-tfidf-mlp",
        draft_goal,
        draft_evidence,
    )
    draft_packet = phase5_graph_context.make_draft_plan_packet(draft_plan, draft_evidence, top)
    audit_ids = {item.get("id") for item in audit.get("checks", [])}
    node_names = {item.get("name") for item in draft_plan.get("planned_nodes", [])}
    return [
        case_result(
            "graph.audit_structure",
            "5",
            (
                audit.get("schema") == phase5_graph_context.AUDIT_SCHEMA
                and {"has_nodes", "has_dataset_source", "has_model_layer", "has_loss", "has_optimizer", "acyclic"}.issubset(audit_ids)
            ),
            {
                "schema": phase5_graph_context.AUDIT_SCHEMA,
                "required_check_ids": [
                    "has_nodes",
                    "has_dataset_source",
                    "has_model_layer",
                    "has_loss",
                    "has_optimizer",
                    "acyclic",
                ],
            },
            {
                "schema": audit.get("schema"),
                "overall_status": audit.get("overall_status"),
                "check_ids": sorted(audit_ids),
            },
        ),
        case_result(
            "graph.draft_plan_tfidf_mlp",
            "5",
            (
                draft_plan.get("schema") == phase5_graph_context.DRAFT_PLAN_SCHEMA
                and draft_plan.get("status") == "draft_plan_only"
                and "TF-IDF vectorizer" in node_names
                and draft_packet.get("schema") == phase5_graph_context.PACKET_SCHEMA
                and "Graph draft plan facts:" in draft_packet.get("question", "")
            ),
            {
                "schema": phase5_graph_context.DRAFT_PLAN_SCHEMA,
                "status": "draft_plan_only",
                "node": "TF-IDF vectorizer",
                "packet_schema": phase5_graph_context.PACKET_SCHEMA,
            },
            {
                "schema": draft_plan.get("schema"),
                "status": draft_plan.get("status"),
                "node_names": sorted(str(name) for name in node_names),
                "packet_schema": draft_packet.get("schema"),
            },
        ),
    ]


def evaluation_report(index_path: Path, top: int) -> dict[str, Any]:
    index = retrieval.load_index(index_path)
    results = []
    results.extend(evaluate_retrieval(index))
    results.extend(evaluate_debugger(index, top))
    results.extend(evaluate_training(index, top))
    results.extend(evaluate_graph(index, top))
    return {
        "schema": REPORT_SCHEMA,
        "index": str(index_path),
        "case_count": len(results),
        "passed": sum(1 for item in results if item["ok"]),
        "failed": sum(1 for item in results if not item["ok"]),
        "ok": all(item["ok"] for item in results),
        "model_runtime": "not_used",
        "results": results,
    }


def bad_answer_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": BAD_ANSWER_SCHEMA,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": args.case_id,
        "query": args.query,
        "expected_citation": args.expected_citation,
        "actual_output": args.actual_output,
        "failure_mode": args.failure_mode,
        "reviewed_correction": args.reviewed_correction,
        "notes": args.notes,
    }


def load_bad_answer_records(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists():
        return [], []
    records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            invalid.append(
                {
                    "line": line_number,
                    "error": str(exc),
                }
            )
            continue
        if not isinstance(value, dict):
            invalid.append(
                {
                    "line": line_number,
                    "error": "record is not a JSON object",
                }
            )
            continue
        records.append(value)
    return records, invalid


def bad_answer_review(path: Path, min_reviewed_examples: int) -> dict[str, Any]:
    records, invalid = load_bad_answer_records(path)
    schema_mismatch = [
        idx + 1
        for idx, record in enumerate(records)
        if record.get("schema") != BAD_ANSWER_SCHEMA
    ]
    reviewed = [
        record
        for record in records
        if str(record.get("reviewed_correction", "")).strip()
    ]
    unreviewed = [
        record
        for record in records
        if not str(record.get("reviewed_correction", "")).strip()
    ]
    failure_modes: dict[str, int] = {}
    for record in records:
        mode = str(record.get("failure_mode", "")).strip() or "unspecified"
        failure_modes[mode] = failure_modes.get(mode, 0) + 1
    threshold_met = len(reviewed) >= min_reviewed_examples
    return {
        "schema": BAD_ANSWER_REVIEW_SCHEMA,
        "path": str(path),
        "exists": path.exists(),
        "valid_jsonl": not invalid,
        "record_count": len(records),
        "reviewed_count": len(reviewed),
        "unreviewed_count": len(unreviewed),
        "invalid_records": invalid,
        "schema_mismatch_records": schema_mismatch,
        "failure_modes": failure_modes,
        "min_reviewed_examples": min_reviewed_examples,
        "threshold_met": threshold_met,
        "ready_for_finetune_dataset": threshold_met and not invalid and not schema_mismatch,
    }


def list_bad_answers(path: Path, only_unreviewed: bool, only_reviewed: bool) -> dict[str, Any]:
    records, invalid = load_bad_answer_records(path)
    items = []
    for idx, record in enumerate(records, start=1):
        reviewed = bool(str(record.get("reviewed_correction", "")).strip())
        if only_unreviewed and reviewed:
            continue
        if only_reviewed and not reviewed:
            continue
        items.append(
            {
                "record_number": idx,
                "case_id": record.get("case_id", ""),
                "query": record.get("query", ""),
                "expected_citation": record.get("expected_citation", ""),
                "failure_mode": record.get("failure_mode", ""),
                "reviewed": reviewed,
                "notes": record.get("notes", ""),
            }
        )
    return {
        "schema": "cyxwiz.tofix42.phase6.bad_answer_list.v1",
        "path": str(path),
        "exists": path.exists(),
        "valid_jsonl": not invalid,
        "invalid_records": invalid,
        "record_count": len(records),
        "listed_count": len(items),
        "items": items,
    }


def review_queue_items(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    items = []
    for idx, record in enumerate(records, start=1):
        if record.get("schema") != BAD_ANSWER_SCHEMA:
            continue
        if str(record.get("reviewed_correction", "")).strip():
            continue
        items.append(
            {
                "record_number": idx,
                "case_id": record.get("case_id", ""),
                "query": record.get("query", ""),
                "expected_citation": record.get("expected_citation", ""),
                "failure_mode": record.get("failure_mode", ""),
                "actual_output_preview": str(record.get("actual_output", ""))[:240],
                "notes": record.get("notes", ""),
                "review_command": (
                    "python \"docs/Data Studio/tofix42/phase6_eval_capture.py\" "
                    f"review-bad-answer --record-number {idx} "
                    "--reviewed-correction \"<write corrected cited answer>\""
                ),
            }
        )
        if limit > 0 and len(items) >= limit:
            break
    return items


def review_queue(path: Path, limit: int) -> dict[str, Any]:
    records, invalid = load_bad_answer_records(path)
    items = review_queue_items(records, limit)
    unreviewed_count = sum(
        1
        for record in records
        if record.get("schema") == BAD_ANSWER_SCHEMA
        and not str(record.get("reviewed_correction", "")).strip()
    )
    return {
        "schema": BAD_ANSWER_REVIEW_QUEUE_SCHEMA,
        "path": str(path),
        "exists": path.exists(),
        "valid_jsonl": not invalid,
        "invalid_records": invalid,
        "record_count": len(records),
        "unreviewed_count": unreviewed_count,
        "listed_count": len(items),
        "limit": limit,
        "items": items,
    }


def show_bad_answer(path: Path, record_number: int | None, case_id: str) -> dict[str, Any]:
    records, invalid = load_bad_answer_records(path)
    if invalid:
        return {
            "schema": "cyxwiz.tofix42.phase6.bad_answer_detail.v1",
            "ok": False,
            "path": str(path),
            "error": "invalid_jsonl",
            "invalid_records": invalid,
            "record": {},
        }
    if record_number is not None:
        if record_number < 1 or record_number > len(records):
            return {
                "schema": "cyxwiz.tofix42.phase6.bad_answer_detail.v1",
                "ok": False,
                "path": str(path),
                "error": "record_number_out_of_range",
                "invalid_records": [],
                "record": {},
            }
        return {
            "schema": "cyxwiz.tofix42.phase6.bad_answer_detail.v1",
            "ok": True,
            "path": str(path),
            "error": "",
            "invalid_records": [],
            "record_number": record_number,
            "record": records[record_number - 1],
        }
    matches = [
        (idx, record)
        for idx, record in enumerate(records, start=1)
        if str(record.get("case_id", "")) == case_id
    ]
    if not matches:
        return {
            "schema": "cyxwiz.tofix42.phase6.bad_answer_detail.v1",
            "ok": False,
            "path": str(path),
            "error": "no_matching_record",
            "invalid_records": [],
            "record": {},
        }
    if len(matches) > 1:
        return {
            "schema": "cyxwiz.tofix42.phase6.bad_answer_detail.v1",
            "ok": False,
            "path": str(path),
            "error": "ambiguous_case_id_use_record_number",
            "invalid_records": [],
            "matching_record_numbers": [idx for idx, _record in matches],
            "record": {},
        }
    idx, record = matches[0]
    return {
        "schema": "cyxwiz.tofix42.phase6.bad_answer_detail.v1",
        "ok": True,
        "path": str(path),
        "error": "",
        "invalid_records": [],
        "record_number": idx,
        "record": record,
    }


def failure_modes_from_probe_result(result: dict[str, Any]) -> list[str]:
    modes: list[str] = []
    if not result.get("evidence_ok", False):
        modes.append("evidence_not_expected_path")
    if not result.get("runtime_ok", False):
        modes.append("runtime_failed")
    if not result.get("parsed", False):
        modes.append("structured_sections_not_parsed")
    if result.get("sections_missing"):
        modes.append("sections_missing")
    if not result.get("expected_path_in_output", False):
        modes.append("expected_path_missing_from_output")
    if result.get("required_terms_missing"):
        modes.append("required_terms_missing")
    if result.get("forbidden_terms_present"):
        modes.append("forbidden_terms_present")
    if not modes and not result.get("ok", False):
        modes.append("case_not_ok")
    return sorted(set(modes))


def output_from_probe_result(result: dict[str, Any]) -> str:
    for key in ("raw_model_output", "answer", "error"):
        value = result.get(key, "")
        if isinstance(value, str) and value.strip():
            return value
    sections = result.get("model_output_sections", {})
    if isinstance(sections, dict):
        joined = "\n\n".join(
            str(sections.get(key, ""))
            for key in ("answer", "evidence", "unknowns", "unsupported_or_not_implemented")
            if str(sections.get(key, "")).strip()
        )
        if joined.strip():
            return joined
    return "[NO MODEL OUTPUT CAPTURED]"


def bad_answer_identity(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("case_id", "")),
        str(record.get("query", "")),
        str(record.get("failure_mode", "")),
    )


def capture_probe_failures(report_path: Path, output_path: Path, dedupe: bool) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict):
        raise ValueError("Expected probe report JSON object")
    results = [item for item in report.get("results", []) if isinstance(item, dict)]
    existing, invalid_existing = load_bad_answer_records(output_path)
    existing_keys = {bad_answer_identity(record) for record in existing} if dedupe else set()
    records: list[dict[str, Any]] = []
    skipped_duplicates = 0
    skipped_passing = 0
    for result in results:
        if result.get("ok", False):
            skipped_passing += 1
            continue
        modes = failure_modes_from_probe_result(result)
        mode = ",".join(modes) if modes else "case_not_ok"
        record = {
            "schema": BAD_ANSWER_SCHEMA,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "case_id": f"phase2.probe.{result.get('name', '')}",
            "query": result.get("query", ""),
            "expected_citation": result.get("expected_path", ""),
            "actual_output": output_from_probe_result(result),
            "failure_mode": mode,
            "reviewed_correction": "",
            "notes": (
                f"Captured from {report_path}; top_path={result.get('top_path', '')}; "
                f"endpoint={report.get('endpoint', '')}"
            ),
        }
        key = bad_answer_identity(record)
        if dedupe and key in existing_keys:
            skipped_duplicates += 1
            continue
        existing_keys.add(key)
        records.append(record)
    if records:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return {
        "schema": PROBE_FAILURE_CAPTURE_SCHEMA,
        "probe_report": str(report_path),
        "output": str(output_path),
        "report_ok": report.get("ok", False),
        "probe_case_count": len(results),
        "captured_count": len(records),
        "skipped_passing": skipped_passing,
        "skipped_duplicates": skipped_duplicates,
        "existing_invalid_records": invalid_existing,
        "records": records,
    }


def reviewed_example(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": REVIEWED_EXPORT_SCHEMA,
        "case_id": record.get("case_id", ""),
        "query": record.get("query", ""),
        "expected_citation": record.get("expected_citation", ""),
        "bad_output": record.get("actual_output", ""),
        "failure_mode": record.get("failure_mode", ""),
        "corrected_output": record.get("reviewed_correction", ""),
        "notes": record.get("notes", ""),
        "source_record_schema": record.get("schema", ""),
    }


def export_reviewed_corrections(input_path: Path, output_path: Path, min_reviewed_examples: int) -> dict[str, Any]:
    records, invalid = load_bad_answer_records(input_path)
    reviewed = [
        reviewed_example(record)
        for record in records
        if record.get("schema") == BAD_ANSWER_SCHEMA
        and str(record.get("reviewed_correction", "")).strip()
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in reviewed:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")
    return {
        "schema": "cyxwiz.tofix42.phase6.reviewed_correction_export.v1",
        "input": str(input_path),
        "output": str(output_path),
        "valid_jsonl": not invalid,
        "invalid_records": invalid,
        "exported_count": len(reviewed),
        "min_reviewed_examples": min_reviewed_examples,
        "threshold_met": len(reviewed) >= min_reviewed_examples,
        "training_ready": len(reviewed) >= min_reviewed_examples and not invalid,
    }


def record_matches_review_filter(
    record: dict[str, Any],
    case_id: str,
    failure_mode: str,
    query: str,
) -> bool:
    if record.get("schema") != BAD_ANSWER_SCHEMA:
        return False
    if str(record.get("case_id", "")) != case_id:
        return False
    if failure_mode and str(record.get("failure_mode", "")) != failure_mode:
        return False
    if query and str(record.get("query", "")) != query:
        return False
    return True


def write_bad_answer_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    tmp.replace(path)


def review_bad_answer(
    path: Path,
    record_number: int | None,
    case_id: str,
    reviewed_correction: str,
    failure_mode: str,
    query: str,
    notes: str,
    update_all: bool,
) -> dict[str, Any]:
    records, invalid = load_bad_answer_records(path)
    if invalid:
        return {
            "schema": "cyxwiz.tofix42.phase6.bad_answer_review_update.v1",
            "ok": False,
            "path": str(path),
            "matched_count": 0,
            "updated_count": 0,
            "error": "invalid_jsonl",
            "invalid_records": invalid,
        }
    if record_number is not None:
        if record_number < 1 or record_number > len(records):
            return {
                "schema": "cyxwiz.tofix42.phase6.bad_answer_review_update.v1",
                "ok": False,
                "path": str(path),
                "matched_count": 0,
                "updated_count": 0,
                "error": "record_number_out_of_range",
                "invalid_records": [],
            }
        record = records[record_number - 1]
        if record.get("schema") != BAD_ANSWER_SCHEMA:
            return {
                "schema": "cyxwiz.tofix42.phase6.bad_answer_review_update.v1",
                "ok": False,
                "path": str(path),
                "matched_count": 0,
                "updated_count": 0,
                "error": "record_schema_mismatch",
                "invalid_records": [],
            }
        record["reviewed_correction"] = reviewed_correction
        if notes:
            existing_notes = str(record.get("notes", "")).strip()
            record["notes"] = f"{existing_notes}\n{notes}".strip() if existing_notes else notes
        write_bad_answer_records(path, records)
        return {
            "schema": "cyxwiz.tofix42.phase6.bad_answer_review_update.v1",
            "ok": True,
            "path": str(path),
            "matched_count": 1,
            "updated_count": 1,
            "error": "",
            "invalid_records": [],
        }
    matches = [
        idx
        for idx, record in enumerate(records)
        if record_matches_review_filter(record, case_id, failure_mode, query)
    ]
    if not matches:
        return {
            "schema": "cyxwiz.tofix42.phase6.bad_answer_review_update.v1",
            "ok": False,
            "path": str(path),
            "matched_count": 0,
            "updated_count": 0,
            "error": "no_matching_record",
            "invalid_records": [],
        }
    if len(matches) > 1 and not update_all:
        return {
            "schema": "cyxwiz.tofix42.phase6.bad_answer_review_update.v1",
            "ok": False,
            "path": str(path),
            "matched_count": len(matches),
            "updated_count": 0,
            "error": "ambiguous_match_use_update_all_or_more_filters",
            "invalid_records": [],
        }
    update_indexes = matches if update_all else matches[:1]
    for idx in update_indexes:
        records[idx]["reviewed_correction"] = reviewed_correction
        if notes:
            existing_notes = str(records[idx].get("notes", "")).strip()
            records[idx]["notes"] = f"{existing_notes}\n{notes}".strip() if existing_notes else notes
    write_bad_answer_records(path, records)
    return {
        "schema": "cyxwiz.tofix42.phase6.bad_answer_review_update.v1",
        "ok": True,
        "path": str(path),
        "matched_count": len(matches),
        "updated_count": len(update_indexes),
        "error": "",
        "invalid_records": [],
    }


def cmd_run(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}", file=sys.stderr)
        print("Run phase1a_retrieval.py build first.", file=sys.stderr)
        return 2
    report = evaluation_report(args.index, args.top)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print(f"OK: {report['ok']}")
        print(f"Cases: {report['case_count']} passed={report['passed']} failed={report['failed']}")
        for item in report["results"]:
            status = "PASS" if item["ok"] else "FAIL"
            print(f"{status}: {item['id']}")
    return 0 if report["ok"] else 1


def cmd_list_cases(args: argparse.Namespace) -> int:
    report = evaluation_report(args.index, args.top) if args.index.exists() else {"results": []}
    payload = {
        "schema": "cyxwiz.tofix42.phase6.eval_cases.v1",
        "cases": [
            {"id": item["id"], "phase": item["phase"], "expected": item["expected"]}
            for item in report.get("results", [])
        ],
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        for item in payload["cases"]:
            print(f"{item['phase']}: {item['id']}")
    return 0


def cmd_capture_bad_answer(args: argparse.Namespace) -> int:
    record = bad_answer_record(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    if args.json:
        print(json.dumps(record, indent=2, ensure_ascii=True))
    else:
        print(f"Captured bad-answer record: {args.output}")
    return 0


def cmd_review_status(args: argparse.Namespace) -> int:
    review = bad_answer_review(args.bad_answers, args.min_reviewed_examples)
    if args.json:
        print(json.dumps(review, indent=2, ensure_ascii=True))
    else:
        print(f"Path: {review['path']}")
        print(f"Exists: {review['exists']}")
        print(f"Valid JSONL: {review['valid_jsonl']}")
        print(
            f"Records: {review['record_count']} "
            f"reviewed={review['reviewed_count']} unreviewed={review['unreviewed_count']}"
        )
        print(
            f"Threshold: {review['reviewed_count']}/{review['min_reviewed_examples']} "
            f"met={review['threshold_met']}"
        )
        if review["invalid_records"]:
            print("Invalid records:")
            for item in review["invalid_records"]:
                print(f"- line {item.get('line')}: {item.get('error')}")
        if review["schema_mismatch_records"]:
            print(f"Schema mismatch record numbers: {review['schema_mismatch_records']}")
    if not review["valid_jsonl"] or review["schema_mismatch_records"]:
        return 1
    if args.require_threshold and not review["threshold_met"]:
        return 1
    return 0


def cmd_list_bad_answers(args: argparse.Namespace) -> int:
    listing = list_bad_answers(args.bad_answers, args.unreviewed, args.reviewed)
    if args.json:
        print(json.dumps(listing, indent=2, ensure_ascii=True))
    else:
        print(f"Path: {listing['path']}")
        print(f"Exists: {listing['exists']}")
        print(f"Valid JSONL: {listing['valid_jsonl']}")
        print(f"Records: {listing['record_count']} listed={listing['listed_count']}")
        for item in listing["items"]:
            status = "reviewed" if item["reviewed"] else "unreviewed"
            print(f"- #{item['record_number']} {status}: {item['case_id']}")
            print(f"  failure_mode: {item['failure_mode']}")
            print(f"  expected_citation: {item['expected_citation']}")
            if item["query"]:
                print(f"  query: {item['query']}")
        if listing["invalid_records"]:
            print("Invalid records:")
            for item in listing["invalid_records"]:
                print(f"- line {item.get('line')}: {item.get('error')}")
    return 0 if listing["valid_jsonl"] else 1


def cmd_show_bad_answer(args: argparse.Namespace) -> int:
    if args.record_number is None and not args.case_id:
        print("error: provide --record-number or --case-id", file=sys.stderr)
        return 2
    detail = show_bad_answer(args.bad_answers, args.record_number, args.case_id)
    if args.json:
        print(json.dumps(detail, indent=2, ensure_ascii=True))
    else:
        print(f"Path: {detail['path']}")
        print(f"OK: {detail['ok']}")
        if detail["error"]:
            print(f"Error: {detail['error']}")
            if detail.get("matching_record_numbers"):
                print(f"Matching record numbers: {detail['matching_record_numbers']}")
            return 1
        record = detail["record"]
        print(f"Record: #{detail.get('record_number', '')}")
        print(f"Case: {record.get('case_id', '')}")
        print(f"Failure mode: {record.get('failure_mode', '')}")
        print(f"Expected citation: {record.get('expected_citation', '')}")
        print(f"Reviewed: {bool(str(record.get('reviewed_correction', '')).strip())}")
        print("\nQuery:")
        print(record.get("query", ""))
        print("\nActual output:")
        print(record.get("actual_output", ""))
        print("\nReviewed correction:")
        print(record.get("reviewed_correction", ""))
        if record.get("notes"):
            print("\nNotes:")
            print(record.get("notes", ""))
    return 0 if detail["ok"] else 1


def cmd_review_queue(args: argparse.Namespace) -> int:
    queue = review_queue(args.bad_answers, args.limit)
    if args.json:
        print(json.dumps(queue, indent=2, ensure_ascii=True))
    else:
        print(f"Path: {queue['path']}")
        print(f"Exists: {queue['exists']}")
        print(f"Valid JSONL: {queue['valid_jsonl']}")
        print(f"Unreviewed: {queue['unreviewed_count']} listed={queue['listed_count']}")
        for item in queue["items"]:
            print(f"- #{item['record_number']}: {item['case_id']}")
            print(f"  failure_mode: {item['failure_mode']}")
            print(f"  expected_citation: {item['expected_citation']}")
            if item["query"]:
                print(f"  query: {item['query']}")
            if item["actual_output_preview"]:
                print(f"  actual_output_preview: {item['actual_output_preview']}")
            print(f"  review_command: {item['review_command']}")
        if queue["invalid_records"]:
            print("Invalid records:")
            for item in queue["invalid_records"]:
                print(f"- line {item.get('line')}: {item.get('error')}")
    return 0 if queue["valid_jsonl"] else 1


def cmd_capture_probe_failures(args: argparse.Namespace) -> int:
    capture = capture_probe_failures(args.report, args.output, not args.no_dedupe)
    if args.json:
        print(json.dumps(capture, indent=2, ensure_ascii=True))
    else:
        print(f"Probe report: {capture['probe_report']}")
        print(f"Output: {capture['output']}")
        print(f"Captured: {capture['captured_count']}")
        print(f"Skipped passing: {capture['skipped_passing']}")
        print(f"Skipped duplicates: {capture['skipped_duplicates']}")
        if capture["existing_invalid_records"]:
            print("Existing invalid JSONL records:")
            for item in capture["existing_invalid_records"]:
                print(f"- line {item.get('line')}: {item.get('error')}")
    return 0 if not capture["existing_invalid_records"] else 1


def cmd_export_reviewed(args: argparse.Namespace) -> int:
    export = export_reviewed_corrections(args.bad_answers, args.output, args.min_reviewed_examples)
    if args.json:
        print(json.dumps(export, indent=2, ensure_ascii=True))
    else:
        print(f"Input: {export['input']}")
        print(f"Output: {export['output']}")
        print(f"Exported: {export['exported_count']}")
        print(f"Threshold: {export['exported_count']}/{export['min_reviewed_examples']} met={export['threshold_met']}")
        print(f"Training ready: {export['training_ready']}")
        if export["invalid_records"]:
            print("Invalid records:")
            for item in export["invalid_records"]:
                print(f"- line {item.get('line')}: {item.get('error')}")
    if not export["valid_jsonl"]:
        return 1
    if args.require_threshold and not export["threshold_met"]:
        return 1
    return 0


def cmd_review_bad_answer(args: argparse.Namespace) -> int:
    if args.record_number is None and not args.case_id:
        print("error: provide --record-number or --case-id", file=sys.stderr)
        return 2
    update = review_bad_answer(
        args.bad_answers,
        args.record_number,
        args.case_id,
        args.reviewed_correction,
        args.failure_mode,
        args.query,
        args.notes,
        args.update_all,
    )
    if args.json:
        print(json.dumps(update, indent=2, ensure_ascii=True))
    else:
        print(f"Path: {update['path']}")
        print(f"Matched: {update['matched_count']}")
        print(f"Updated: {update['updated_count']}")
        print(f"OK: {update['ok']}")
        if update["error"]:
            print(f"Error: {update['error']}")
    return 0 if update["ok"] else 1


def synthetic_probe_failure_report() -> dict[str, Any]:
    return {
        "schema": "cyxwiz.tofix42.phase2.probe_suite.v1",
        "endpoint": "http://127.0.0.1:8765/completion",
        "case_count": 1,
        "ok": False,
        "results": [
            {
                "name": "dataloader_pin_memory_truth",
                "query": "DataLoader pin_memory unsupported current batchers compatibility",
                "expected_path": "cyxwiz-engine/src/core/graph_compiler.cpp",
                "top_path": "cyxwiz-engine/src/core/graph_compiler.cpp",
                "evidence_ok": True,
                "runtime_ok": True,
                "parsed": True,
                "sections_missing": [],
                "required_terms_missing": ["ignored"],
                "forbidden_terms_present": [],
                "expected_path_in_output": False,
                "ok": False,
                "answer": "pin_memory enables fast CUDA pinned transfer.",
                "error": "",
            }
        ],
    }


def cmd_check(args: argparse.Namespace) -> int:
    report = evaluation_report(args.index, args.top)
    synthetic_review = bad_answer_review(Path("__missing_phase6_bad_answers_for_check__.jsonl"), 1)
    synthetic_result = synthetic_probe_failure_report()["results"][0]
    synthetic_export = reviewed_example(
        {
            "schema": BAD_ANSWER_SCHEMA,
            "case_id": "sample",
            "query": "What happened?",
            "expected_citation": "source.cpp",
            "actual_output": "unsupported claim",
            "failure_mode": "unsupported_claim",
            "reviewed_correction": "Use only cited evidence.",
            "notes": "sample",
        }
    )
    synthetic_records = [
        {
            "schema": BAD_ANSWER_SCHEMA,
            "case_id": "case.one",
            "query": "question",
            "expected_citation": "source.cpp",
            "actual_output": "bad",
            "failure_mode": "missing_citation",
            "reviewed_correction": "",
            "notes": "",
        }
    ]
    synthetic_match = [
        idx
        for idx, record in enumerate(synthetic_records)
        if record_matches_review_filter(record, "case.one", "missing_citation", "")
    ]
    synthetic_listing = {
        "items": [
            {
                "case_id": record.get("case_id", ""),
                "reviewed": bool(str(record.get("reviewed_correction", "")).strip()),
            }
            for record in synthetic_records
        ]
    }
    synthetic_detail = {
        "ok": True,
        "record_number": 1,
        "record": synthetic_records[0],
    }
    synthetic_queue = review_queue_items(synthetic_records, 10)
    checks = [
        report["schema"] == REPORT_SCHEMA,
        report["case_count"] >= 10,
        report["ok"],
        any(item["id"] == "retrieval.debug_trace_record_definition" for item in report["results"]),
        any(item["id"] == "debugger.explanation_required_column_missing" for item in report["results"]),
        any(item["id"] == "training.terminal_reason" for item in report["results"]),
        any(item["id"] == "graph.draft_plan_tfidf_mlp" for item in report["results"]),
        synthetic_review["schema"] == BAD_ANSWER_REVIEW_SCHEMA,
        synthetic_review["record_count"] == 0,
        not synthetic_review["threshold_met"],
        "required_terms_missing" in failure_modes_from_probe_result(synthetic_result),
        "expected_path_missing_from_output" in failure_modes_from_probe_result(synthetic_result),
        output_from_probe_result(synthetic_result) == "pin_memory enables fast CUDA pinned transfer.",
        synthetic_export["schema"] == REVIEWED_EXPORT_SCHEMA,
        synthetic_export["corrected_output"] == "Use only cited evidence.",
        synthetic_match == [0],
        synthetic_listing["items"][0]["case_id"] == "case.one",
        not synthetic_listing["items"][0]["reviewed"],
        synthetic_detail["record"]["actual_output"] == "bad",
        synthetic_queue[0]["record_number"] == 1,
        "--record-number 1" in synthetic_queue[0]["review_command"],
    ]
    if not all(checks):
        print(json.dumps(report, indent=2, sort_keys=True))
        print("Phase 6 evaluation check failed", file=sys.stderr)
        return 1
    print("All Phase 6 evaluation checks passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run deterministic evaluation cases")
    run.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    run.add_argument("--top", type=int, default=5)
    run.add_argument("--output", type=Path)
    run.add_argument("--json", action="store_true")
    run.set_defaults(func=cmd_run)

    list_cases = sub.add_parser("list-cases", help="list deterministic evaluation cases")
    list_cases.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    list_cases.add_argument("--top", type=int, default=5)
    list_cases.add_argument("--json", action="store_true")
    list_cases.set_defaults(func=cmd_list_cases)

    capture = sub.add_parser("capture-bad-answer", help="append a bad-answer QA record as JSONL")
    capture.add_argument("--case-id", required=True)
    capture.add_argument("--query", required=True)
    capture.add_argument("--expected-citation", default="")
    capture.add_argument("--actual-output", required=True)
    capture.add_argument("--failure-mode", required=True)
    capture.add_argument("--reviewed-correction", default="")
    capture.add_argument("--notes", default="")
    capture.add_argument("--output", type=Path, default=DEFAULT_BAD_ANSWERS)
    capture.add_argument("--json", action="store_true")
    capture.set_defaults(func=cmd_capture_bad_answer)

    review = sub.add_parser("review-status", help="summarize reviewed bad-answer corrections")
    review.add_argument("--bad-answers", type=Path, default=DEFAULT_BAD_ANSWERS)
    review.add_argument("--min-reviewed-examples", type=int, default=20)
    review.add_argument(
        "--require-threshold",
        action="store_true",
        help="exit non-zero when reviewed corrections are below the threshold",
    )
    review.add_argument("--json", action="store_true")
    review.set_defaults(func=cmd_review_status)

    list_bad = sub.add_parser("list-bad-answers", help="list captured bad-answer records")
    list_bad.add_argument("--bad-answers", type=Path, default=DEFAULT_BAD_ANSWERS)
    list_bad.add_argument("--unreviewed", action="store_true", help="show only unreviewed records")
    list_bad.add_argument("--reviewed", action="store_true", help="show only reviewed records")
    list_bad.add_argument("--json", action="store_true")
    list_bad.set_defaults(func=cmd_list_bad_answers)

    show_bad = sub.add_parser("show-bad-answer", help="show one captured bad-answer record")
    show_bad.add_argument("--bad-answers", type=Path, default=DEFAULT_BAD_ANSWERS)
    show_bad.add_argument("--record-number", type=int, default=None)
    show_bad.add_argument("--case-id", default="")
    show_bad.add_argument("--json", action="store_true")
    show_bad.set_defaults(func=cmd_show_bad_answer)

    review_queue_parser = sub.add_parser(
        "review-queue",
        help="show unreviewed bad-answer records with review command templates",
    )
    review_queue_parser.add_argument("--bad-answers", type=Path, default=DEFAULT_BAD_ANSWERS)
    review_queue_parser.add_argument("--limit", type=int, default=10)
    review_queue_parser.add_argument("--json", action="store_true")
    review_queue_parser.set_defaults(func=cmd_review_queue)

    capture_failures = sub.add_parser(
        "capture-probe-failures",
        help="append failed Phase 2 probe cases to the bad-answer JSONL log",
    )
    capture_failures.add_argument("--report", type=Path, required=True)
    capture_failures.add_argument("--output", type=Path, default=DEFAULT_BAD_ANSWERS)
    capture_failures.add_argument("--no-dedupe", action="store_true")
    capture_failures.add_argument("--json", action="store_true")
    capture_failures.set_defaults(func=cmd_capture_probe_failures)

    export_reviewed = sub.add_parser(
        "export-reviewed",
        help="export reviewed bad-answer corrections as JSONL training candidates",
    )
    export_reviewed.add_argument("--bad-answers", type=Path, default=DEFAULT_BAD_ANSWERS)
    export_reviewed.add_argument("--output", type=Path, default=DEFAULT_REVIEWED_EXPORT)
    export_reviewed.add_argument("--min-reviewed-examples", type=int, default=20)
    export_reviewed.add_argument("--require-threshold", action="store_true")
    export_reviewed.add_argument("--json", action="store_true")
    export_reviewed.set_defaults(func=cmd_export_reviewed)

    review_bad = sub.add_parser(
        "review-bad-answer",
        help="mark matching bad-answer JSONL records as reviewed",
    )
    review_bad.add_argument("--bad-answers", type=Path, default=DEFAULT_BAD_ANSWERS)
    review_bad.add_argument("--record-number", type=int, default=None)
    review_bad.add_argument("--case-id", default="")
    review_bad.add_argument("--reviewed-correction", required=True)
    review_bad.add_argument("--failure-mode", default="")
    review_bad.add_argument("--query", default="")
    review_bad.add_argument("--notes", default="")
    review_bad.add_argument("--update-all", action="store_true")
    review_bad.add_argument("--json", action="store_true")
    review_bad.set_defaults(func=cmd_review_bad_answer)

    check = sub.add_parser("check", help="run deterministic Phase 6 checks")
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
