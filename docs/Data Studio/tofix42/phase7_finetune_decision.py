#!/usr/bin/env python3
"""Phase 7 fine-tuning readiness decision for tofix42.

This is a decision gate, not a training script. It keeps retrieval as the truth
layer and recommends deferring fine-tuning unless deterministic QA, real local
model probes, graph validation evidence, and reviewed correction examples are
already available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import phase1a_retrieval as retrieval
import phase6_eval_capture


DECISION_SCHEMA = "cyxwiz.tofix42.phase7.finetune_decision.v1"
STUB_MARKERS = (
    "Stub runtime received",
    "Real answer quality is not tested by this stub",
    "No real model inference was performed",
    "A real local model answer has not been validated by this stub",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def load_bad_answers(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            records.append(
                {
                    "schema": phase6_eval_capture.BAD_ANSWER_SCHEMA,
                    "case_id": f"invalid_json_line_{line_number}",
                    "reviewed_correction": "",
                    "parse_error": True,
                }
            )
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def criterion(
    criterion_id: str,
    status: str,
    title: str,
    detail: str,
    evidence: list[str],
) -> dict[str, Any]:
    return {
        "id": criterion_id,
        "status": status,
        "title": title,
        "detail": detail,
        "evidence": evidence,
    }


def real_model_probe_ok(report_path: Path | None) -> tuple[bool, list[str]]:
    if report_path is None:
        return False, ["real_model_probe_report="]
    if not report_path.exists():
        return False, [f"real_model_probe_report={report_path}", "report_exists=False"]
    report = load_json(report_path)
    results = [item for item in report.get("results", []) if isinstance(item, dict)]
    text = "\n".join(
        json.dumps(item.get("model_output_sections", {}), ensure_ascii=True)
        + "\n"
        + str(item.get("answer", ""))
        + "\n"
        + str(item.get("raw_model_output", ""))
        for item in results
    )
    stub_detected = any(marker in text for marker in STUB_MARKERS)
    return bool(report.get("ok")) and not stub_detected, [
        f"real_model_probe_report={report_path}",
        f"report.schema={report.get('schema', '')}",
        f"report.ok={report.get('ok', False)}",
        f"report.case_count={report.get('case_count', '')}",
        f"stub_runtime_detected={stub_detected}",
    ]


def make_decision(
    *,
    index_path: Path,
    bad_answers_path: Path,
    real_model_probe_report: Path | None,
    min_reviewed_examples: int,
    top: int,
) -> dict[str, Any]:
    phase6_report = phase6_eval_capture.evaluation_report(index_path, top)
    bad_answers = load_bad_answers(bad_answers_path)
    reviewed_records = [
        record
        for record in bad_answers
        if str(record.get("reviewed_correction", "")).strip()
    ]
    graph_cases = [
        item
        for item in phase6_report.get("results", [])
        if str(item.get("id", "")).startswith("graph.")
    ]
    graph_cases_ok = bool(graph_cases) and all(item.get("ok") for item in graph_cases)
    probe_ok, probe_evidence = real_model_probe_ok(real_model_probe_report)

    criteria = [
        criterion(
            "rag_deterministic_eval_passed",
            "pass" if phase6_report.get("ok") else "fail",
            "Deterministic RAG evaluation passes",
            "Phase 6 must pass before model specialization is considered.",
            [
                f"phase6.schema={phase6_report.get('schema', '')}",
                f"phase6.ok={phase6_report.get('ok', False)}",
                f"phase6.case_count={phase6_report.get('case_count', 0)}",
                f"phase6.failed={phase6_report.get('failed', 0)}",
            ],
        ),
        criterion(
            "real_local_model_probe_passed",
            "pass" if probe_ok else "fail",
            "Real localhost JSON model probe passes",
            "A real local model must satisfy the strict cited-answer contract before fine-tuning.",
            probe_evidence,
        ),
        criterion(
            "reviewed_corrections_available",
            "pass" if len(reviewed_records) >= min_reviewed_examples else "fail",
            "Reviewed correction examples are available",
            "Fine-tuning needs enough reviewed bad-answer corrections to justify the maintenance cost.",
            [
                f"bad_answers_path={bad_answers_path}",
                f"bad_answer_records={len(bad_answers)}",
                f"reviewed_correction_records={len(reviewed_records)}",
                f"min_reviewed_examples={min_reviewed_examples}",
            ],
        ),
        criterion(
            "graph_draft_validation_evidence",
            "pass" if graph_cases_ok else "fail",
            "Graph draft and audit fixtures pass",
            "Graph-related training data should not be used until graph plan fixtures are reproducible.",
            [
                f"graph_case_count={len(graph_cases)}",
                f"graph_cases_ok={graph_cases_ok}",
                "graph_case_ids="
                + ",".join(str(item.get("id", "")) for item in graph_cases),
            ],
        ),
        criterion(
            "retrieval_remains_truth_layer",
            "pass",
            "Retrieval remains mandatory",
            "The decision keeps retrieval citations as the source of truth regardless of model choice.",
            [
                f"index={index_path}",
                "answer_contract_requires_citations=True",
                "fine_tuning_must_not_replace_retrieval=True",
            ],
        ),
    ]

    blocking = [item for item in criteria if item["status"] != "pass"]
    recommendation = "approve_fine_tuning_experiment" if not blocking else "defer_fine_tuning"
    return {
        "schema": DECISION_SCHEMA,
        "recommendation": recommendation,
        "approved": recommendation == "approve_fine_tuning_experiment",
        "criteria": criteria,
        "blocking_criteria": [item["id"] for item in blocking],
        "next_actions": next_actions(blocking),
        "non_goals": [
            "Do not train or download a model from this command.",
            "Do not weaken citation requirements.",
            "Do not use unreviewed bad answers as training data.",
        ],
    }


def next_actions(blocking: list[dict[str, Any]]) -> list[str]:
    if not blocking:
        return [
            "Write an experiment plan with dataset provenance and rollback criteria.",
            "Keep retrieval evidence mandatory in all prompts and evaluations.",
        ]
    actions: list[str] = []
    blocked = {item["id"] for item in blocking}
    if "real_local_model_probe_passed" in blocked:
        actions.append("Run phase2_real_model_check.py against a real localhost JSON model and save the report.")
    if "reviewed_corrections_available" in blocked:
        actions.append("Capture and review more bad-answer corrections with Phase 6 JSONL records.")
    if "rag_deterministic_eval_passed" in blocked:
        actions.append("Fix deterministic retrieval/context regressions before evaluating models.")
    if "graph_draft_validation_evidence" in blocked:
        actions.append("Fix graph audit or draft-plan fixture failures before graph-related training data.")
    return actions


def cmd_decide(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"Index not found: {args.index}", file=sys.stderr)
        print("Run phase1a_retrieval.py build first.", file=sys.stderr)
        return 2
    decision = make_decision(
        index_path=args.index,
        bad_answers_path=args.bad_answers,
        real_model_probe_report=args.real_model_probe_report,
        min_reviewed_examples=args.min_reviewed_examples,
        top=args.top,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(decision, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(decision, indent=2, ensure_ascii=True))
    else:
        print(f"Recommendation: {decision['recommendation']}")
        print(f"Approved: {decision['approved']}")
        if decision["blocking_criteria"]:
            print("Blocking criteria:")
            for item in decision["blocking_criteria"]:
                print(f"- {item}")
        print("Next actions:")
        for item in decision["next_actions"]:
            print(f"- {item}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    decision = make_decision(
        index_path=args.index,
        bad_answers_path=args.bad_answers,
        real_model_probe_report=args.real_model_probe_report,
        min_reviewed_examples=args.min_reviewed_examples,
        top=args.top,
    )
    checks = [
        decision["schema"] == DECISION_SCHEMA,
        decision["recommendation"] == "defer_fine_tuning",
        "real_local_model_probe_passed" in decision["blocking_criteria"],
        "reviewed_corrections_available" in decision["blocking_criteria"],
        any(item["id"] == "retrieval_remains_truth_layer" and item["status"] == "pass" for item in decision["criteria"]),
    ]
    if not all(checks):
        print(json.dumps(decision, indent=2, sort_keys=True))
        print("Phase 7 fine-tuning decision check failed", file=sys.stderr)
        return 1
    print("All Phase 7 fine-tuning decision checks passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    decide = sub.add_parser("decide", help="emit a fine-tuning readiness decision")
    decide.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    decide.add_argument("--bad-answers", type=Path, default=phase6_eval_capture.DEFAULT_BAD_ANSWERS)
    decide.add_argument("--real-model-probe-report", type=Path, default=None)
    decide.add_argument("--min-reviewed-examples", type=int, default=20)
    decide.add_argument("--top", type=int, default=5)
    decide.add_argument("--output", type=Path)
    decide.add_argument("--json", action="store_true")
    decide.set_defaults(func=cmd_decide)

    check = sub.add_parser("check", help="run deterministic Phase 7 checks")
    check.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    check.add_argument("--bad-answers", type=Path, default=phase6_eval_capture.DEFAULT_BAD_ANSWERS)
    check.add_argument("--real-model-probe-report", type=Path, default=None)
    check.add_argument("--min-reviewed-examples", type=int, default=20)
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
