#!/usr/bin/env python3
"""Summarize current tofix42 status and next actions.

This command is read-only and local. It aggregates existing deterministic
reports without starting model servers, launching Studio, mutating graphs, or
running training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import phase1a_retrieval as retrieval
import phase2_endpoint_doctor
import phase6_eval_capture
import phase7_finetune_decision
import tofix42_check_all


STATUS_SCHEMA = "cyxwiz.tofix42.status.v1"


def index_status(index_path: Path, config_path: Path) -> dict[str, Any]:
    if not index_path.exists():
        return {
            "exists": False,
            "fresh": False,
            "indexed_files": 0,
            "current_files": 0,
            "indexed_chunks": 0,
            "status": "missing",
        }
    index = retrieval.load_index(index_path)
    config = retrieval.load_config(config_path)
    indexed_by_path = {
        item["path"]: item
        for item in index.get("files", [])
        if isinstance(item, dict) and "path" in item
    }
    current_files = [
        retrieval.file_metadata(path)
        for path in retrieval.iter_initial_files(config["source_patterns"])
    ]
    current_by_path = {item.path: item for item in current_files}
    changed = [
        path
        for path, current in current_by_path.items()
        if indexed_by_path.get(path, {}).get("content_hash") != current.content_hash
    ]
    added = [path for path in current_by_path if path not in indexed_by_path]
    removed = [path for path in indexed_by_path if path not in current_by_path]
    config_changed = index.get("config_hash") != config.get("config_hash")
    fresh = not changed and not added and not removed and not config_changed
    return {
        "exists": True,
        "fresh": fresh,
        "indexed_files": len(indexed_by_path),
        "current_files": len(current_by_path),
        "indexed_chunks": index.get("chunk_count", len(index.get("chunks", []))),
        "status": "fresh" if fresh else "stale",
        "changed": sorted(changed),
        "added": sorted(added),
        "removed": sorted(removed),
        "config_changed": config_changed,
    }


def endpoint_status(endpoint: str, max_tokens: int, timeout_seconds: int) -> dict[str, Any]:
    if not endpoint:
        return {
            "checked": False,
            "ok": None,
            "reachable": None,
            "runtime_kind": "not_checked",
        }
    report = phase2_endpoint_doctor.endpoint_doctor(endpoint, max_tokens, timeout_seconds)
    return {
        "checked": True,
        "schema": report.get("schema", ""),
        "endpoint": report.get("endpoint", endpoint),
        "ok": report.get("ok", False),
        "local_endpoint": report.get("local_endpoint", False),
        "reachable": report.get("reachable", False),
        "runtime_kind": report.get("runtime_kind", "unknown"),
        "stub_runtime_detected": report.get("stub_runtime_detected", False),
        "parsed": report.get("parsed", False),
        "sections_present": report.get("sections_present", []),
        "sections_missing": report.get("sections_missing", []),
        "error": report.get("error", ""),
        "note": report.get("note", ""),
    }


def make_status(args: argparse.Namespace) -> dict[str, Any]:
    index = index_status(args.index, args.config)
    endpoint = endpoint_status(
        args.endpoint,
        args.endpoint_max_tokens,
        args.endpoint_timeout_seconds,
    )
    check_report = tofix42_check_all.run_all(args.timeout_seconds, stop_on_fail=False)
    eval_report = phase6_eval_capture.evaluation_report(args.index, args.top) if args.index.exists() else {
        "ok": False,
        "case_count": 0,
        "passed": 0,
        "failed": 0,
    }
    decision = phase7_finetune_decision.make_decision(
        index_path=args.index,
        bad_answers_path=args.bad_answers,
        real_model_probe_report=args.real_model_probe_report,
        min_reviewed_examples=args.min_reviewed_examples,
        top=args.top,
    ) if args.index.exists() else {
        "recommendation": "defer_fine_tuning",
        "approved": False,
        "blocking_criteria": ["index_missing"],
        "next_actions": ["Build the Phase 1A index."],
    }
    bad_answer_review = phase6_eval_capture.bad_answer_review(
        args.bad_answers,
        args.min_reviewed_examples,
    )
    blocked = []
    if not index["fresh"]:
        blocked.append("index_not_fresh")
    if not check_report["ok"]:
        blocked.append("deterministic_checks_failed")
    blocked.extend(decision.get("blocking_criteria", []))
    return {
        "schema": STATUS_SCHEMA,
        "ok": index["fresh"] and check_report["ok"],
        "index": index,
        "deterministic_checks": {
            "ok": check_report["ok"],
            "check_count": check_report["check_count"],
            "passed": check_report["passed"],
            "failed": check_report["failed"],
        },
        "evaluation": {
            "ok": eval_report.get("ok", False),
            "case_count": eval_report.get("case_count", 0),
            "passed": eval_report.get("passed", 0),
            "failed": eval_report.get("failed", 0),
        },
        "endpoint": endpoint,
        "bad_answers": {
            "path": bad_answer_review["path"],
            "exists": bad_answer_review["exists"],
            "valid_jsonl": bad_answer_review["valid_jsonl"],
            "record_count": bad_answer_review["record_count"],
            "reviewed_count": bad_answer_review["reviewed_count"],
            "unreviewed_count": bad_answer_review["unreviewed_count"],
            "min_reviewed_examples": bad_answer_review["min_reviewed_examples"],
            "threshold_met": bad_answer_review["threshold_met"],
        },
        "fine_tuning": {
            "recommendation": decision.get("recommendation", ""),
            "approved": decision.get("approved", False),
            "blocking_criteria": decision.get("blocking_criteria", []),
        },
        "blocked_or_pending": sorted(set(blocked)),
        "next_actions": decision.get("next_actions", []),
        "non_goals": [
            "No model server is started.",
            "No graph or source mutation is performed.",
            "No training or fine-tuning is performed.",
        ],
    }


def cmd_show(args: argparse.Namespace) -> int:
    status = make_status(args)
    if args.json:
        print(json.dumps(status, indent=2, ensure_ascii=True))
    else:
        print(f"OK: {status['ok']}")
        print(f"Index: {status['index']['status']} files={status['index']['indexed_files']} chunks={status['index']['indexed_chunks']}")
        checks = status["deterministic_checks"]
        print(f"Checks: ok={checks['ok']} passed={checks['passed']} failed={checks['failed']}")
        eval_report = status["evaluation"]
        print(f"Evaluation: ok={eval_report['ok']} passed={eval_report['passed']} failed={eval_report['failed']}")
        endpoint = status["endpoint"]
        if endpoint["checked"]:
            print(
                "Endpoint: "
                f"reachable={endpoint['reachable']} "
                f"runtime_kind={endpoint['runtime_kind']} "
                f"parsed={endpoint['parsed']}"
            )
            if endpoint["error"]:
                print(f"Endpoint error: {endpoint['error']}")
        bad_answers = status["bad_answers"]
        print(
            "Bad answers: "
            f"records={bad_answers['record_count']} "
            f"reviewed={bad_answers['reviewed_count']} "
            f"threshold={bad_answers['min_reviewed_examples']} "
            f"met={bad_answers['threshold_met']}"
        )
        fine_tuning = status["fine_tuning"]
        print(f"Fine-tuning: {fine_tuning['recommendation']} approved={fine_tuning['approved']}")
        if status["blocked_or_pending"]:
            print("Blocked or pending:")
            for item in status["blocked_or_pending"]:
                print(f"- {item}")
        if status["next_actions"]:
            print("Next actions:")
            for item in status["next_actions"]:
                print(f"- {item}")
    return 0 if status["ok"] else 1


def cmd_check(args: argparse.Namespace) -> int:
    status = make_status(args)
    checks = [
        status["schema"] == STATUS_SCHEMA,
        status["index"]["fresh"],
        status["deterministic_checks"]["ok"],
        status["evaluation"]["ok"],
        status["endpoint"]["runtime_kind"] == "not_checked" or status["endpoint"]["checked"],
        status["bad_answers"]["valid_jsonl"],
        status["fine_tuning"]["recommendation"] == "defer_fine_tuning",
        "real_local_model_probe_passed" in status["fine_tuning"]["blocking_criteria"],
    ]
    if not all(checks):
        print(json.dumps(status, indent=2, sort_keys=True))
        print("tofix42 status check failed", file=sys.stderr)
        return 1
    print("tofix42 status check passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show", help="show current tofix42 status")
    show.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    show.add_argument("--config", type=Path, default=retrieval.DEFAULT_CONFIG)
    show.add_argument("--bad-answers", type=Path, default=phase6_eval_capture.DEFAULT_BAD_ANSWERS)
    show.add_argument("--real-model-probe-report", type=Path, default=None)
    show.add_argument("--min-reviewed-examples", type=int, default=20)
    show.add_argument("--top", type=int, default=5)
    show.add_argument("--timeout-seconds", type=int, default=120)
    show.add_argument("--endpoint", default="", help="optional local JSON endpoint to probe")
    show.add_argument("--endpoint-max-tokens", type=int, default=128)
    show.add_argument("--endpoint-timeout-seconds", type=int, default=10)
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=cmd_show)

    check = sub.add_parser("check", help="run status self gate")
    check.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    check.add_argument("--config", type=Path, default=retrieval.DEFAULT_CONFIG)
    check.add_argument("--bad-answers", type=Path, default=phase6_eval_capture.DEFAULT_BAD_ANSWERS)
    check.add_argument("--real-model-probe-report", type=Path, default=None)
    check.add_argument("--min-reviewed-examples", type=int, default=20)
    check.add_argument("--top", type=int, default=5)
    check.add_argument("--timeout-seconds", type=int, default=120)
    check.add_argument("--endpoint", default="", help="optional local JSON endpoint to probe")
    check.add_argument("--endpoint-max-tokens", type=int, default=128)
    check.add_argument("--endpoint-timeout-seconds", type=int, default=10)
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
