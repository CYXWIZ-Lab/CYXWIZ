#!/usr/bin/env python3
"""Probe a local Phase 2 JSON runtime endpoint.

This is a reachability and shape check. It does not run the full probe suite,
does not start a server, and does not prove answer quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import phase1b_answer as answer_adapter
import phase2_probe_report_check


REPORT_SCHEMA = "cyxwiz.tofix42.phase2.endpoint_doctor.v1"
DOCTOR_PROMPT = "\n".join(
    [
        "Return these headings exactly:",
        "Answer: Say endpoint doctor response.",
        "Evidence: Say endpoint doctor has no evidence.",
        "Unknowns: Say this is not a full probe.",
        "Unsupported or not implemented: Say this is not answer-quality validation.",
    ]
)


def present_sections(text: str) -> tuple[dict[str, Any], list[str], list[str]]:
    sections = answer_adapter.parse_structured_answer(text)
    present = [
        key
        for key in answer_adapter.ANSWER_SECTION_KEYS.values()
        if isinstance(sections.get(key), str) and sections.get(key)
    ]
    missing = [
        key
        for key in answer_adapter.ANSWER_SECTION_KEYS.values()
        if key not in present
    ]
    return sections, present, missing


def endpoint_doctor(endpoint: str, max_tokens: int, timeout_seconds: int) -> dict[str, Any]:
    if not answer_adapter.is_local_json_endpoint(endpoint):
        return {
            "schema": REPORT_SCHEMA,
            "endpoint": endpoint,
            "ok": False,
            "local_endpoint": False,
            "reachable": False,
            "runtime_kind": "rejected_non_local_endpoint",
            "parsed": False,
            "sections_present": [],
            "sections_missing": list(answer_adapter.ANSWER_SECTION_KEYS.values()),
            "error": "Endpoint must be localhost, 127.0.0.1, or ::1.",
            "raw_output": "",
        }
    try:
        text = answer_adapter.run_json_http(endpoint, DOCTOR_PROMPT, max_tokens, timeout_seconds)
    except answer_adapter.LocalRunnerError as exc:
        return {
            "schema": REPORT_SCHEMA,
            "endpoint": endpoint,
            "ok": False,
            "local_endpoint": True,
            "reachable": False,
            "runtime_kind": "unreachable_or_error",
            "parsed": False,
            "sections_present": [],
            "sections_missing": list(answer_adapter.ANSWER_SECTION_KEYS.values()),
            "error": str(exc),
            "raw_output": "",
        }
    sections, present, missing = present_sections(text)
    stub_detected = any(marker in text for marker in phase2_probe_report_check.STUB_MARKERS)
    runtime_kind = "stub" if stub_detected else "real_candidate_or_unknown"
    return {
        "schema": REPORT_SCHEMA,
        "endpoint": endpoint,
        "ok": True,
        "local_endpoint": True,
        "reachable": True,
        "runtime_kind": runtime_kind,
        "stub_runtime_detected": stub_detected,
        "parsed": bool(sections.get("parsed")),
        "sections_present": present,
        "sections_missing": missing,
        "error": "",
        "raw_output": text,
        "note": "Run phase2_probe_suite.py for validation; doctor only checks endpoint shape.",
    }


def cmd_check(args: argparse.Namespace) -> int:
    report = endpoint_doctor(args.endpoint, args.max_tokens, args.timeout_seconds)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print(f"Endpoint: {report['endpoint']}")
        print(f"OK: {report['ok']}")
        print(f"Reachable: {report['reachable']}")
        print(f"Runtime kind: {report['runtime_kind']}")
        print(f"Parsed sections: {report['parsed']}")
        if report["sections_missing"]:
            print(f"Sections missing: {', '.join(report['sections_missing'])}")
        if report["error"]:
            print(f"Error: {report['error']}")
    return 0 if report["ok"] else 1


def cmd_self_check(args: argparse.Namespace) -> int:
    rejected = endpoint_doctor("https://example.com/completion", 8, 1)
    checks = [
        rejected["schema"] == REPORT_SCHEMA,
        not rejected["ok"],
        not rejected["local_endpoint"],
        rejected["runtime_kind"] == "rejected_non_local_endpoint",
    ]
    if not all(checks):
        print(json.dumps(rejected, indent=2, sort_keys=True))
        print("Phase 2 endpoint doctor self-check failed", file=sys.stderr)
        return 1
    print("All Phase 2 endpoint doctor checks passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="probe a local JSON endpoint")
    check.add_argument("--endpoint", default="http://127.0.0.1:8765/completion")
    check.add_argument("--max-tokens", type=int, default=128)
    check.add_argument("--timeout-seconds", type=int, default=10)
    check.add_argument("--json", action="store_true")
    check.set_defaults(func=cmd_check)

    self_check = sub.add_parser("self-check", help="run deterministic doctor self-check")
    self_check.set_defaults(func=cmd_self_check)
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
