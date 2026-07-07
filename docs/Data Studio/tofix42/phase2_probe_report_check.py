#!/usr/bin/env python3
"""Validate saved Phase 2 probe-suite reports.

This is offline. It reads a JSON report produced by phase2_probe_suite.py and
does not call a model endpoint, start a server, download models, or mutate
files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import phase2_probe_suite


REPORT_SCHEMA = "cyxwiz.tofix42.phase2.probe_report_check.v1"
PROBE_SUITE_SCHEMA = "cyxwiz.tofix42.phase2.probe_suite.v1"
STUB_MARKERS = (
    "Stub runtime received",
    "Real answer quality is not tested by this stub",
    "No real model inference was performed",
    "A real local model answer has not been validated by this stub",
)


def load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected probe report JSON object")
    return value


def expected_case_names() -> set[str]:
    return {case.name for case in phase2_probe_suite.PROBE_CASES}


def result_failures(result: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not result.get("evidence_ok", False):
        failures.append("evidence_not_expected_path")
    if not result.get("runtime_ok", False):
        failures.append("runtime_failed")
    if not result.get("parsed", False):
        failures.append("structured_sections_not_parsed")
    if result.get("sections_missing"):
        failures.append("sections_missing")
    if not result.get("expected_path_in_output", False):
        failures.append("expected_path_missing_from_output")
    if result.get("required_terms_missing"):
        failures.append("required_terms_missing")
    if result.get("forbidden_terms_present"):
        failures.append("forbidden_terms_present")
    if not result.get("ok", False):
        failures.append("case_not_ok")
    return sorted(set(failures))


def result_text(result: dict[str, Any]) -> str:
    parts = [
        str(result.get("answer", "")),
        str(result.get("raw_model_output", "")),
        str(result.get("error", "")),
    ]
    sections = result.get("model_output_sections", {})
    if isinstance(sections, dict):
        parts.extend(str(value) for value in sections.values())
    return "\n".join(parts)


def report_uses_stub_runtime(results: list[dict[str, Any]]) -> bool:
    return any(
        any(marker in result_text(result) for marker in STUB_MARKERS)
        for result in results
    )


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    results = [item for item in report.get("results", []) if isinstance(item, dict)]
    seen_cases = {str(item.get("name", "")) for item in results}
    missing_cases = sorted(expected_case_names() - seen_cases)
    unknown_cases = sorted(seen_cases - expected_case_names())
    stub_runtime_detected = report_uses_stub_runtime(results)
    case_reports = []
    for result in results:
        failures = result_failures(result)
        case_reports.append(
            {
                "name": result.get("name", ""),
                "ok": not failures,
                "failures": failures,
                "expected_path": result.get("expected_path", ""),
                "top_path": result.get("top_path", ""),
                "error": result.get("error", ""),
            }
        )
    failures = [
        "schema_mismatch" if report.get("schema") != PROBE_SUITE_SCHEMA else "",
        "report_not_ok" if not report.get("ok", False) else "",
        "missing_cases" if missing_cases else "",
        "unknown_cases" if unknown_cases else "",
        "stub_runtime_detected" if stub_runtime_detected else "",
    ]
    failures.extend(
        f"case_failed:{item['name']}"
        for item in case_reports
        if not item["ok"]
    )
    failures = [item for item in failures if item]
    return {
        "schema": REPORT_SCHEMA,
        "ok": not failures,
        "probe_schema": report.get("schema", ""),
        "endpoint": report.get("endpoint", ""),
        "case_count": len(results),
        "expected_case_count": len(expected_case_names()),
        "missing_cases": missing_cases,
        "unknown_cases": unknown_cases,
        "stub_runtime_detected": stub_runtime_detected,
        "failures": failures,
        "cases": case_reports,
    }


def synthetic_passing_report() -> dict[str, Any]:
    results = []
    for case in phase2_probe_suite.PROBE_CASES:
        results.append(
            {
                "name": case.name,
                "query": case.query,
                "expected_path": case.expected_path,
                "top_path": case.expected_path,
                "evidence_ok": True,
                "runtime_ok": True,
                "parsed": True,
                "sections_present": ["answer", "evidence", "unknowns", "unsupported_or_not_implemented"],
                "sections_missing": [],
                "required_terms_missing": [],
                "forbidden_terms_present": [],
                "expected_path_in_output": True,
                "answer": "Real model answer placeholder.",
                "model_output_sections": {
                    "answer": "Real model answer placeholder.",
                    "evidence": f"The answer cites {case.expected_path}.",
                    "unknowns": "None.",
                    "unsupported_or_not_implemented": "None.",
                    "parsed": True,
                },
                "error": "",
                "ok": True,
            }
        )
    return {
        "schema": PROBE_SUITE_SCHEMA,
        "endpoint": "http://127.0.0.1:8765/completion",
        "case_count": len(results),
        "selected_cases": sorted(expected_case_names()),
        "ok": True,
        "results": results,
    }


def cmd_validate(args: argparse.Namespace) -> int:
    report = load_report(args.report)
    validation = validate_report(report)
    if args.json:
        print(json.dumps(validation, indent=2, ensure_ascii=True))
    else:
        print(f"OK: {validation['ok']}")
        print(f"Cases: {validation['case_count']} expected={validation['expected_case_count']}")
        if validation["failures"]:
            print("Failures:")
            for item in validation["failures"]:
                print(f"- {item}")
        for case in validation["cases"]:
            status = "PASS" if case["ok"] else "FAIL"
            print(f"{status}: {case['name']} -> {case['top_path']}")
            if case["failures"]:
                print(f"  failures: {', '.join(case['failures'])}")
            if case["error"]:
                print(f"  error: {case['error']}")
    return 0 if validation["ok"] else 1


def cmd_check(args: argparse.Namespace) -> int:
    validation = validate_report(synthetic_passing_report())
    checks = [
        validation["schema"] == REPORT_SCHEMA,
        validation["ok"],
        validation["case_count"] == len(phase2_probe_suite.PROBE_CASES),
        not validation["missing_cases"],
        not validation["unknown_cases"],
        not validation["stub_runtime_detected"],
    ]
    if not all(checks):
        print(json.dumps(validation, indent=2, sort_keys=True))
        print("Phase 2 probe report checker failed", file=sys.stderr)
        return 1
    print("All Phase 2 probe report checker checks passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate a saved Phase 2 probe-suite report")
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(func=cmd_validate)

    check = sub.add_parser("check", help="run deterministic checker self-test")
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
