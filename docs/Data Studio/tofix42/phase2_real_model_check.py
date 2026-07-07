#!/usr/bin/env python3
"""Run the Phase 2 real-model validation sequence against a localhost endpoint.

This command orchestrates existing checks: endpoint doctor, probe suite, offline
probe-report validation, optional bad-answer capture, and Phase 7 decision. It
does not start a model server, download a model, call non-local endpoints, or
train anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import phase1a_retrieval as retrieval
import phase2_endpoint_doctor
import phase2_probe_report_check
import phase2_probe_suite
import phase6_eval_capture
import phase7_finetune_decision


REPORT_SCHEMA = "cyxwiz.tofix42.phase2.real_model_check.v1"
DEFAULT_OUTPUT = Path(os.environ.get("TEMP", ".")) / "tofix42_phase2_probe_suite.json"


def build_probe_report(args: argparse.Namespace) -> dict[str, Any]:
    cases, unknown = phase2_probe_suite.selected_cases(args.case)
    if unknown:
        return {
            "schema": "cyxwiz.tofix42.phase2.probe_suite.v1",
            "endpoint": args.endpoint,
            "case_count": 0,
            "selected_cases": [],
            "ok": False,
            "output": str(args.output),
            "output_error": "",
            "error": "Unknown probe case.",
            "unknown_cases": unknown,
            "available_cases": phase2_probe_suite.case_names(),
            "results": [],
        }
    index = retrieval.load_index(args.index)
    results = [
        phase2_probe_suite.run_case(
            case,
            index,
            args.endpoint,
            args.max_tokens,
            args.timeout_seconds,
            args.max_chars_per_evidence,
            args.include_raw_output,
            args.include_prompt,
        )
        for case in cases
    ]
    return {
        "schema": "cyxwiz.tofix42.phase2.probe_suite.v1",
        "endpoint": args.endpoint,
        "case_count": len(results),
        "selected_cases": [case.name for case in cases],
        "ok": all(item["ok"] for item in results),
        "output": str(args.output),
        "output_error": "",
        "results": results,
    }


def write_probe_report(report: dict[str, Any], output: Path) -> str:
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return ""
    except OSError as exc:
        report["ok"] = False
        report["output_error"] = str(exc)
        return str(exc)


def failed_case_count(validation: dict[str, Any]) -> int:
    return sum(
        1
        for case in validation.get("cases", [])
        if isinstance(case, dict) and not case.get("ok", False)
    )


def next_actions(
    *,
    endpoint: str,
    output: Path,
    capture_failures: bool,
    doctor: dict[str, Any],
    validation: dict[str, Any],
    capture: dict[str, Any],
    decision: dict[str, Any],
    output_error: str,
) -> list[str]:
    actions: list[str] = []
    if output_error:
        actions.append(f"Fix the probe report output path or permissions: {output}")
    if not doctor.get("ok", False):
        actions.append(f"Start a localhost JSON model server or proxy at {endpoint}, then rerun this command.")
        return actions
    if doctor.get("runtime_kind") == "stub":
        actions.append("Point --endpoint at a real local model endpoint or the OpenAI-compatible proxy, not phase2_stub_runtime.py.")
    if validation and failed_case_count(validation) > 0:
        if not capture_failures:
            actions.append("Rerun with --capture-failures to append failed probe cases to the Phase 6 bad-answer log.")
        elif capture.get("captured_count", 0) > 0:
            actions.append("Run phase6_eval_capture.py review-queue, then review captured bad-answer corrections.")
        else:
            actions.append("Review the saved probe report; failed cases were duplicates or were not captured.")
    blockers = set(decision.get("blocking_criteria", []))
    if "reviewed_corrections_available" in blockers:
        actions.append("Use phase6_eval_capture.py review-queue and review-bad-answer until the reviewed-correction threshold is met.")
    if "real_local_model_probe_passed" in blockers and doctor.get("runtime_kind") != "stub":
        actions.append("Fix model output to satisfy all required sections, citations, and forbidden-term checks, then rerun this command.")
    if not actions and validation.get("ok", False):
        actions.append("Use the saved probe report with phase7_finetune_decision.py decide.")
    return actions


def readiness_summary(
    *,
    validation: dict[str, Any],
    decision: dict[str, Any],
    output_error: str,
) -> dict[str, Any]:
    real_model_probe_accepted = bool(validation.get("ok", False)) and not output_error
    fine_tuning_ready = bool(decision.get("approved", False))
    if fine_tuning_ready:
        meaning = "Real-model probe passed and Phase 7 approved the fine-tuning experiment gate."
    elif real_model_probe_accepted:
        meaning = "Real-model probe passed; Phase 7 still has remaining blockers before fine-tuning."
    else:
        meaning = "Real-model probe evidence has not been accepted yet."
    return {
        "real_model_probe_accepted": real_model_probe_accepted,
        "fine_tuning_ready": fine_tuning_ready,
        "meaning": meaning,
    }


def run_real_model_check(args: argparse.Namespace) -> dict[str, Any]:
    doctor = phase2_endpoint_doctor.endpoint_doctor(
        args.endpoint,
        args.doctor_max_tokens,
        args.doctor_timeout_seconds,
    )
    if not doctor.get("ok", False):
        return {
            "schema": REPORT_SCHEMA,
            "ok": False,
            "endpoint": args.endpoint,
            "probe_report_path": str(args.output),
            "doctor": doctor,
            "probe_report": {},
            "validation": {},
            "capture": {},
            "phase7_decision": {},
            "readiness": readiness_summary(
                validation={},
                decision={},
                output_error="endpoint_doctor_failed",
            ),
            "next_actions": next_actions(
                endpoint=args.endpoint,
                output=args.output,
                capture_failures=args.capture_failures,
                doctor=doctor,
                validation={},
                capture={},
                decision={},
                output_error="",
            ),
            "error": "endpoint_doctor_failed",
        }

    probe_report = build_probe_report(args)
    output_error = write_probe_report(probe_report, args.output)
    validation = phase2_probe_report_check.validate_report(probe_report)
    capture: dict[str, Any] = {}
    if args.capture_failures and not validation.get("ok", False) and not output_error:
        capture = phase6_eval_capture.capture_probe_failures(
            args.output,
            args.bad_answers,
            dedupe=not args.no_dedupe,
        )

    decision: dict[str, Any] = {}
    if not output_error:
        decision = phase7_finetune_decision.make_decision(
            index_path=args.index,
            bad_answers_path=args.bad_answers,
            real_model_probe_report=args.output,
            min_reviewed_examples=args.min_reviewed_examples,
            top=args.top,
        )

    decision_summary = {
        "recommendation": decision.get("recommendation", ""),
        "approved": decision.get("approved", False),
        "blocking_criteria": decision.get("blocking_criteria", []),
        "next_actions": decision.get("next_actions", []),
    }
    doctor_summary = {
        "ok": doctor.get("ok", False),
        "reachable": doctor.get("reachable", False),
        "runtime_kind": doctor.get("runtime_kind", ""),
        "parsed": doctor.get("parsed", False),
        "error": doctor.get("error", ""),
    }
    return {
        "schema": REPORT_SCHEMA,
        "ok": bool(validation.get("ok", False)) and not output_error,
        "endpoint": args.endpoint,
        "probe_report_path": str(args.output),
        "doctor": doctor_summary,
        "probe_report": {
            "ok": probe_report.get("ok", False),
            "case_count": probe_report.get("case_count", 0),
            "selected_cases": probe_report.get("selected_cases", []),
            "output_error": probe_report.get("output_error", ""),
        },
        "validation": validation,
        "capture": capture,
        "phase7_decision": decision_summary,
        "readiness": readiness_summary(
            validation=validation,
            decision=decision_summary,
            output_error=output_error,
        ),
        "next_actions": next_actions(
            endpoint=args.endpoint,
            output=args.output,
            capture_failures=args.capture_failures,
            doctor=doctor_summary,
            validation=validation,
            capture=capture,
            decision=decision_summary,
            output_error=output_error,
        ),
        "error": output_error,
    }


def print_text_report(report: dict[str, Any]) -> None:
    print(f"OK: {report['ok']}")
    print(f"Endpoint: {report['endpoint']}")
    print(f"Probe report: {report['probe_report_path']}")
    doctor = report["doctor"]
    print(
        "Doctor: "
        f"ok={doctor.get('ok')} "
        f"reachable={doctor.get('reachable')} "
        f"runtime_kind={doctor.get('runtime_kind')} "
        f"parsed={doctor.get('parsed')}"
    )
    if doctor.get("error"):
        print(f"Doctor error: {doctor['error']}")
    probe = report["probe_report"]
    if probe:
        print(f"Probe suite: ok={probe.get('ok')} cases={probe.get('case_count')}")
        if probe.get("output_error"):
            print(f"Output error: {probe['output_error']}")
    validation = report["validation"]
    if validation:
        print(f"Validation: ok={validation.get('ok')} failures={len(validation.get('failures', []))}")
        for failure in validation.get("failures", []):
            print(f"- {failure}")
        failed_cases = [
            case
            for case in validation.get("cases", [])
            if isinstance(case, dict) and not case.get("ok", False)
        ]
        if failed_cases:
            print("Failed cases:")
            for case in failed_cases:
                print(f"- {case.get('name', '')}")
                print(f"  failures: {', '.join(case.get('failures', []))}")
                print(f"  expected_path: {case.get('expected_path', '')}")
                print(f"  top_path: {case.get('top_path', '')}")
                if case.get("error"):
                    print(f"  error: {case.get('error')}")
    capture = report["capture"]
    if capture:
        print(f"Captured failures: {capture.get('captured_count', 0)}")
    decision = report["phase7_decision"]
    if decision:
        print(f"Phase 7: {decision.get('recommendation')} approved={decision.get('approved')}")
        if decision.get("blocking_criteria"):
            print("Phase 7 blockers:")
            for item in decision["blocking_criteria"]:
                print(f"- {item}")
    readiness = report.get("readiness", {})
    if readiness:
        print(
            "Readiness: "
            f"real_model_probe_accepted={readiness.get('real_model_probe_accepted')} "
            f"fine_tuning_ready={readiness.get('fine_tuning_ready')}"
        )
        print(f"Meaning: {readiness.get('meaning')}")
    if report.get("next_actions"):
        print("Next actions:")
        for item in report["next_actions"]:
            print(f"- {item}")


def cmd_run(args: argparse.Namespace) -> int:
    report = run_real_model_check(args)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print_text_report(report)
    return 0 if report["ok"] else 1


def cmd_self_check(args: argparse.Namespace) -> int:
    synthetic = phase2_probe_report_check.validate_report(
        phase2_probe_report_check.synthetic_passing_report()
    )
    probe_passed_decision = {
        "approved": False,
        "blocking_criteria": ["reviewed_corrections_available"],
    }
    probe_passed_readiness = readiness_summary(
        validation=synthetic,
        decision=probe_passed_decision,
        output_error="",
    )
    probe_passed_actions = next_actions(
        endpoint="http://127.0.0.1:8766/completion",
        output=DEFAULT_OUTPUT,
        capture_failures=True,
        doctor={"ok": True, "runtime_kind": "real_candidate_or_unknown"},
        validation=synthetic,
        capture={},
        decision=probe_passed_decision,
        output_error="",
    )
    stub_actions = next_actions(
        endpoint="http://127.0.0.1:8765/completion",
        output=DEFAULT_OUTPUT,
        capture_failures=False,
        doctor={"ok": True, "runtime_kind": "stub"},
        validation={"ok": False, "cases": []},
        capture={},
        decision={"blocking_criteria": ["real_local_model_probe_passed"]},
        output_error="",
    )
    checks = [
        synthetic["schema"] == phase2_probe_report_check.REPORT_SCHEMA,
        synthetic["ok"],
        synthetic["case_count"] == len(phase2_probe_suite.PROBE_CASES),
        probe_passed_readiness["real_model_probe_accepted"],
        not probe_passed_readiness["fine_tuning_ready"],
        "reviewed-correction threshold" in " ".join(probe_passed_actions),
        any("not phase2_stub_runtime.py" in item for item in stub_actions),
    ]
    if not all(checks):
        print(json.dumps(synthetic, indent=2, sort_keys=True))
        print("Phase 2 real-model check self-check failed", file=sys.stderr)
        return 1
    print("All Phase 2 real-model check checks passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run real-model validation sequence")
    run.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    run.add_argument("--endpoint", default="http://127.0.0.1:8765/completion")
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--max-tokens", type=int, default=384)
    run.add_argument("--timeout-seconds", type=int, default=30)
    run.add_argument("--doctor-max-tokens", type=int, default=128)
    run.add_argument("--doctor-timeout-seconds", type=int, default=10)
    run.add_argument("--max-chars-per-evidence", type=int, default=2400)
    run.add_argument("--case", action="append", default=[])
    run.add_argument("--include-raw-output", action="store_true")
    run.add_argument("--include-prompt", action="store_true")
    run.add_argument("--capture-failures", action="store_true")
    run.add_argument("--bad-answers", type=Path, default=phase6_eval_capture.DEFAULT_BAD_ANSWERS)
    run.add_argument("--no-dedupe", action="store_true")
    run.add_argument("--min-reviewed-examples", type=int, default=20)
    run.add_argument("--top", type=int, default=5)
    run.add_argument("--json", action="store_true")
    run.set_defaults(func=cmd_run)

    self_check = sub.add_parser("self-check", help="run deterministic self-check")
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
