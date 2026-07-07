#!/usr/bin/env python3
"""Run all deterministic tofix42 checks.

This is a local harness. It does not start model servers, call external
endpoints, launch Studio, mutate graphs, or run training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "cyxwiz.tofix42.check_all.v1"
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]


@dataclass(frozen=True)
class CheckCommand:
    phase: str
    name: str
    argv: list[str]


def check_commands() -> list[CheckCommand]:
    return [
        CheckCommand(
            phase="1A",
            name="index_status",
            argv=["phase1a_retrieval.py", "status"],
        ),
        CheckCommand(
            phase="1A",
            name="retrieval_check",
            argv=["phase1a_retrieval.py", "check"],
        ),
        CheckCommand(
            phase="1B",
            name="answer_adapter_check",
            argv=["phase1b_answer.py", "check", "--packet", str(THIS_DIR / "phase1b_test_packet.json")],
        ),
        CheckCommand(
            phase="2",
            name="probe_case_listing",
            argv=["phase2_probe_suite.py", "--list-cases", "--json"],
        ),
        CheckCommand(
            phase="2",
            name="probe_report_checker",
            argv=["phase2_probe_report_check.py", "check"],
        ),
        CheckCommand(
            phase="2",
            name="endpoint_doctor_self_check",
            argv=["phase2_endpoint_doctor.py", "self-check"],
        ),
        CheckCommand(
            phase="2",
            name="local_endpoint_scan_self_check",
            argv=["phase2_local_endpoint_scan.py", "self-check"],
        ),
        CheckCommand(
            phase="2",
            name="openai_compat_proxy_self_check",
            argv=["phase2_openai_compat_proxy.py", "self-check"],
        ),
        CheckCommand(
            phase="2",
            name="real_model_check_self_check",
            argv=["phase2_real_model_check.py", "self-check"],
        ),
        CheckCommand(
            phase="3",
            name="debug_context_check",
            argv=["phase3_debug_context.py", "check"],
        ),
        CheckCommand(
            phase="4",
            name="training_context_check",
            argv=["phase4_training_context.py", "check"],
        ),
        CheckCommand(
            phase="5",
            name="graph_context_check",
            argv=["phase5_graph_context.py", "check"],
        ),
        CheckCommand(
            phase="6",
            name="eval_capture_check",
            argv=["phase6_eval_capture.py", "check"],
        ),
        CheckCommand(
            phase="7",
            name="finetune_decision_check",
            argv=["phase7_finetune_decision.py", "check"],
        ),
    ]


def tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def run_one(command: CheckCommand, timeout_seconds: int) -> dict[str, Any]:
    argv = [sys.executable, str(THIS_DIR / command.argv[0]), *command.argv[1:]]
    completed = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_seconds if timeout_seconds > 0 else None,
    )
    return {
        "phase": command.phase,
        "name": command.name,
        "argv": command.argv,
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout": tail(completed.stdout),
        "stderr": tail(completed.stderr),
    }


def run_all(timeout_seconds: int, stop_on_fail: bool) -> dict[str, Any]:
    results = []
    for command in check_commands():
        try:
            result = run_one(command, timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            result = {
                "phase": command.phase,
                "name": command.name,
                "argv": command.argv,
                "returncode": None,
                "ok": False,
                "stdout": tail(exc.stdout or ""),
                "stderr": tail(exc.stderr or ""),
                "error": f"timed out after {timeout_seconds} seconds",
            }
        results.append(result)
        if stop_on_fail and not result["ok"]:
            break
    return {
        "schema": REPORT_SCHEMA,
        "ok": all(item["ok"] for item in results),
        "check_count": len(results),
        "passed": sum(1 for item in results if item["ok"]),
        "failed": sum(1 for item in results if not item["ok"]),
        "model_runtime": "not_used",
        "results": results,
    }


def cmd_run(args: argparse.Namespace) -> int:
    report = run_all(args.timeout_seconds, args.stop_on_fail)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print(f"OK: {report['ok']}")
        print(f"Checks: {report['check_count']} passed={report['passed']} failed={report['failed']}")
        for result in report["results"]:
            status = "PASS" if result["ok"] else "FAIL"
            print(f"{status}: phase {result['phase']} {result['name']}")
            if result.get("error"):
                print(f"  error: {result['error']}")
            if result["stderr"]:
                print(f"  stderr: {result['stderr'].strip()}")
    return 0 if report["ok"] else 1


def cmd_check(args: argparse.Namespace) -> int:
    report = run_all(args.timeout_seconds, stop_on_fail=False)
    checks = [
        report["schema"] == REPORT_SCHEMA,
        report["ok"],
        report["check_count"] == len(check_commands()),
        any(item["name"] == "retrieval_check" for item in report["results"]),
        any(item["name"] == "finetune_decision_check" for item in report["results"]),
    ]
    if not all(checks):
        print(json.dumps(report, indent=2, sort_keys=True))
        print("tofix42 check-all failed", file=sys.stderr)
        return 1
    print("All tofix42 deterministic checks passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run all deterministic checks")
    run.add_argument("--timeout-seconds", type=int, default=120)
    run.add_argument("--stop-on-fail", action="store_true")
    run.add_argument("--output", type=Path)
    run.add_argument("--json", action="store_true")
    run.set_defaults(func=cmd_run)

    check = sub.add_parser("check", help="run check-all self gate")
    check.add_argument("--timeout-seconds", type=int, default=120)
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
