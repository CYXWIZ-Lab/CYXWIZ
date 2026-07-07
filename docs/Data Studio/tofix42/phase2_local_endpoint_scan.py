#!/usr/bin/env python3
"""Scan common local Phase 2 model endpoint locations.

This is a read-only local helper. It does not start servers, download models,
train, or call non-local endpoints. It uses the endpoint doctor for tofix42
`/completion` endpoints and a TCP reachability check for common
OpenAI-compatible upstream ports.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import urllib.parse
from dataclasses import dataclass
from typing import Any

import phase1b_answer as answer_adapter
import phase2_endpoint_doctor


REPORT_SCHEMA = "cyxwiz.tofix42.phase2.local_endpoint_scan.v1"


@dataclass(frozen=True)
class ScanTarget:
    name: str
    endpoint: str
    kind: str
    note: str


DEFAULT_TARGETS = [
    ScanTarget(
        name="tofix42_direct_or_stub",
        endpoint="http://127.0.0.1:8765/completion",
        kind="completion",
        note="Direct tofix42 /completion endpoint or phase2_stub_runtime.py.",
    ),
    ScanTarget(
        name="tofix42_openai_proxy",
        endpoint="http://127.0.0.1:8766/completion",
        kind="completion",
        note="phase2_openai_compat_proxy.py /completion endpoint.",
    ),
    ScanTarget(
        name="openai_compat_upstream",
        endpoint="http://127.0.0.1:1234/v1/chat/completions",
        kind="tcp",
        note="Common local OpenAI-compatible chat endpoint such as LM Studio.",
    ),
]


def tcp_reachable(endpoint: str, timeout_seconds: int) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(endpoint)
    hostname = parsed.hostname or ""
    port = parsed.port
    if not answer_adapter.is_local_json_endpoint(endpoint):
        return {
            "reachable": False,
            "local_endpoint": False,
            "error": "Endpoint must be localhost, 127.0.0.1, or ::1.",
        }
    if port is None:
        return {
            "reachable": False,
            "local_endpoint": True,
            "error": "Endpoint has no explicit port.",
        }
    try:
        with socket.create_connection((hostname, port), timeout=timeout_seconds):
            return {"reachable": True, "local_endpoint": True, "error": ""}
    except OSError as exc:
        return {"reachable": False, "local_endpoint": True, "error": str(exc)}


def scan_target(target: ScanTarget, max_tokens: int, timeout_seconds: int) -> dict[str, Any]:
    if target.kind == "completion":
        doctor = phase2_endpoint_doctor.endpoint_doctor(
            target.endpoint,
            max_tokens,
            timeout_seconds,
        )
        return {
            "name": target.name,
            "endpoint": target.endpoint,
            "kind": target.kind,
            "note": target.note,
            "reachable": doctor.get("reachable", False),
            "runtime_kind": doctor.get("runtime_kind", ""),
            "parsed": doctor.get("parsed", False),
            "error": doctor.get("error", ""),
        }
    tcp = tcp_reachable(target.endpoint, timeout_seconds)
    return {
        "name": target.name,
        "endpoint": target.endpoint,
        "kind": target.kind,
        "note": target.note,
        "reachable": tcp["reachable"],
        "runtime_kind": "tcp_open" if tcp["reachable"] else "tcp_closed",
        "parsed": False,
        "error": tcp["error"],
    }


def next_actions(results: list[dict[str, Any]]) -> list[str]:
    by_name = {item["name"]: item for item in results}
    direct = by_name.get("tofix42_direct_or_stub", {})
    proxy = by_name.get("tofix42_openai_proxy", {})
    upstream = by_name.get("openai_compat_upstream", {})

    actions: list[str] = []
    if direct.get("runtime_kind") == "real_candidate_or_unknown":
        actions.append(
            "Run phase2_real_model_check.py against http://127.0.0.1:8765/completion."
        )
    if proxy.get("runtime_kind") == "real_candidate_or_unknown":
        actions.append(
            "Run phase2_real_model_check.py against http://127.0.0.1:8766/completion."
        )
    if direct.get("runtime_kind") == "stub":
        actions.append("Do not use http://127.0.0.1:8765/completion as real-model evidence; it is the stub.")
    if not upstream.get("reachable"):
        actions.append(
            "Start a real local OpenAI-compatible model server at http://127.0.0.1:1234/v1/chat/completions, or change the proxy --upstream."
        )
    if upstream.get("reachable") and not proxy.get("reachable"):
        actions.append(
            "Start phase2_openai_compat_proxy.py on port 8766 for the reachable OpenAI-compatible upstream."
        )
    if not actions:
        actions.append("Start a real local model server, then rerun this scan.")
    return actions


def scan(max_tokens: int, timeout_seconds: int) -> dict[str, Any]:
    results = [
        scan_target(target, max_tokens, timeout_seconds)
        for target in DEFAULT_TARGETS
    ]
    return {
        "schema": REPORT_SCHEMA,
        "ok": True,
        "results": results,
        "next_actions": next_actions(results),
    }


def cmd_scan(args: argparse.Namespace) -> int:
    report = scan(args.max_tokens, args.timeout_seconds)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print(f"OK: {report['ok']}")
        for item in report["results"]:
            print(
                f"{item['name']}: reachable={item['reachable']} "
                f"runtime_kind={item['runtime_kind']} endpoint={item['endpoint']}"
            )
            if item["error"]:
                print(f"  error: {item['error']}")
        print("Next actions:")
        for action in report["next_actions"]:
            print(f"- {action}")
    return 0


def cmd_self_check(args: argparse.Namespace) -> int:
    fake_results = [
        {
            "name": "tofix42_direct_or_stub",
            "endpoint": "http://127.0.0.1:8765/completion",
            "reachable": True,
            "runtime_kind": "stub",
        },
        {
            "name": "tofix42_openai_proxy",
            "endpoint": "http://127.0.0.1:8766/completion",
            "reachable": False,
            "runtime_kind": "unreachable_or_error",
        },
        {
            "name": "openai_compat_upstream",
            "endpoint": "http://127.0.0.1:1234/v1/chat/completions",
            "reachable": True,
            "runtime_kind": "tcp_open",
        },
    ]
    actions = next_actions(fake_results)
    checks = [
        answer_adapter.is_local_json_endpoint("http://127.0.0.1:1234/v1/chat/completions"),
        not answer_adapter.is_local_json_endpoint("https://example.com/v1/chat/completions"),
        any("not use http://127.0.0.1:8765" in action for action in actions),
        any("phase2_openai_compat_proxy.py" in action for action in actions),
    ]
    if not all(checks):
        print(json.dumps({"schema": REPORT_SCHEMA, "ok": False, "actions": actions}, indent=2))
        print("Phase 2 local endpoint scan self-check failed", file=sys.stderr)
        return 1
    print("All Phase 2 local endpoint scan checks passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    scan_parser = sub.add_parser("scan", help="scan common local endpoint locations")
    scan_parser.add_argument("--max-tokens", type=int, default=64)
    scan_parser.add_argument("--timeout-seconds", type=int, default=2)
    scan_parser.add_argument("--json", action="store_true")
    scan_parser.set_defaults(func=cmd_scan)

    self_check = sub.add_parser("self-check", help="run deterministic scan checks")
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
