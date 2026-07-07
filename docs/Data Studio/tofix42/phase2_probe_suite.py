#!/usr/bin/env python3
"""Run Phase 2 local JSON runtime probe cases.

This uses the existing Phase 1A index and Phase 1B JSON runtime adapter. It
does not build the index, start a server, download models, or call non-local
endpoints.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import phase1a_retrieval as retrieval
import phase1b_answer as answer_adapter


@dataclass
class ProbeCase:
    name: str
    query: str
    expected_path: str
    required_terms: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)


PROBE_CASES = [
    ProbeCase(
        name="debug_trace_record_definition",
        query="What source file defines DebugTraceRecord",
        expected_path="cyxwiz-engine/src/core/debug_trace_record.h",
    ),
    ProbeCase(
        name="training_trace_terminal_reason",
        query="TrainingTraceEvent terminal_reason field",
        expected_path="cyxwiz-engine/src/core/training_trace_collector.h",
    ),
    ProbeCase(
        name="tfidf_sentiment_graph",
        query="TFIDFVectorizer sentiment graph",
        expected_path="examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph",
    ),
    ProbeCase(
        name="dataloader_pin_memory_truth",
        query="DataLoader pin_memory unsupported current batchers compatibility",
        expected_path="cyxwiz-engine/src/core/graph_compiler.cpp",
        required_terms=["pin_memory", "compatibility", "ignored"],
        forbidden_terms=["implemented cuda pinned"],
    ),
    ProbeCase(
        name="tfidf_min_df_validation",
        query="TFIDFVectorizer min_df must be >= 1 Configure",
        expected_path="cyxwiz-engine/src/core/node_executors/tfidf_vectorizer_operator.cpp",
    ),
]


def case_names() -> list[str]:
    return [case.name for case in PROBE_CASES]


def selected_cases(names: list[str]) -> tuple[list[ProbeCase], list[str]]:
    if not names:
        return list(PROBE_CASES), []
    by_name = {case.name: case for case in PROBE_CASES}
    unknown = [name for name in names if name not in by_name]
    return [by_name[name] for name in names if name in by_name], unknown


def sections_report(text: str) -> tuple[bool, list[str], list[str]]:
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
    return bool(sections.get("parsed")), present, missing


def run_case(
    case: ProbeCase,
    index: dict,
    endpoint: str,
    max_tokens: int,
    timeout_seconds: int,
    max_chars_per_evidence: int,
    include_raw_output: bool,
    include_prompt: bool,
) -> dict:
    packet = retrieval.make_answer_packet(index, case.query, top=1)
    evidence = packet.get("evidence", [])
    citation = {}
    if evidence:
        citation = evidence[0].get("citation", {})
    top_path = citation.get("path", "")

    report = {
        "name": case.name,
        "query": case.query,
        "expected_path": case.expected_path,
        "top_path": top_path,
        "top_citation": {
            "path": top_path,
            "line_start": citation.get("line_start"),
            "line_end": citation.get("line_end"),
            "title": citation.get("title"),
            "source_type": citation.get("source_type"),
        },
        "evidence_ok": top_path == case.expected_path,
        "runtime_ok": False,
        "parsed": False,
        "sections_present": [],
        "sections_missing": list(answer_adapter.ANSWER_SECTION_KEYS.values()),
        "model_output_sections": {
            "answer": "",
            "evidence": "",
            "unknowns": "",
            "unsupported_or_not_implemented": "",
            "parsed": False,
        },
        "required_terms_missing": [],
        "forbidden_terms_present": [],
        "expected_path_in_output": False,
        "answer": "",
        "raw_model_output": "",
        "prompt": "",
        "error": "",
        "ok": False,
    }

    if not report["evidence_ok"]:
        report["error"] = "Top retrieved evidence did not match expected path."
        return report

    prompt = answer_adapter.build_prompt(packet, max_chars_per_evidence)
    if include_prompt:
        report["prompt"] = prompt
    try:
        text = answer_adapter.run_json_http(
            endpoint,
            prompt,
            max_tokens,
            timeout_seconds,
        )
    except answer_adapter.LocalRunnerError as exc:
        report["error"] = str(exc)
        return report

    parsed = answer_adapter.parse_structured_answer(text)
    parsed_flag, present, missing = sections_report(text)
    answer_text = parsed.get("answer", "") if parsed_flag else text
    lower_output = text.lower()
    expected_path_in_output = case.expected_path.lower() in lower_output
    required_missing = [
        term for term in case.required_terms if term.lower() not in lower_output
    ]
    forbidden_present = [
        term for term in case.forbidden_terms if term.lower() in lower_output
    ]

    report.update(
        {
            "runtime_ok": True,
            "parsed": parsed_flag,
            "sections_present": present,
            "sections_missing": missing,
            "model_output_sections": parsed,
            "required_terms_missing": required_missing,
            "forbidden_terms_present": forbidden_present,
            "expected_path_in_output": expected_path_in_output,
            "answer": answer_text,
            "raw_model_output": text if include_raw_output else "",
        }
    )
    report["ok"] = (
        report["evidence_ok"]
        and report["runtime_ok"]
        and report["parsed"]
        and report["expected_path_in_output"]
        and not report["sections_missing"]
        and not report["required_terms_missing"]
        and not report["forbidden_terms_present"]
    )
    return report


def cmd_run(args: argparse.Namespace) -> int:
    if args.list_cases:
        payload = {
            "schema": "cyxwiz.tofix42.phase2.probe_cases.v1",
            "cases": [
                {
                    "name": case.name,
                    "query": case.query,
                    "expected_path": case.expected_path,
                }
                for case in PROBE_CASES
            ],
        }
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=True))
        else:
            for case in payload["cases"]:
                print(f"{case['name']}: {case['expected_path']}")
        return 0

    cases, unknown = selected_cases(args.case)
    if unknown:
        print(
            json.dumps(
                {
                    "schema": "cyxwiz.tofix42.phase2.probe_suite.v1",
                    "ok": False,
                    "error": "Unknown probe case.",
                    "unknown_cases": unknown,
                    "available_cases": case_names(),
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 2

    if not answer_adapter.is_local_json_endpoint(args.endpoint):
        print(
            json.dumps(
                {
                    "schema": "cyxwiz.tofix42.phase2.probe_suite.v1",
                    "ok": False,
                    "error": "Endpoint must be localhost, 127.0.0.1, or ::1.",
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 2

    index = retrieval.load_index(args.index)
    results = [
        run_case(
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
    report = {
        "schema": "cyxwiz.tofix42.phase2.probe_suite.v1",
        "endpoint": args.endpoint,
        "case_count": len(results),
        "selected_cases": [case.name for case in cases],
        "ok": all(item["ok"] for item in results),
        "output": str(args.output) if args.output else "",
        "output_error": "",
        "results": results,
    }

    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            report["ok"] = False
            report["output_error"] = str(exc)

    report_json = json.dumps(report, indent=2, ensure_ascii=True)

    if args.json:
        print(report_json)
    else:
        print(f"Endpoint: {report['endpoint']}")
        print(f"OK: {report['ok']}")
        if args.output:
            print(f"Output: {args.output}")
        for item in results:
            status = "PASS" if item["ok"] else "FAIL"
            print(f"{status}: {item['name']} -> {item['top_path']}")
            if item["error"]:
                print(f"  error: {item['error']}")
            if item["sections_missing"]:
                print(f"  missing sections: {', '.join(item['sections_missing'])}")
            if not item["expected_path_in_output"]:
                print("  expected path missing from model output")
            if item["required_terms_missing"]:
                print(f"  missing terms: {', '.join(item['required_terms_missing'])}")
            if item["forbidden_terms_present"]:
                print(f"  forbidden terms: {', '.join(item['forbidden_terms_present'])}")

    return 0 if report["ok"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="tofix42 Phase 2 probe suite")
    parser.add_argument("--index", type=Path, default=retrieval.DEFAULT_INDEX)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8765/completion")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--max-chars-per-evidence", type=int, default=2400)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--include-raw-output", action="store_true")
    parser.add_argument("--include-prompt", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=cmd_run)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
