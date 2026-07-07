#!/usr/bin/env python3
"""Phase 1B minimal answer adapter for tofix42.

This consumes a Phase 1A answer packet and prepares a strict evidence-grounded
prompt. It can either print the prompt or call an explicitly configured local
runner or local JSON endpoint. It does not download models, start servers, edit
graphs, or mutate source files.
"""

from __future__ import annotations

import argparse
import http.server
import json
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


PACKET_SCHEMA = "cyxwiz.tofix42.phase1a.answer_packet.v1"
ANSWER_SCHEMA = "cyxwiz.tofix42.phase1b.answer.v1"
WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")
MAX_RUNNER_OUTPUT_CHARS = 4000
MODEL_OUTPUT_SENTINEL = "###CYXWIZ_TOFIX42_MODEL_OUTPUT###"
ANSWER_SECTION_KEYS = {
    "Answer": "answer",
    "Evidence": "evidence",
    "Unknowns": "unknowns",
    "Unsupported or not implemented": "unsupported_or_not_implemented",
}
ANSWER_SECTION_RE = re.compile(
    r"^(Answer|Evidence|Unknowns|Unsupported or not implemented):\s*(.*)$"
)


class LocalRunnerError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        command: list[str],
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out


def load_packet(path: Path | None) -> dict:
    if path is None or str(path) == "-":
        raw = sys.stdin.read()
    else:
        raw = read_text_any(path)
    raw = raw.lstrip("\ufeff")
    raw = raw.removeprefix(chr(239) + chr(187) + chr(191))
    packet = json.loads(raw)
    schema = packet.get("schema")
    if schema != PACKET_SCHEMA:
        raise ValueError(f"Unsupported packet schema: {schema!r}")
    return packet


def read_text_any(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def citation_label(citation: dict) -> str:
    return (
        f"{citation.get('path', '')}:"
        f"{citation.get('line_start', '')}-{citation.get('line_end', '')}"
    )


def build_prompt(packet: dict, max_chars_per_evidence: int) -> str:
    question = packet.get("question", "")
    evidence_blocks = []
    for item in packet.get("evidence", []):
        citation = item.get("citation", {})
        text = evidence_excerpt(
            item.get("text", ""),
            question,
            max_chars_per_evidence,
        )
        evidence_blocks.append(
            "\n".join(
                [
                    f"[E{item.get('rank', '?')}] {citation_label(citation)}",
                    f"title: {citation.get('title', '')}",
                    f"type: {citation.get('source_type', '')}",
                    "text:",
                    text,
                ]
            )
        )

    evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "(no evidence)"
    missing = packet.get("missing_evidence_notes", [])
    missing_text = "\n".join(f"- {note}" for note in missing) if missing else "- none"

    return f"""You are the local CyxWiz source-aware assistant.

Answer only from the cited evidence below.
Separate facts from inference.
If evidence is missing, say what is missing.
Do not claim unsupported CyxWiz behavior exists.
Do not suggest graph or source mutation unless explicitly approved.

Question:
{question}

Evidence:
{evidence_text}

Missing evidence notes:
{missing_text}

Return this structure:
Answer:
Evidence:
Unknowns:
Unsupported or not implemented:

Begin the answer after this exact marker. Do not repeat the marker:
{MODEL_OUTPUT_SENTINEL}
"""


def evidence_excerpt(text: str, question: str, limit: int) -> str:
    if len(text) <= limit:
        return text

    lower = text.lower()
    best = -1
    for term in sorted(query_terms(question), key=len, reverse=True):
        idx = lower.find(term.lower())
        if idx >= 0:
            best = idx
            break

    if best < 0:
        return text[:limit] + "\n[truncated]"

    start = max(0, best - limit // 3)
    end = min(len(text), start + limit)
    prefix = "[truncated]\n" if start > 0 else ""
    suffix = "\n[truncated]" if end < len(text) else ""
    return prefix + text[start:end] + suffix


def query_terms(question: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "by",
        "current",
        "does",
        "for",
        "from",
        "in",
        "is",
        "of",
        "the",
        "this",
        "to",
        "what",
        "where",
    }
    out = []
    seen = set()
    for token in WORD_RE.findall(question):
        term = token.lower()
        if len(term) <= 1 or term in stopwords or term in seen:
            continue
        out.append(term)
        seen.add(term)
    return out


def runner_command_for_report(cmd: list[str], prompt: str) -> list[str]:
    reported = list(cmd)
    try:
        prompt_index = reported.index("-p") + 1
    except ValueError:
        return reported
    if prompt_index < len(reported):
        reported[prompt_index] = f"<prompt omitted; chars={len(prompt)}>"
    return reported


def trim_runner_output(text: str, limit: int = MAX_RUNNER_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[truncated]"


def extract_model_output(stdout: str) -> str:
    text = stdout.strip()
    if MODEL_OUTPUT_SENTINEL in text:
        text = text.rsplit(MODEL_OUTPUT_SENTINEL, 1)[1].strip()
    return text


def parse_structured_answer(text: str) -> dict:
    sections = {key: "" for key in ANSWER_SECTION_KEYS.values()}
    current_key = ""
    saw_heading = False
    for raw_line in text.splitlines():
        match = ANSWER_SECTION_RE.match(raw_line.strip())
        if match:
            saw_heading = True
            current_key = ANSWER_SECTION_KEYS[match.group(1)]
            sections[current_key] = match.group(2).strip()
            continue
        if not current_key:
            continue
        line = raw_line.rstrip()
        if sections[current_key]:
            sections[current_key] += "\n" + line
        else:
            sections[current_key] = line.lstrip()

    for key, value in list(sections.items()):
        sections[key] = value.strip()
    sections["parsed"] = saw_heading
    return sections


def answer_from_model_output(packet: dict, text: str, mode: str = "llama-cli") -> dict:
    parsed = parse_structured_answer(text)
    answer_text = parsed["answer"] if parsed["parsed"] and parsed["answer"] else text
    return {
        "schema": ANSWER_SCHEMA,
        "mode": mode,
        "question": packet.get("question", ""),
        "answer": answer_text,
        "raw_model_output": text,
        "model_output_sections": parsed,
        "citations": [item.get("citation", {}) for item in packet.get("evidence", [])],
        "unknowns": packet.get("missing_evidence_notes", []),
        "unsupported_or_not_implemented": [],
    }


def runner_error_answer(packet: dict, exc: LocalRunnerError) -> dict:
    return {
        "schema": ANSWER_SCHEMA,
        "mode": "runtime_error",
        "question": packet.get("question", ""),
        "answer": "",
        "citations": [item.get("citation", {}) for item in packet.get("evidence", [])],
        "unknowns": packet.get("missing_evidence_notes", []),
        "unsupported_or_not_implemented": [
            "The configured local model runner did not produce a successful answer."
        ],
        "runner": {
            "command": exc.command,
            "returncode": exc.returncode,
            "timed_out": exc.timed_out,
            "error": str(exc),
            "stdout": trim_runner_output(exc.stdout),
            "stderr": trim_runner_output(exc.stderr),
        },
    }


def run_llama_cli(
    runner: str,
    model: Path,
    prompt: str,
    max_tokens: int,
    timeout_seconds: int,
) -> str:
    executable = shutil.which(runner) if not Path(runner).exists() else runner
    cmd = [
        executable or runner,
        "-m",
        str(model),
        "-p",
        prompt,
        "-n",
        str(max_tokens),
        "--no-conversation",
        "--no-display-prompt",
        "--simple-io",
        "--no-warmup",
        "--log-disable",
    ]
    reported_cmd = runner_command_for_report(cmd, prompt)
    if not executable:
        raise LocalRunnerError(f"Runner not found: {runner}", command=reported_cmd)
    if not model.exists():
        raise LocalRunnerError(f"Model not found: {model}", command=reported_cmd)

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            text=True,
            input="",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds if timeout_seconds > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        raise LocalRunnerError(
            f"Local runner timed out after {timeout_seconds} seconds",
            command=reported_cmd,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            timed_out=True,
        ) from exc

    if completed.returncode != 0:
        raise LocalRunnerError(
            f"Local runner failed with exit code {completed.returncode}",
            command=reported_cmd,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    return extract_model_output(completed.stdout)


def extract_json_runtime_text(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ""

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]

    for key in ("content", "response", "completion", "generated_text", "text"):
        value = payload.get(key)
        if isinstance(value, str):
            return value

    return ""


def is_local_json_endpoint(endpoint: str) -> bool:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in {"http", "https"}:
        return False
    hostname = parsed.hostname or ""
    return hostname.lower() in {"localhost", "127.0.0.1", "::1"}


def require_local_json_endpoint(endpoint: str) -> None:
    if not is_local_json_endpoint(endpoint):
        raise LocalRunnerError(
            "JSON runtime endpoint must be localhost, 127.0.0.1, or ::1",
            command=["json-http", endpoint],
        )


def run_json_http(
    endpoint: str,
    prompt: str,
    max_tokens: int,
    timeout_seconds: int,
) -> str:
    command = ["json-http", endpoint]
    require_local_json_endpoint(endpoint)
    body = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds if timeout_seconds > 0 else None,
        ) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise LocalRunnerError(
            f"JSON runtime failed with HTTP {exc.code}",
            command=command,
            returncode=exc.code,
            stdout=raw,
        ) from exc
    except TimeoutError as exc:
        raise LocalRunnerError(
            f"JSON runtime timed out after {timeout_seconds} seconds",
            command=command,
            timed_out=True,
        ) from exc
    except urllib.error.URLError as exc:
        raise LocalRunnerError(
            f"JSON runtime request failed: {exc.reason}",
            command=command,
            stderr=str(exc),
        ) from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LocalRunnerError(
            "JSON runtime returned invalid JSON",
            command=command,
            stdout=raw,
        ) from exc

    text = extract_json_runtime_text(payload)
    if not text:
        raise LocalRunnerError(
            "JSON runtime response did not include model text",
            command=command,
            stdout=raw,
        )
    return extract_model_output(text)


def answer_from_evidence(packet: dict) -> dict:
    citations = []
    for item in packet.get("evidence", []):
        citation = item.get("citation", {})
        citations.append(
            {
                "rank": item.get("rank"),
                "path": citation.get("path"),
                "line_start": citation.get("line_start"),
                "line_end": citation.get("line_end"),
                "title": citation.get("title"),
                "source_type": citation.get("source_type"),
            }
        )
    return {
        "schema": ANSWER_SCHEMA,
        "mode": "runtime_unavailable",
        "question": packet.get("question", ""),
        "answer": "",
        "citations": citations,
        "unknowns": packet.get("missing_evidence_notes", []),
        "unsupported_or_not_implemented": [
            "No local model runner was configured for this Phase 1B call."
        ],
    }


def cmd_prompt(args: argparse.Namespace) -> int:
    packet = load_packet(args.packet)
    print(build_prompt(packet, args.max_chars_per_evidence))
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    packet = load_packet(args.packet)
    prompt = build_prompt(packet, args.max_chars_per_evidence)

    if args.runtime == "none":
        answer = answer_from_evidence(packet)
        if args.json:
            print(json.dumps(answer, indent=2, ensure_ascii=True))
        else:
            print("Runtime unavailable: no local model runner configured.")
            print("Citations:")
            for citation in answer["citations"]:
                print(
                    f"- {citation['path']}:{citation['line_start']}-"
                    f"{citation['line_end']} ({citation['title']})"
                )
        return 0

    if args.runtime == "llama-cli":
        try:
            text = run_llama_cli(
                args.runner,
                args.model,
                prompt,
                args.max_tokens,
                args.timeout_seconds,
            )
        except LocalRunnerError as exc:
            if args.json:
                print(json.dumps(runner_error_answer(packet, exc), indent=2, ensure_ascii=True))
            else:
                print(f"Runtime unavailable: {exc}", file=sys.stderr)
                if exc.returncode is not None:
                    print(f"Runner exit code: {exc.returncode}", file=sys.stderr)
                if exc.stderr:
                    print(trim_runner_output(exc.stderr), file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Runtime unavailable: {exc}", file=sys.stderr)
            return 2
        result = answer_from_model_output(packet, text)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=True))
        else:
            print(text)
        return 0

    if args.runtime == "json-http":
        try:
            text = run_json_http(
                args.endpoint,
                prompt,
                args.max_tokens,
                args.timeout_seconds,
            )
        except LocalRunnerError as exc:
            if args.json:
                print(json.dumps(runner_error_answer(packet, exc), indent=2, ensure_ascii=True))
            else:
                print(f"Runtime unavailable: {exc}", file=sys.stderr)
                if exc.returncode is not None:
                    print(f"Runtime status code: {exc.returncode}", file=sys.stderr)
                if exc.stderr:
                    print(trim_runner_output(exc.stderr), file=sys.stderr)
            return 2
        result = answer_from_model_output(packet, text, mode="json-http")
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=True))
        else:
            print(text)
        return 0

    print(f"Unsupported runtime: {args.runtime}", file=sys.stderr)
    return 2


def run_json_http_fixture_checks(prompt: str) -> list[str]:
    failures: list[str] = []
    seen_bodies: list[dict] = []

    if not is_local_json_endpoint("http://127.0.0.1:8080/completion"):
        failures.append("local JSON endpoint was rejected")
    if is_local_json_endpoint("https://example.com/completion"):
        failures.append("non-local JSON endpoint was accepted")
    try:
        run_json_http(
            "https://example.com/completion",
            prompt,
            max_tokens=17,
            timeout_seconds=1,
        )
        failures.append("non-local JSON runtime request was not rejected")
    except LocalRunnerError as exc:
        if "localhost" not in str(exc):
            failures.append("non-local JSON runtime rejection reason missing")

    class FixtureHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = {}
            seen_bodies.append(body)

            if self.path == "/ok":
                payload = {
                    "content": "\n".join(
                        [
                            "Answer: fixture JSON runtime answer.",
                            "Evidence: fixture evidence.",
                            "Unknowns: none.",
                            "Unsupported or not implemented: none.",
                        ]
                    )
                }
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return

            if self.path == "/fail":
                encoded = b'{"error":"fixture unavailable"}'
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        text = run_json_http(f"{base}/ok", prompt, max_tokens=17, timeout_seconds=5)
        if not text.startswith("Answer: fixture JSON runtime answer."):
            failures.append("fixture JSON runtime success returned wrong text")
        if not seen_bodies or seen_bodies[0].get("prompt") != prompt:
            failures.append("fixture JSON runtime request prompt missing")
        if not seen_bodies or seen_bodies[0].get("n_predict") != 17:
            failures.append("fixture JSON runtime request token limit missing")

        try:
            run_json_http(f"{base}/fail", prompt, max_tokens=17, timeout_seconds=5)
            failures.append("fixture JSON runtime failure did not raise")
        except LocalRunnerError as exc:
            if exc.returncode != 503:
                failures.append("fixture JSON runtime failure status missing")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    return failures


def cmd_check(args: argparse.Namespace) -> int:
    packet = load_packet(args.packet)
    prompt = build_prompt(packet, args.max_chars_per_evidence)
    failed = 0

    required = [
        "Answer only from the cited evidence below.",
        "cyxwiz-engine/src/core/graph_compiler.cpp",
        "DataLoader pin_memory=true is unsupported by current",
        "batchers and will be ignored",
        "backend exists",
    ]
    for text in required:
        if text not in prompt:
            print(f"FAIL: prompt missing {text!r}")
            failed += 1
    if MODEL_OUTPUT_SENTINEL not in prompt:
        print("FAIL: prompt output sentinel missing")
        failed += 1
    echoed = "\n".join(
        [
            "llama-cli banner",
            prompt,
            "Answer: clean answer",
            "Evidence: clean evidence",
        ]
    )
    if not extract_model_output(echoed).startswith("Answer: clean answer"):
        print("FAIL: echoed prompt was not stripped before parsing")
        failed += 1

    answer = answer_from_evidence(packet)
    if answer["schema"] != ANSWER_SCHEMA:
        print("FAIL: answer schema mismatch")
        failed += 1
    if answer["mode"] != "runtime_unavailable":
        print("FAIL: runtime-unavailable mode missing")
        failed += 1
    if not answer["citations"]:
        print("FAIL: answer citations missing")
        failed += 1
    error = runner_error_answer(
        packet,
        LocalRunnerError(
            "test runner failure",
            command=["llama-cli", "-m", "model.gguf", "-p", "<prompt omitted; chars=1>", "-n", "1"],
            returncode=1,
            stderr="test stderr",
        ),
    )
    if error["mode"] != "runtime_error":
        print("FAIL: runtime-error mode missing")
        failed += 1
    if error["runner"]["returncode"] != 1:
        print("FAIL: runtime-error return code missing")
        failed += 1
    model_answer = answer_from_model_output(
        packet,
        "\n".join(
            [
                "Answer: pin_memory=true is serialized for compatibility.",
                "Evidence: graph_compiler.cpp warning text.",
                "Unknowns: none.",
                "Unsupported or not implemented: pinned transfer backend.",
            ]
        ),
    )
    if model_answer["answer"] != "pin_memory=true is serialized for compatibility.":
        print("FAIL: structured model answer was not parsed")
        failed += 1
    if not model_answer["model_output_sections"]["parsed"]:
        print("FAIL: structured model answer parsed flag missing")
        failed += 1
    partial_model_answer = answer_from_model_output(
        packet,
        "Answer: only answer section.",
        mode="json-http",
    )
    partial_sections = partial_model_answer["model_output_sections"]
    partial_present = [
        key
        for key in ANSWER_SECTION_KEYS.values()
        if isinstance(partial_sections.get(key), str) and partial_sections.get(key)
    ]
    partial_missing = [key for key in ANSWER_SECTION_KEYS.values() if key not in partial_present]
    if not partial_missing:
        print("FAIL: partial structured answer did not report missing sections")
        failed += 1
    json_text = extract_json_runtime_text(
        {
            "content": "\n".join(
                [
                    "Answer: JSON runtime answer.",
                    "Evidence: graph_compiler.cpp.",
                    "Unknowns: none.",
                    "Unsupported or not implemented: none.",
                ]
            )
        }
    )
    json_answer = answer_from_model_output(packet, json_text, mode="json-http")
    if json_answer["mode"] != "json-http":
        print("FAIL: JSON runtime answer mode missing")
        failed += 1
    if json_answer["answer"] != "JSON runtime answer.":
        print("FAIL: JSON runtime answer was not parsed")
        failed += 1
    openai_text = extract_json_runtime_text(
        {
            "choices": [
                {
                    "message": {
                        "content": "Answer: OpenAI-compatible local response."
                    }
                }
            ]
        }
    )
    if not openai_text.startswith("Answer: OpenAI-compatible"):
        print("FAIL: OpenAI-compatible JSON runtime text was not extracted")
        failed += 1
    json_error = runner_error_answer(
        packet,
        LocalRunnerError(
            "test JSON runtime failure",
            command=["json-http", "http://127.0.0.1:8080/completion"],
            returncode=503,
            stdout='{"error":"unavailable"}',
        ),
    )
    if json_error["mode"] != "runtime_error" or json_error["runner"]["returncode"] != 503:
        print("FAIL: JSON runtime error envelope missing")
        failed += 1
    for message in run_json_http_fixture_checks(prompt):
        print(f"FAIL: {message}")
        failed += 1

    if failed:
        print(f"{failed} Phase 1B check(s) failed.")
        return 1

    print("All Phase 1B adapter checks passed.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    runner_path = shutil.which(args.runner) if not Path(args.runner).exists() else str(args.runner)
    model_exists = args.model.exists()
    packet_status = {
        "provided": args.packet is not None,
        "valid": None,
        "schema": None,
        "evidence_count": 0,
        "error": "",
    }

    if args.packet is not None:
        try:
            packet = load_packet(args.packet)
            packet_status["valid"] = True
            packet_status["schema"] = packet.get("schema")
            packet_status["evidence_count"] = len(packet.get("evidence", []))
        except Exception as exc:
            packet_status["valid"] = False
            packet_status["error"] = str(exc)

    ready = args.runtime == "none"
    if args.runtime == "llama-cli":
        ready = runner_path is not None and model_exists
    if args.runtime == "json-http":
        ready = bool(args.endpoint) and is_local_json_endpoint(args.endpoint)

    report = {
        "schema": "cyxwiz.tofix42.phase1b.doctor.v1",
        "runtime": args.runtime,
        "runner": {
            "requested": args.runner,
            "available": runner_path is not None,
            "resolved_path": runner_path or "",
        },
        "endpoint": {
            "url": args.endpoint,
            "probed": False,
            "local_only": True,
            "allowed": is_local_json_endpoint(args.endpoint),
        },
        "model": {
            "path": str(args.model),
            "exists": model_exists,
        },
        "packet": packet_status,
        "ready": ready,
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print(f"Runtime: {report['runtime']}")
        print(f"Runner available: {report['runner']['available']}")
        if report["runner"]["resolved_path"]:
            print(f"Runner path: {report['runner']['resolved_path']}")
        if args.runtime == "json-http":
            print(f"Endpoint: {report['endpoint']['url']} (not probed)")
            print(f"Endpoint allowed: {report['endpoint']['allowed']}")
        print(f"Model exists: {report['model']['exists']} ({report['model']['path']})")
        if packet_status["provided"]:
            print(f"Packet valid: {packet_status['valid']}")
            print(f"Packet evidence count: {packet_status['evidence_count']}")
            if packet_status["error"]:
                print(f"Packet error: {packet_status['error']}")
        print(f"Ready: {report['ready']}")

    if packet_status["provided"] and not packet_status["valid"]:
        return 1
    if args.runtime == "llama-cli" and not report["ready"]:
        return 1
    if args.runtime == "json-http" and not report["ready"]:
        return 1
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    packet = load_packet(args.packet)
    prompt = build_prompt(packet, args.max_chars_per_evidence)
    report = {
        "schema": "cyxwiz.tofix42.phase2.runtime_probe.v1",
        "runtime": "json-http",
        "endpoint": {
            "url": args.endpoint,
            "local_only": True,
            "allowed": is_local_json_endpoint(args.endpoint),
        },
        "packet": {
            "schema": packet.get("schema"),
            "evidence_count": len(packet.get("evidence", [])),
        },
        "ok": False,
        "parsed": False,
        "sections_present": [],
        "sections_missing": list(ANSWER_SECTION_KEYS.values()),
        "answer": "",
        "error": "",
    }

    try:
        text = run_json_http(
            args.endpoint,
            prompt,
            args.max_tokens,
            args.timeout_seconds,
        )
        answer = answer_from_model_output(packet, text, mode="json-http")
        sections = answer.get("model_output_sections", {})
        present = [
            key
            for key in ANSWER_SECTION_KEYS.values()
            if isinstance(sections.get(key), str) and sections.get(key)
        ]
        missing = [key for key in ANSWER_SECTION_KEYS.values() if key not in present]
        report["parsed"] = bool(sections.get("parsed"))
        report["sections_present"] = present
        report["sections_missing"] = missing
        report["answer"] = answer.get("answer", "")
        report["ok"] = report["parsed"] and not missing
    except LocalRunnerError as exc:
        report["error"] = str(exc)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print(f"Runtime: {report['runtime']}")
        print(f"Endpoint: {report['endpoint']['url']}")
        print(f"Endpoint allowed: {report['endpoint']['allowed']}")
        print(f"Parsed sections: {report['parsed']}")
        print(f"Sections present: {', '.join(report['sections_present'])}")
        if report["sections_missing"]:
            print(f"Sections missing: {', '.join(report['sections_missing'])}")
        print(f"OK: {report['ok']}")
        if report["error"]:
            print(f"Error: {report['error']}")

    return 0 if report["ok"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="tofix42 Phase 1B answer adapter")
    sub = parser.add_subparsers(dest="command", required=True)

    prompt_parser = sub.add_parser("prompt", help="build a strict evidence prompt")
    prompt_parser.add_argument("--packet", type=Path, default=Path("-"))
    prompt_parser.add_argument("--max-chars-per-evidence", type=int, default=2400)
    prompt_parser.set_defaults(func=cmd_prompt)

    answer_parser = sub.add_parser("answer", help="answer using a configured local runtime")
    answer_parser.add_argument("--packet", type=Path, default=Path("-"))
    answer_parser.add_argument(
        "--runtime",
        choices=["none", "llama-cli", "json-http"],
        default="none",
    )
    answer_parser.add_argument("--runner", default="llama-cli")
    answer_parser.add_argument("--model", type=Path, default=Path("smollm-135m.Q4_K_M.gguf"))
    answer_parser.add_argument("--endpoint", default="http://127.0.0.1:8080/completion")
    answer_parser.add_argument("--max-tokens", type=int, default=384)
    answer_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=0,
        help="terminate local runner after this many seconds; 0 disables timeout",
    )
    answer_parser.add_argument("--max-chars-per-evidence", type=int, default=2400)
    answer_parser.add_argument("--json", action="store_true")
    answer_parser.set_defaults(func=cmd_answer)

    check_parser = sub.add_parser("check", help="run Phase 1B adapter checks")
    check_parser.add_argument("--packet", type=Path, default=Path("-"))
    check_parser.add_argument("--max-chars-per-evidence", type=int, default=2400)
    check_parser.set_defaults(func=cmd_check)

    doctor_parser = sub.add_parser("doctor", help="report Phase 1B runtime readiness")
    doctor_parser.add_argument("--packet", type=Path)
    doctor_parser.add_argument(
        "--runtime",
        choices=["none", "llama-cli", "json-http"],
        default="none",
    )
    doctor_parser.add_argument("--runner", default="llama-cli")
    doctor_parser.add_argument("--model", type=Path, default=Path("smollm-135m.Q4_K_M.gguf"))
    doctor_parser.add_argument("--endpoint", default="http://127.0.0.1:8080/completion")
    doctor_parser.add_argument("--json", action="store_true")
    doctor_parser.set_defaults(func=cmd_doctor)

    probe_parser = sub.add_parser("probe", help="probe a manually started local JSON runtime")
    probe_parser.add_argument("--packet", type=Path, default=Path("-"))
    probe_parser.add_argument("--endpoint", default="http://127.0.0.1:8080/completion")
    probe_parser.add_argument("--max-tokens", type=int, default=384)
    probe_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="terminate local JSON runtime probe after this many seconds",
    )
    probe_parser.add_argument("--max-chars-per-evidence", type=int, default=2400)
    probe_parser.add_argument("--json", action="store_true")
    probe_parser.set_defaults(func=cmd_probe)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
