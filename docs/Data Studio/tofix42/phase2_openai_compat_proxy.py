#!/usr/bin/env python3
"""Local proxy from tofix42 /completion to an OpenAI-compatible chat endpoint.

This is a localhost-only bridge for manually started local model servers such
as LM Studio, llama.cpp server, or other OpenAI-compatible runtimes. It does
not download, start, or validate a model by itself.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import phase1b_answer as answer_adapter


REPORT_SCHEMA = "cyxwiz.tofix42.phase2.openai_compat_proxy.v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766
DEFAULT_UPSTREAM = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_MODEL = "local-model"


@dataclass(frozen=True)
class ProxyConfig:
    host: str
    port: int
    upstream: str
    model: str
    timeout_seconds: int
    api_key: str


def is_local_host(host: str) -> bool:
    return host.lower() in {"localhost", "127.0.0.1", "::1"}


def require_local_host(host: str, label: str) -> None:
    if not is_local_host(host):
        raise ValueError(f"{label} must be localhost, 127.0.0.1, or ::1")


def validate_config(config: ProxyConfig) -> None:
    require_local_host(config.host, "Proxy bind host")
    if not answer_adapter.is_local_json_endpoint(config.upstream):
        raise ValueError("Upstream endpoint must be localhost, 127.0.0.1, or ::1")


def between(text: str, start: str, end: str) -> str:
    try:
        left = text.index(start) + len(start)
        right = text.index(end, left)
    except ValueError:
        return ""
    return text[left:right].strip()


def adapt_phase1b_prompt(prompt: str) -> str:
    question = between(prompt, "Question:\n", "\n\nEvidence:\n")
    evidence = between(prompt, "Evidence:\n", "\n\nMissing evidence notes:\n")
    missing = between(
        prompt,
        "Missing evidence notes:\n",
        "\n\nReturn this structure:\n",
    )
    if question and evidence:
        parts = [
            "Use only the provided CyxWiz evidence.",
            "Question:",
            question,
            "",
            "Evidence:",
            evidence,
        ]
        if missing:
            parts.extend(["", "Missing evidence notes:", missing])
        return "\n".join(parts).strip()

    text = prompt.replace(answer_adapter.MODEL_OUTPUT_SENTINEL, "").strip()
    marker = "\n\nReturn this structure:\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()
    return text


def openai_chat_request(prompt: str, model: str, max_tokens: int) -> dict[str, Any]:
    format_contract = (
        "Respond ONLY using this exact format, one section per line:\n"
        "Answer: <your direct answer>\n"
        "Evidence: <code path(s), symbol names, and minimal evidence>\n"
        "Unknowns: <things you are not sure about>\n"
        "Unsupported or not implemented: <anything not covered by available evidence>\n"
        "Copy any cited file path exactly when you name it.\n"
        "Do not add markdown, extra prose, JSON, or extra headers. "
        "Always include all four sections in this order."
    )
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": format_contract,
            },
            {
                "role": "user",
                "content": adapt_phase1b_prompt(prompt),
            }
        ],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0,
    }


def extract_request_prompt(payload: dict[str, Any]) -> tuple[str, int]:
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("missing prompt")
    max_tokens = payload.get("n_predict", payload.get("max_tokens", 384))
    if not isinstance(max_tokens, int) or max_tokens < 1:
        max_tokens = 384
    return prompt, max_tokens


def forward_completion(config: ProxyConfig, prompt: str, max_tokens: int) -> str:
    body = openai_chat_request(prompt, config.model, max_tokens)
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    request = urllib.request.Request(
        config.upstream,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
        raw = response.read().decode("utf-8", errors="replace")
    payload = json.loads(raw)
    text = answer_adapter.extract_json_runtime_text(payload)
    if not text:
        raise ValueError("upstream response did not include model text")
    return text


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    server_version = "tofix42-openai-compat-proxy/1"

    @property
    def config(self) -> ProxyConfig:
        return self.server.config  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        if self.path == "/health":
            self.write_json(
                200,
                {
                    "schema": REPORT_SCHEMA,
                    "ok": True,
                    "upstream": self.config.upstream,
                    "model": self.config.model,
                },
            )
            return
        self.write_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/completion":
            self.write_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            prompt, max_tokens = extract_request_prompt(payload)
            text = forward_completion(self.config, prompt, max_tokens)
        except json.JSONDecodeError:
            self.write_json(400, {"error": "invalid json"})
            return
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            self.write_json(exc.code, {"error": "upstream_http_error", "detail": error_body})
            return
        except (TimeoutError, urllib.error.URLError) as exc:
            self.write_json(502, {"error": "upstream_request_failed", "detail": str(exc)})
            return
        except Exception as exc:
            self.write_json(400, {"error": str(exc)})
            return

        self.write_json(200, {"content": text})

    def write_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

    def log_message(self, format: str, *args: object) -> None:
        return


class ProxyServer(http.server.ThreadingHTTPServer):
    config: ProxyConfig


class SelfCheckUpstreamHandler(http.server.BaseHTTPRequestHandler):
    server_version = "tofix42-openai-compat-self-check/1"

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.write_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.write_json(400, {"error": "invalid json"})
            return
        self.server.last_request = payload  # type: ignore[attr-defined]
        self.server.last_authorization = self.headers.get("Authorization", "")  # type: ignore[attr-defined]
        self.write_json(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "\n".join(
                                [
                                    "Answer: proxy self-check ok.",
                                    "Evidence: fake local upstream received the prompt.",
                                    "Unknowns: no real model was used.",
                                    "Unsupported or not implemented: answer quality is not validated.",
                                ]
                            )
                        }
                    }
                ]
            },
        )

    def write_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


class SelfCheckUpstreamServer(http.server.ThreadingHTTPServer):
    last_request: dict[str, Any]
    last_authorization: str


def serve(config: ProxyConfig) -> None:
    validate_config(config)
    server = ProxyServer((config.host, config.port), ProxyHandler)
    server.config = config
    print(
        f"tofix42 OpenAI-compatible proxy listening on http://{config.host}:{config.port}/completion",
        flush=True,
    )
    print(f"upstream: {config.upstream}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def forwarding_self_check() -> dict[str, Any]:
    server = SelfCheckUpstreamServer(("127.0.0.1", 0), SelfCheckUpstreamHandler)
    server.last_request = {}
    server.last_authorization = ""
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        text = forward_completion(
            ProxyConfig(
                host="127.0.0.1",
                port=0,
                upstream=f"http://127.0.0.1:{port}/v1/chat/completions",
                model="self-check-model",
                timeout_seconds=5,
                api_key="self-check-token",
            ),
            "Answer: self-check prompt",
            12,
        )
        return {
            "ok": True,
            "text": text,
            "request": server.last_request,
            "authorization": server.last_authorization,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def cmd_self_check(args: argparse.Namespace) -> int:
    sample = openai_chat_request("Answer: hello", "local", 8)
    parsed_text = answer_adapter.extract_json_runtime_text(
        {"choices": [{"message": {"content": "Answer: ok"}}]}
    )
    forwarding = forwarding_self_check()
    forwarded_request = forwarding.get("request", {})
    forwarded_messages = forwarded_request.get("messages", [])
    forwarded_content = ""
    if isinstance(forwarded_messages, list) and forwarded_messages:
        first_message = forwarded_messages[0]
        if isinstance(first_message, dict):
            forwarded_content = str(first_message.get("content", ""))
    checks = [
        sample["model"] == "local",
        sample["messages"][0]["content"] == "Answer: hello",
        sample["max_tokens"] == 8,
        sample["stream"] is False,
        sample["temperature"] == 0,
        parsed_text == "Answer: ok",
        forwarding["ok"],
        str(forwarding["text"]).startswith("Answer: proxy self-check ok."),
        forwarded_request.get("model") == "self-check-model",
        forwarded_request.get("max_tokens") == 12,
        forwarded_request.get("stream") is False,
        forwarded_request.get("temperature") == 0,
        forwarded_content == "Answer: self-check prompt",
        forwarding.get("authorization") == "Bearer self-check-token",
        answer_adapter.is_local_json_endpoint(DEFAULT_UPSTREAM),
        not answer_adapter.is_local_json_endpoint("https://example.com/v1/chat/completions"),
    ]
    if not all(checks):
        print(
            json.dumps(
                {
                    "schema": REPORT_SCHEMA,
                    "ok": False,
                    "sample": sample,
                    "forwarding": forwarding,
                },
                indent=2,
            )
        )
        return 1
    print("All Phase 2 OpenAI-compatible proxy checks passed.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
    serve(
        ProxyConfig(
            host=args.host,
            port=args.port,
            upstream=args.upstream,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            api_key=api_key,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve_parser = sub.add_parser("serve", help="start localhost /completion proxy")
    serve_parser.add_argument("--host", default=DEFAULT_HOST)
    serve_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve_parser.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    serve_parser.add_argument("--model", default=DEFAULT_MODEL)
    serve_parser.add_argument("--timeout-seconds", type=int, default=120)
    serve_parser.add_argument("--api-key-env", default="")
    serve_parser.set_defaults(func=cmd_serve)

    self_check = sub.add_parser("self-check", help="run deterministic proxy checks")
    self_check.set_defaults(func=cmd_self_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
