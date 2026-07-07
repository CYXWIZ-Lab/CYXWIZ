#!/usr/bin/env python3
"""Local stub JSON runtime for tofix42 Phase 2.

This is not a model. It is a small localhost-only fixture server for manually
testing the Phase 2 JSON HTTP runtime path end to end.
"""

from __future__ import annotations

import argparse
import http.server
import json
import re
from dataclasses import dataclass


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
PROMPT_CITATION_RE = re.compile(r"^\[E\d+\]\s+([^:\n]+):\d+-\d+", re.MULTILINE)


@dataclass
class StubConfig:
    host: str
    port: int


def build_stub_answer(prompt: str) -> str:
    match = PROMPT_CITATION_RE.search(prompt)
    citation_path = match.group(1) if match else "unknown local evidence"
    if "pin_memory" in prompt:
        return "\n".join(
            [
                "Answer: The retrieved evidence says DataLoader pin_memory=true is serialized for compatibility, but current batchers ignore it.",
                f"Evidence: The cited {citation_path} warning states that pin_memory=true is unsupported by current batchers and will be ignored until a pinned host-memory transfer backend exists.",
                "Unknowns: This stub runtime does not inspect any evidence beyond the submitted prompt.",
                "Unsupported or not implemented: A real local model answer has not been validated by this stub.",
            ]
        )

    return "\n".join(
        [
            "Answer: Stub runtime received the prompt.",
            f"Evidence: Stub runtime received cited evidence from {citation_path}.",
            "Unknowns: No real model inference was performed.",
            "Unsupported or not implemented: Real answer quality is not tested by this stub.",
        ]
    )


class StubHandler(http.server.BaseHTTPRequestHandler):
    server_version = "tofix42-stub-runtime/1"

    def do_GET(self) -> None:
        if self.path == "/health":
            self.write_json(200, {"ok": True})
            return
        self.write_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/completion":
            self.write_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            self.write_json(400, {"error": "invalid json"})
            return

        prompt = request.get("prompt")
        if not isinstance(prompt, str):
            self.write_json(400, {"error": "missing prompt"})
            return

        self.write_json(200, {"content": build_stub_answer(prompt)})

    def write_json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


def serve(config: StubConfig) -> None:
    if config.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Stub runtime must bind to localhost only")
    server = http.server.ThreadingHTTPServer((config.host, config.port), StubHandler)
    print(f"tofix42 stub runtime listening on http://{config.host}:{config.port}/completion", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description="tofix42 local JSON runtime stub")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    serve(StubConfig(host=args.host, port=args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
