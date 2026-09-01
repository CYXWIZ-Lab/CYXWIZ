#!/usr/bin/env python3
"""Classify compiler warnings and enforce the CyxWiz-owned warning baseline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


CLANG_GCC_WARNING = re.compile(
    r"^(?P<path>(?:[A-Za-z]:)?[^:\r\n]+):(?P<line>\d+):"
    r"(?:(?P<column>\d+):)?\s*warning:\s*(?P<message>.*)$"
)
MSVC_WARNING = re.compile(
    r"^(?P<path>.+?)\((?P<line>\d+)(?:,(?P<column>\d+))?\):\s*"
    r"warning\s+(?P<code>[A-Z]+\d+):\s*(?P<message>.*)$",
    re.IGNORECASE,
)
MSVC_TOOL_WARNING = re.compile(
    r"^(?P<tool>cl|link)(?:\.exe)?\s*:\s*(?:command line\s+)?"
    r"warning\s+(?P<code>[A-Z]+\d+):\s*(?P<message>.*)$",
    re.IGNORECASE,
)
WARNING_FLAG = re.compile(r"\s+\[(?P<code>-W[^\]]+)\]\s*$")
PROJECT_SUFFIX = re.compile(r"\s+\[(?P<project>[^\]]+\.vcxproj)\]\s*$", re.IGNORECASE)
GENERIC_WARNING = re.compile(
    r"^(?P<tool>ld|link|clang(?:\+\+)?|gcc|g\+\+|cmake|ninja):\s*warning:\s*(?P<message>.*)$",
    re.IGNORECASE,
)
TARGET_FROM_OBJECT = re.compile(r"CMakeFiles[/\\](?P<target>[^/\\]+)\.dir[/\\]")

OWNED_ROOTS = (
    "cyxwiz-backend/",
    "cyxwiz-engine/",
    "cyxwiz-protocol/",
    "cyxwiz-server-node/",
    "plugins/",
    "redist/",
    "tests/",
)
DEPENDENCY_SEGMENTS = (
    "/build/",
    "/external/",
    "/third_party/",
    "/vcpkg/",
    "/vcpkg_installed/",
)
VENDORED_SOURCES = (
    "cyxwiz-engine/src/plugin/security/tweetnacl.c",
    "cyxwiz-engine/external/",
)


@dataclass(frozen=True)
class Diagnostic:
    source: str | None
    line: int | None
    column: int | None
    code: str
    message: str
    target: str
    ownership: str


def _slash_path(value: str) -> str:
    return value.strip().replace("\\", "/")


def _relative_source(source: str, repo_root: Path) -> str:
    normalized = _slash_path(source)
    root = _slash_path(str(repo_root.resolve())).rstrip("/")
    if normalized == root:
        return "."
    if normalized.startswith(root + "/"):
        return normalized[len(root) + 1 :]
    checkout_marker = f"/{repo_root.resolve().name}/"
    # GitHub's Windows checkout commonly repeats the repository name
    # (for example D:/a/CYXWIZ/CYXWIZ/...). The last marker identifies the
    # actual repository-relative source path.
    marker_offset = normalized.lower().rfind(checkout_marker.lower())
    if marker_offset >= 0:
        return normalized[marker_offset + len(checkout_marker) :]
    return normalized.removeprefix("./")


def classify_source(source: str | None, repo_root: Path) -> str:
    if source is None:
        return "toolchain"

    relative = _relative_source(source, repo_root)
    comparable = "/" + relative.lower().lstrip("/")

    if any(relative.startswith(path) for path in VENDORED_SOURCES):
        return "vendored"
    if any(segment in comparable for segment in DEPENDENCY_SEGMENTS):
        return "generated" if "/build/" in comparable else "external"
    if any(relative.startswith(root) for root in OWNED_ROOTS):
        return "owned"

    # CI compilers commonly print an absolute checkout path. Recognize an
    # owned root only at a complete path boundary, after dependency paths have
    # already been excluded.
    if any(f"/{root}" in comparable for root in OWNED_ROOTS):
        return "owned"
    return "external"


def _target_from_line(line: str, current_target: str) -> str:
    match = TARGET_FROM_OBJECT.search(line)
    return match.group("target") if match else current_target


def parse_diagnostics(lines: Iterable[str], repo_root: Path) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    source_diagnostic_keys: set[tuple[object, ...]] = set()
    current_target = "unknown"

    def append(diagnostic: Diagnostic) -> None:
        if diagnostic.source is None:
            diagnostics.append(diagnostic)
            return
        # MSVC emits template expansion details as several warning-prefixed
        # lines at the same source location. The source location, code, and
        # target identify one compiler diagnostic; retain its primary line.
        key = (
            diagnostic.source,
            diagnostic.line,
            diagnostic.column,
            diagnostic.code,
            diagnostic.target,
        )
        if key not in source_diagnostic_keys:
            source_diagnostic_keys.add(key)
            diagnostics.append(diagnostic)

    for raw_line in lines:
        line = raw_line.strip()
        current_target = _target_from_line(line, current_target)

        match = MSVC_WARNING.match(line)
        if match:
            message = match.group("message")
            project = PROJECT_SUFFIX.search(message)
            target = current_target
            if project:
                target = Path(_slash_path(project.group("project"))).stem
                message = message[: project.start()].rstrip()
            source = _relative_source(match.group("path"), repo_root)
            append(
                Diagnostic(
                    source=source,
                    line=int(match.group("line")),
                    column=int(match.group("column")) if match.group("column") else None,
                    code=match.group("code").upper(),
                    message=message,
                    target=target,
                    ownership=classify_source(source, repo_root),
                )
            )
            continue

        match = MSVC_TOOL_WARNING.match(line)
        if match:
            message = match.group("message")
            project = PROJECT_SUFFIX.search(message)
            target = current_target
            if project:
                target = Path(_slash_path(project.group("project"))).stem
                message = message[: project.start()].rstrip()
            append(
                Diagnostic(
                    source=None,
                    line=None,
                    column=None,
                    code=match.group("code").upper(),
                    message=message,
                    target=target,
                    ownership="toolchain",
                )
            )
            continue

        match = CLANG_GCC_WARNING.match(line)
        if match:
            message = match.group("message")
            flag = WARNING_FLAG.search(message)
            code = flag.group("code") if flag else "unclassified"
            if flag:
                message = message[: flag.start()].rstrip()
            source = _relative_source(match.group("path"), repo_root)
            append(
                Diagnostic(
                    source=source,
                    line=int(match.group("line")),
                    column=int(match.group("column")) if match.group("column") else None,
                    code=code,
                    message=message,
                    target=current_target,
                    ownership=classify_source(source, repo_root),
                )
            )
            continue

        match = GENERIC_WARNING.match(line)
        if match:
            diagnostics.append(
                Diagnostic(
                    source=None,
                    line=None,
                    column=None,
                    code=match.group("tool").lower(),
                    message=match.group("message"),
                    target=current_target,
                    ownership="toolchain",
                )
            )

    return diagnostics


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def build_report(
    diagnostics: Sequence[Diagnostic], platform: str, owned_limit: int
) -> dict[str, object]:
    owned = [item for item in diagnostics if item.ownership == "owned"]
    owned_count = len(owned)
    return {
        "schema_version": 1,
        "platform": platform,
        "passed": owned_count <= owned_limit,
        "owned_warning_limit": owned_limit,
        "total_warnings": len(diagnostics),
        "owned_warnings": owned_count,
        "counts_by_ownership": _sorted_counts(item.ownership for item in diagnostics),
        "counts_by_code": _sorted_counts(item.code for item in diagnostics),
        "counts_by_target": _sorted_counts(item.target for item in diagnostics),
        "owned_counts_by_code": _sorted_counts(item.code for item in owned),
        "owned_counts_by_target": _sorted_counts(item.target for item in owned),
        "diagnostics": [asdict(item) for item in diagnostics],
    }


def render_markdown(report: dict[str, object]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    lines = [
        "# CyxWiz compiler warning report",
        "",
        f"- Platform: `{report['platform']}`",
        f"- Gate: **{status}**",
        f"- CyxWiz-owned warnings: `{report['owned_warnings']}` "
        f"(limit `{report['owned_warning_limit']}`)",
        f"- All classified warnings: `{report['total_warnings']}`",
        "",
        "## Counts by ownership",
        "",
        "| Ownership | Count |",
        "| --- | ---: |",
    ]
    ownership_counts = report["counts_by_ownership"]
    assert isinstance(ownership_counts, dict)
    for ownership, count in ownership_counts.items():
        lines.append(f"| {ownership} | {count} |")

    for title, key, heading in (
        ("Counts by diagnostic code", "counts_by_code", "Code"),
        ("Counts by target", "counts_by_target", "Target"),
    ):
        lines.extend(["", f"## {title}", "", f"| {heading} | Count |", "| --- | ---: |"])
        counts = report[key]
        assert isinstance(counts, dict)
        for value, count in counts.items():
            lines.append(f"| `{value}` | {count} |")

    lines.extend(["", "## CyxWiz-owned diagnostics", ""])
    diagnostics = report["diagnostics"]
    assert isinstance(diagnostics, list)
    owned = [item for item in diagnostics if item["ownership"] == "owned"]
    if not owned:
        lines.append("None.")
    else:
        for item in owned:
            location = f"{item['source']}:{item['line']}"
            if item["column"] is not None:
                location += f":{item['column']}"
            lines.append(
                f"- `{location}` `{item['code']}` ({item['target']}): {item['message']}"
            )
    lines.append("")
    return "\n".join(lines)


def load_owned_limit(baseline_path: Path, platform: str) -> int:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    limits = baseline.get("owned_warning_limit")
    if not isinstance(limits, dict):
        raise ValueError("baseline must contain an owned_warning_limit object")
    limit = limits.get(platform, limits.get("default"))
    if not isinstance(limit, int) or limit < 0:
        raise ValueError(f"baseline has no non-negative limit for platform '{platform}'")
    return limit


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="captured compiler output")
    parser.add_argument("--platform", required=True, help="windows, linux, or macos")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        owned_limit = load_owned_limit(args.baseline, args.platform)
        diagnostics = parse_diagnostics(
            args.log.read_text(encoding="utf-8", errors="replace").splitlines(),
            args.repo_root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"warning report failed: {error}", file=sys.stderr)
        return 2

    report = build_report(diagnostics, args.platform, owned_limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "warning-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "warning-report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(render_markdown(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
