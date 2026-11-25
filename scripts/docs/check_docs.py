#!/usr/bin/env python3
"""Check the CyxWiz docs: front matter on pages in the new tree, internal links everywhere.

Usage:  python scripts/docs/check_docs.py [--docs DIR]

Runs offline and changes nothing. Exit code 1 when an error is found.

- Pages in the new tree (see docs/README.md) must have complete, valid front matter.
- Every relative link and image in docs/ must point to a file that exists (anchors are not checked yet).
- Published pages must not contain internal ticket numbers, Windows drive paths or IPv4 addresses.
Pages still in the old folders (usage/, extending/, architecture/, research/) are link-checked only.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

NEW_TREE = {
    "install", "tutorials", "user-guide", "main-window", "concepts", "reference", "sdk", "python",
    "server-and-cloud", "distributed-training", "troubleshooting", "whats-new",
    "release-notes", "internals", "security", "legal", "about",
}
NEW_TREE_FILES = {"index.md", "faq.md", "known-issues.md", "glossary.md", "deprecations.md"}
SKIP_DIRS = {"templates", "next"}  # templates hold placeholders; release-note fragments have no front matter

REQUIRED = ["title", "description", "type", "area", "status", "since", "owner", "last_reviewed"]
ALLOWED = {
    "type": {"tutorial", "how-to", "reference", "explanation", "landing", "troubleshooting", "release-notes"},
    "area": {"general", "install", "graph-editor", "data-studio", "training", "evaluation", "deployment",
             "simulation", "scripting", "plugins", "remote", "sdk", "python", "server-and-cloud", "internals",
             "security", "legal", "about"},
    "audience": {"user", "plugin-developer", "operator", "contributor", "organisation"},
    "status": {"experimental", "beta", "stable", "deprecated", "removed"},
}
LOCAL_ADDRESSES = {"127.0.0.1", "0.0.0.0"}  # documented on purpose (localhost, listen on all)
VERSION = re.compile(r"^\d+\.\d+(\.\d+)?$")
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FORBIDDEN = [
    (re.compile(r"\bTOFIX\d+", re.I), "internal ticket number"),
    (re.compile(r"\b[A-Z]:[\\/](?:Dev|dev|Users|demo|Docs)\b"), "local Windows path"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "IPv4 address"),
]


def front_matter(text: str) -> tuple[dict[str, str | list[str]] | None, int]:
    """Parse simple YAML front matter (scalars and '- item' lists). Returns (fields, body start line)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, 0
    fields: dict[str, str | list[str]] = {}
    key = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return fields, i + 1
        if re.match(r"^\s+-\s+", line) and key:
            value = fields.get(key)
            items = value if isinstance(value, list) else []
            items.append(line.split("-", 1)[1].split("#", 1)[0].strip())
            fields[key] = items
            continue
        m = re.match(r"^([A-Za-z_]+):\s*(.*)$", line)
        if m:
            key = m.group(1)
            raw = m.group(2).split(" #", 1)[0].strip()
            fields[key] = [] if raw == "[]" else raw.strip('"\'')
    return None, 0


def in_new_tree(rel: Path) -> bool:
    if rel.parts[0] in NEW_TREE:
        return not (set(rel.parts[:-1]) & SKIP_DIRS)
    return len(rel.parts) == 1 and rel.name in NEW_TREE_FILES


def check_page(path: Path, rel: Path, docs: Path, errors: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    where = f"docs/{rel.as_posix()}"
    fields, body_start = front_matter(text)
    new = in_new_tree(rel)
    if new:
        if fields is None:
            errors.append(f"{where}: missing front matter")
        else:
            for name in REQUIRED:
                if not fields.get(name):
                    errors.append(f"{where}: front matter '{name}' is missing")
            for name, allowed in ALLOWED.items():
                value = fields.get(name)
                if isinstance(value, str) and value and value not in allowed:
                    errors.append(f"{where}: '{name}: {value}' is not one of {sorted(allowed)}")
            since = fields.get("since")
            if isinstance(since, str) and since and not VERSION.match(since):
                errors.append(f"{where}: 'since: {since}' is not a version like 1.0.7")
            reviewed = fields.get("last_reviewed")
            if isinstance(reviewed, str) and reviewed:
                try:
                    if date.fromisoformat(reviewed) > date.today():
                        errors.append(f"{where}: last_reviewed {reviewed} is in the future")
                except ValueError:
                    errors.append(f"{where}: last_reviewed '{reviewed}' is not YYYY-MM-DD")
            title = fields.get("title")
            if isinstance(title, str) and len(title) > 70:
                errors.append(f"{where}: title is longer than 70 characters")
            if fields.get("status") in {"deprecated", "removed"}:
                for name in ("until", "replaced_by"):
                    if not fields.get(name):
                        errors.append(f"{where}: a {fields.get('status')} page needs '{name}'")
            for old in fields.get("redirect_from") or []:
                target = (path.parent / old).resolve()
                if target == path.resolve():
                    errors.append(f"{where}: redirect_from '{old}' points to this page itself")
                elif target.exists() and "stub" not in target.read_text(encoding="utf-8", errors="replace")[:400].lower():
                    errors.append(f"{where}: redirect_from '{old}' is an existing page, not an old path")
            for name in ("related", "replaced_by"):
                targets = fields.get(name) or []
                for target in [targets] if isinstance(targets, str) else targets:
                    if target and not (path.parent / target).resolve().exists():
                        errors.append(f"{where}: {name} '{target}' does not exist")
        body = text.splitlines()[body_start:]
        for n, line in enumerate(body, start=body_start + 1):
            for pattern, what in FORBIDDEN:
                hit = pattern.search(line)
                if hit and hit.group(0) not in LOCAL_ADDRESSES:
                    errors.append(f"{where}:{n}: {what} in a published page")
    if "templates" in rel.parts:
        return
    for n, line in enumerate(text.splitlines(), start=1):
        for target in LINK.findall(line):
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                continue
            file_part = target.split("#", 1)[0]
            if not file_part:
                continue
            if not (path.parent / file_part).resolve().exists():
                errors.append(f"{where}:{n}: broken link '{target}'")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--docs", default=str(Path(__file__).resolve().parents[2] / "docs"))
    args = parser.parse_args()
    docs = Path(args.docs).resolve()
    errors: list[str] = []
    pages = sorted(docs.rglob("*.md"))
    for path in pages:
        check_page(path, path.relative_to(docs), docs, errors)
    for line in errors:
        print(line)
    new_pages = sum(1 for p in pages if in_new_tree(p.relative_to(docs)))
    print(f"\nChecked {len(pages)} pages ({new_pages} in the new tree): {len(errors)} error(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
