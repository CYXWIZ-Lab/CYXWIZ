#!/usr/bin/env python3
"""List, per docs area, the code commits made since its docs were last checked.

Usage:  python scripts/docs/docs_drift.py [--files] [--area NAME]

Reads docs/sync.yaml: each area has the code paths it documents and
`synced_at`, the commit its docs were last checked against. For each area the
script runs `git log <synced_at>..HEAD -- <paths>` and prints the commits (and
with --files the files) the docs may need to catch up with. After updating an
area's pages, set its synced_at to the commit you checked against.

Exit code 1 when an area is behind or has never been checked. Changes nothing.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SYNC = REPO / "docs" / "sync.yaml"


def read_sync(path: Path) -> dict[str, dict[str, object]]:
    """Parse the small, fixed shape of sync.yaml: area: / code: [..] / synced_at: <commit>."""
    areas: dict[str, dict[str, object]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        top = re.match(r"^([a-z0-9-]+):\s*$", line)
        if top:
            current = top.group(1)
            areas[current] = {"code": [], "synced_at": ""}
            continue
        field = re.match(r"^\s+(code|synced_at):\s*(.*)$", line)
        if field and current:
            name, value = field.group(1), field.group(2).split(" #", 1)[0].strip()
            if name == "code":
                inner = value.strip("[]")
                areas[current]["code"] = [p.strip() for p in inner.split(",") if p.strip()]
            else:
                areas[current]["synced_at"] = value
    return areas


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--files", action="store_true", help="also list the changed files")
    parser.add_argument("--area", help="only this area")
    args = parser.parse_args()
    areas = read_sync(SYNC)
    if args.area:
        areas = {k: v for k, v in areas.items() if k == args.area}
        if not areas:
            sys.exit(f"no area '{args.area}' in docs/sync.yaml")
    head = git("rev-parse", "--short", "HEAD").strip()
    behind = 0
    for name, area in areas.items():
        paths = [f":(glob){p}" for p in area["code"]]
        synced = str(area["synced_at"])
        if not synced:
            print(f"{name:18} not checked yet (set synced_at after reviewing its pages)")
            behind += 1
            continue
        try:
            git("cat-file", "-e", f"{synced}^{{commit}}")
        except subprocess.CalledProcessError:
            print(f"{name:18} synced_at {synced} is not a commit in this repository")
            behind += 1
            continue
        log = git("log", "--oneline", "--no-merges", f"{synced}..HEAD", "--", *paths).strip()
        if not log:
            print(f"{name:18} up to date (checked at {synced}, HEAD {head})")
            continue
        commits = log.splitlines()
        behind += 1
        print(f"{name:18} {len(commits)} commit(s) since {synced} touch its code:")
        for commit in commits:
            print(f"    {commit}")
        if args.files:
            files = git("diff", "--name-only", f"{synced}..HEAD", "--", *paths).split()
            for file in files:
                print(f"      {file}")
    return 1 if behind else 0


if __name__ == "__main__":
    sys.exit(main())
