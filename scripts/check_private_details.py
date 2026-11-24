#!/usr/bin/env python3
"""Check that no private detail is in a tracked file of this public repository.

Usage:  python scripts/check_private_details.py [--patterns FILE]

Private details (lab machine names, their network addresses, remote-login
aliases, personal folders, account names) never go in the repository. They
live in a local file that git ignores:

    config/private-patterns.local.txt   one regular expression per line, # comments

Copy config/private-patterns.example.txt to that name and fill it in. The
patterns themselves are private too, which is why they are read from the
ignored file and not written here.

The script lists every tracked text file with a line that matches, showing the
pattern number but not the matched text, and exits 1 when anything is found.
It changes nothing. With no local patterns file it says so and exits 0.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PATTERNS = REPO / "config" / "private-patterns.local.txt"
SKIP_PREFIXES = ("vcpkg/", "vcpkg-ports/", "third_party/", "external/")
MAX_BYTES = 2_000_000


def load_patterns(path: Path) -> list[re.Pattern[str]]:
    patterns = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            patterns.append(re.compile(text))
        except re.error as error:
            sys.exit(f"{path.name}:{number}: not a valid regular expression ({error})")
    return patterns


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True)
    return [name for name in out.stdout.decode("utf-8").split("\0") if name]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--patterns", type=Path, default=DEFAULT_PATTERNS)
    args = parser.parse_args()
    if not args.patterns.exists():
        print(f"No {args.patterns.name}: copy config/private-patterns.example.txt to it and fill it in.")
        return 0
    patterns = load_patterns(args.patterns)
    hits = 0
    for name in tracked_files():
        if name.startswith(SKIP_PREFIXES):
            continue
        path = REPO / name
        try:
            if not path.is_file() or path.stat().st_size > MAX_BYTES:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue  # binary
        text = data.decode("utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            for index, pattern in enumerate(patterns, start=1):
                if pattern.search(line):
                    print(f"{name}:{number}: matches private pattern {index}")
                    hits += 1
    print(f"\n{len(patterns)} private pattern(s) checked: {hits} match(es).")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
