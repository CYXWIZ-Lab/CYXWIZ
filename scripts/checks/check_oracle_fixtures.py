#!/usr/bin/env python3
"""Committed oracle fixtures move together with the scripts that write them
(TOFIX228 1.4).

The per-push oracle job has no PyTorch: it compares the Engine with fixtures
committed in cyxwiz-engine/tests/computation_truth/fixtures. A fixture is only
as good as the generator it came from, so the manifest
(fixtures/fixture_manifest.json) records, for every fixture, the generator
script and the SHA-256 of both (text hashed with LF line ends). This check
fails when:
  - a generator changed and its fixture did not (re-run the generator),
  - a fixture changed and its generator did not (a hand edit, or a new
    torch version: re-run the generator and record it),
  - a fixture is not in the manifest, or the manifest names a missing file.
After re-running a generator: check_oracle_fixtures.py <root> --update <fixture>...
(--update-all records every entry; use it only after regenerating them all).

Usage: check_oracle_fixtures.py <repo root> [--update <fixture>... | --update-all]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BASE = "cyxwiz-engine/tests/computation_truth"
MANIFEST = f"{BASE}/fixtures/fixture_manifest.json"
TEXT_SUFFIXES = {".json", ".cyxgraph", ".py", ".txt", ".csv"}


def digest(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def fixture_files(root: Path) -> list[str]:
    folder = root / BASE / "fixtures"
    return sorted(p.name for p in folder.iterdir()
                  if p.is_file() and p.suffix in (".json", ".cyxgraph") and p.name != "fixture_manifest.json")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1]).resolve()
    args = sys.argv[2:]
    manifest_path = root / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: dict[str, dict] = manifest["fixtures"]

    if args and args[0] in ("--update", "--update-all"):
        names = list(entries) if args[0] == "--update-all" else args[1:]
        for name in names:
            entry = entries.get(name)
            if entry is None:
                print(f"FAIL: {name} is not in the manifest; add it with its generator first")
                return 1
            entry["fixture_sha256"] = digest(root / BASE / "fixtures" / name)
            entry["generator_sha256"] = digest(root / BASE / entry["generator"])
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print(f"recorded {len(names)} fixture(s)")
        return 0

    problems = []
    on_disk = set(fixture_files(root))
    for name in sorted(on_disk - set(entries)):
        problems.append(f"{name} is not in {MANIFEST}: add it with the generator that writes it")
    for name, entry in sorted(entries.items()):
        fixture = root / BASE / "fixtures" / name
        generator = root / BASE / entry["generator"]
        if not fixture.exists():
            problems.append(f"{name}: the manifest names it, but it does not exist")
            continue
        if not generator.exists():
            problems.append(f"{name}: its generator {entry['generator']} does not exist")
            continue
        fixture_same = digest(fixture) == entry.get("fixture_sha256")
        generator_same = digest(generator) == entry.get("generator_sha256")
        if fixture_same and generator_same:
            continue
        if fixture_same:
            problems.append(f"{name}: {entry['generator']} changed but the fixture did not; re-run the generator, "
                            f"then --update {name}")
        elif generator_same:
            problems.append(f"{name}: the fixture changed but {entry['generator']} did not (hand edit or a new "
                            f"reference version); re-run the generator, then --update {name}")
        else:
            problems.append(f"{name}: fixture and generator both changed; if the fixture is the generator's "
                            f"output, record it with --update {name}")
    for problem in problems:
        print(f"FAIL: {problem}")
    print(f"oracle fixtures: {len(entries)} recorded, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
