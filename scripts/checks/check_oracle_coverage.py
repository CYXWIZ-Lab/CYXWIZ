#!/usr/bin/env python3
"""Every public training component has an oracle row, and every row's evidence
exists (TOFIX228 1.4, ORA-09).

The components are read from the code, not written by hand: the public
layers, modules, activations, losses, optimizers and schedulers of
cyxwiz-backend/include/cyxwiz, and the Engine's batchers. The evidence map,
cyxwiz-engine/tests/computation_truth/oracle_coverage.json, gives each one
either the oracle that checks it (an oracle test of CYXWIZ_ORACLE_TESTS, a
fixture file and a case in it) or a named gap with its reason. This check
fails when:
  - a component has no row (a new layer, loss, ... must say how it is checked),
  - a row names a component that no longer exists,
  - an evidence item points at a test that is not an oracle test, a fixture
    that does not exist or a case the fixture does not have,
  - the generated inventory (coverage_inventory.json) is not what the code
    and the map give today (run with --write to refresh it).

Usage: check_oracle_coverage.py <repo root> [--write]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BACKEND_BASES = ("Module", "Layer", "Activation", "Loss", "Optimizer", "LRScheduler", "AdamOptimizer")
KIND_OF_BASE = {
    "Module": "module",
    "Layer": "layer",
    "Activation": "activation",
    "Loss": "loss",
    "Optimizer": "optimizer",
    "AdamOptimizer": "optimizer",
    "LRScheduler": "scheduler",
}
BATCHER_BASES = ("IBatcher", "ISequenceBatcher")
MAP = "cyxwiz-engine/tests/computation_truth/oracle_coverage.json"
INVENTORY = "cyxwiz-engine/tests/computation_truth/coverage_inventory.json"
FIXTURES = "cyxwiz-engine/tests/computation_truth/fixtures"
ENGINE_CMAKE = "cyxwiz-engine/CMakeLists.txt"


def components(root: Path) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    backend = re.compile(r"class\s+CYXWIZ_API\s+(\w+)\s+(?:final\s+)?:\s*public\s+(\w+)\b")
    for header in sorted((root / "cyxwiz-backend/include/cyxwiz").rglob("*.h")):
        for name, base in backend.findall(header.read_text(encoding="utf-8", errors="replace")):
            if base in BACKEND_BASES:
                found[name] = {"kind": KIND_OF_BASE[base], "header": header.relative_to(root).as_posix()}
    batcher = re.compile(r"class\s+(\w+)\s+(?:final\s+)?:\s*public\s+(\w+)\b")
    for header in sorted((root / "cyxwiz-engine/src").rglob("*.h")):
        for name, base in batcher.findall(header.read_text(encoding="utf-8", errors="replace")):
            if base in BATCHER_BASES:
                found[name] = {"kind": "batcher", "header": header.relative_to(root).as_posix()}
    return found


def oracle_tests(root: Path) -> set[str]:
    text = (root / ENGINE_CMAKE).read_text(encoding="utf-8")
    match = re.search(r"set\(CYXWIZ_ORACLE_TESTS\s*\n(.*?)\n\)", text, re.S)
    if not match:
        raise SystemExit(f"{ENGINE_CMAKE}: no set(CYXWIZ_ORACLE_TESTS ...)")
    return set(match.group(1).split())


def has_case(fixture: object, path: str) -> bool:
    """A case is a key path into the fixture: "cases/adam_family_multistep_f32/adam"."""
    node = fixture
    for part in path.split("/"):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        elif isinstance(node, list) and any(isinstance(item, dict) and item.get("name") == part for item in node):
            node = next(item for item in node if isinstance(item, dict) and item.get("name") == part)
        else:
            return False
    return True


def check(root: Path) -> tuple[list[str], dict]:
    problems: list[str] = []
    code = components(root)
    rows = json.loads((root / MAP).read_text(encoding="utf-8"))["components"]
    tests = oracle_tests(root)
    fixtures: dict[str, object] = {}

    for name in sorted(set(code) - set(rows)):
        problems.append(f"{name} ({code[name]['kind']}, {code[name]['header']}) has no row in {MAP}: "
                        "name the oracle test, fixture and case that check it, or the gap and why")
    for name in sorted(set(rows) - set(code)):
        problems.append(f"{MAP} has a row for {name}, which is not a public component any more")

    inventory = []
    for name in sorted(code):
        row = rows.get(name)
        if row is None:
            continue
        evidence = row.get("evidence", [])
        gap = row.get("gap")
        if not evidence and not gap:
            problems.append(f"{name}: a row needs evidence or a gap")
        for item in evidence:
            test = item.get("test", "")
            if test not in tests:
                problems.append(f"{name}: {test!r} is not in CYXWIZ_ORACLE_TESTS")
            fixture_name = item.get("fixture")
            if fixture_name:
                path = root / FIXTURES / fixture_name
                if not path.exists():
                    problems.append(f"{name}: fixture {fixture_name} does not exist")
                    continue
                if fixture_name not in fixtures:
                    fixtures[fixture_name] = json.loads(path.read_text(encoding="utf-8"))
                for case in item.get("cases", []):
                    if not has_case(fixtures[fixture_name], case):
                        problems.append(f"{name}: {fixture_name} has no case {case}")
        inventory.append({
            "component": name,
            "kind": code[name]["kind"],
            "header": code[name]["header"],
            "status": "covered" if evidence and not gap else ("partial" if evidence else "gap"),
            "checks": row.get("checks", ""),
            "evidence": evidence,
            "gap": gap or "",
        })
    counts: dict[str, int] = {}
    for entry in inventory:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    generated = {
        "schema_version": 2,
        "generated_by": "scripts/checks/check_oracle_coverage.py --write (from the headers and " + MAP + ")",
        "counts": dict(sorted(counts.items())),
        "components": inventory,
    }
    return problems, generated


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1]).resolve()
    write = "--write" in sys.argv[2:]
    problems, generated = check(root)
    text = json.dumps(generated, indent=1, sort_keys=False) + "\n"
    inventory_path = root / INVENTORY
    if write:
        inventory_path.write_text(text, encoding="utf-8", newline="\n")
    elif not inventory_path.exists() or inventory_path.read_text(encoding="utf-8").replace("\r\n", "\n") != text:
        problems.append(f"{INVENTORY} is not what the code and {MAP} give: "
                        f"run scripts/checks/check_oracle_coverage.py . --write")
    for problem in problems:
        print(f"FAIL: {problem}")
    counts = ", ".join(f"{k} {v}" for k, v in generated["counts"].items())
    print(f"oracle coverage: {len(generated['components'])} components ({counts}); {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
