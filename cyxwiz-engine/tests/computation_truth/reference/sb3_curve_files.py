"""The reference-curve files of generate_ppo_sb3_curves.py, generate_sac_sb3_curves.py and
generate_td3_sb3_curves.py: read, merge one
run's tasks and seeds over a file, and write it with one row per line (resources/rl_reference/*.json)."""

from __future__ import annotations

import json
import math
from pathlib import Path


def Short(value):
    """Five significant digits; None (JSON null) for a value not logged yet (NaN)."""
    value = float(value)
    return float(f"{value:.5g}") if math.isfinite(value) else None


def Read(path):
    """A report written by WriteReport, or None when there is none."""
    if not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def Merge(old, new, order):
    """`new` over `old`: its tasks replace the same seeds of the same task, the rest stays; tasks sorted by
    `order` (task ids)."""
    if old is None:
        return new
    tasks = {t["id"]: t for t in old["tasks"]}
    for task in new["tasks"]:
        if task["id"] in tasks:
            kept = [r for r in tasks[task["id"]]["runs"] if r["seed"] not in {n["seed"] for n in task["runs"]}]
            task = dict(task, runs=sorted(kept + task["runs"], key=lambda r: r["seed"]))
        tasks[task["id"]] = task
    return dict(new, tasks=sorted(tasks.values(), key=lambda t: order.index(t["id"]) if t["id"] in order else 99))


def WriteReport(path, report):
    """JSON with one row per line; a value not logged yet is null (strict JSON, as the Engine reads it)."""
    rows = {}
    for task in report["tasks"]:
        for run in task["runs"]:
            token = f"@rows{len(rows)}@"
            rows[token] = [[None if isinstance(v, float) and not math.isfinite(v) else v for v in r]
                           for r in run["updates"]]
            run["updates"] = token
    text = json.dumps(report, indent=1, allow_nan=False)
    for token, updates in rows.items():
        text = text.replace(json.dumps(token),
                            "[\n" + ",\n".join(json.dumps(r, allow_nan=False) for r in updates) + "]")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
