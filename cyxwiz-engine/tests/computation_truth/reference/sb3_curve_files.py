"""The reference-curve files of generate_ppo_sb3_curves.py and the off-policy generators (SAC, TD3, DQN):
read, merge one
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
    `order` (task ids). Refuses runs of another SB3 or another library version, or a task recorded under
    other settings: a reference mixing them would not be any one run's curve."""
    if old is None:
        return new
    if old.get("versions") != new.get("versions"):
        raise ValueError(f"cannot merge runs of {new.get('versions')} into a file of {old.get('versions')}")
    tasks = {t["id"]: t for t in old["tasks"]}
    for task in new["tasks"]:
        if task["id"] in tasks:
            settings = {k: v for k, v in task.items() if k != "runs"}
            recorded = {k: v for k, v in tasks[task["id"]].items() if k != "runs"}
            if settings != recorded:
                raise ValueError(f"{task['id']}: the new runs' settings {settings} differ from the recorded "
                                 f"{recorded}; record all its seeds again under the new settings")
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
