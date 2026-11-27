#!/usr/bin/env python3
"""The RL stability gate's protocol does not drift.

cyxwiz-rl-stability-gate trains each gate task the way its protocol says: the
graph and every node's settings, the step budget, the stop and the layers.
With CYXWIZ_GATE_PRINT_CONFIG=1 it prints that configuration and trains
nothing. This check prints it for every task and learner and compares it with
the stored copy, tests/data/rl_gate_configs.json: a change to the tool, to a
node's defaults or tuned settings, or to an SB3 reference file that would
change what the gate runs fails here and has to be written down by updating
the stored copy on purpose.

  python check_rl_gate_configs.py <cyxwiz-rl-stability-gate> <rl_gate_configs.json>
         [--plugin-dir <MuJoCo plugin folder>] [--update]

Without --plugin-dir only the built-in tasks are checked (the MuJoCo tasks
need the plugin). --update rewrites the stored copy from this run; it needs
--plugin-dir, so that the copy always holds every task. It trains nothing and
needs no GPU: the configuration is the compiled graph.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# (learner flag, task, needs the plugin): the gate's eight runs
RUNS = (
    ("ppo", "CartPole-v1", False),
    ("ppo", "InvertedPendulum-v5", True),
    ("ppo", "HalfCheetah-v5", True),
    ("sac", "InvertedPendulum-v5", True),
    ("sac", "HalfCheetah-v5", True),
    ("td3", "InvertedPendulum-v5", True),
    ("td3", "HalfCheetah-v5", True),
    ("dqn", "CartPole-v1", False),
)


def print_configuration(tool: Path, learner: str, task: str, plugin_dir: Path | None) -> dict:
    command = [str(tool)]
    if learner != "ppo":
        command.append("--" + learner)
    # The seed is part of the graph; the output file is not written in this mode
    command += [task, "1", str(Path(os.devnull))]
    if plugin_dir is not None:
        command.append(str(plugin_dir))
    environment = dict(os.environ, CYXWIZ_GATE_PRINT_CONFIG="1", CYXWIZ_TEST_ARRAYFIRE_BACKEND="cpu")
    for name in ("CYXWIZ_GATE_STEPS", "CYXWIZ_GATE_NO_PROVIDERS"):
        environment.pop(name, None)
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=environment, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{learner} {task}: the tool failed ({result.returncode})\n{result.stderr[-2000:]}")
    try:
        # Standard output is the configuration alone: a log line in it fails here
        return json.loads(result.stdout)
    except ValueError as error:
        raise SystemExit(f"{learner} {task}: standard output is not the configuration alone: {error}\n"
                         f"{result.stdout[:500]}") from error


def differences(stored, printed, path: str = "") -> list[str]:
    if isinstance(stored, dict) and isinstance(printed, dict):
        found: list[str] = []
        for key in sorted(set(stored) | set(printed)):
            where = f"{path}/{key}"
            if key not in printed:
                found.append(f"{where}: gone (was {json.dumps(stored[key])[:80]})")
            elif key not in stored:
                found.append(f"{where}: new ({json.dumps(printed[key])[:80]})")
            else:
                found += differences(stored[key], printed[key], where)
        return found
    if isinstance(stored, list) and isinstance(printed, list) and len(stored) == len(printed):
        found = []
        for index, (left, right) in enumerate(zip(stored, printed)):
            found += differences(left, right, f"{path}[{index}]")
        return found
    return [] if stored == printed else [f"{path}: {json.dumps(stored)[:80]} -> {json.dumps(printed)[:80]}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tool", type=Path)
    parser.add_argument("stored", type=Path)
    parser.add_argument("--plugin-dir", type=Path)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    if args.update and args.plugin_dir is None:
        parser.error("--update needs --plugin-dir: the stored copy holds every task")

    printed: dict[str, dict] = {}
    for learner, task, needs_plugin in RUNS:
        if needs_plugin and args.plugin_dir is None:
            continue
        printed[f"{learner} {task}"] = print_configuration(
            args.tool, learner, task, args.plugin_dir if needs_plugin else None)

    if args.update:
        document = {
            "note": "What cyxwiz-rl-stability-gate runs for each learner and task, as it prints it with "
                    "CYXWIZ_GATE_PRINT_CONFIG=1 (the MuJoCo tasks with the Engine's MuJoCo plugin). "
                    "check_rl_gate_configs.py compares the tool with this copy; rewrite it with --update "
                    "only when the gate's protocol is meant to change.",
            "configurations": printed,
        }
        args.stored.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {len(printed)} configurations to {args.stored}")
        return 0

    stored = json.loads(args.stored.read_text(encoding="utf-8"))["configurations"]
    failed = 0
    for name, configuration in printed.items():
        if name not in stored:
            print(f"FAIL {name}: not in the stored copy")
            failed += 1
            continue
        found = differences(stored[name], configuration)
        if found:
            failed += 1
            print(f"FAIL {name}: the gate would run another configuration")
            for line in found[:20]:
                print("   ", line)
        else:
            print(f"ok   {name}")
    skipped = len(RUNS) - len(printed)
    if skipped:
        print(f"{skipped} MuJoCo task(s) not checked: no --plugin-dir")
    print(f"{len(printed) - failed} of {len(printed)} configurations match the stored copy")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
