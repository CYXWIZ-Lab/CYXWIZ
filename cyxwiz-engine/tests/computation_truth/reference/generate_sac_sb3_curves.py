#!/usr/bin/env python3
"""Reference learning curves for the Engine's SAC trainer (core/rl/sac_trainer).

Trains Stable-Baselines3 SAC on the Engine's own environments through
pycyxwiz.rl (so the physics is the same MuJoCo build), with the settings the
stability gate gives the SAC node, for a few seeds, and writes a row every
1,000 environment steps (the SAC node's report interval): the mean return of
the last 100 episodes and the last gradient step's losses and entropy weight.

    python generate_sac_sb3_curves.py <folder of pycyxwiz and _cyxwiz_envs> <MuJoCo plugin folder>
        [--seeds 1 2 3] [--tasks HalfCheetah-v5] [--threads 1] [--output <file>]
    python generate_sac_sb3_curves.py --merge <part.json> [<part.json> ...] [--output <file>]

The output keeps the tasks and seeds it already holds that this run did not
train, so one task or one seed can be run on its own (seeds as parallel
processes, each with its own --output, then --merge into the reference).

SAC is continuous control only: CartPole (discrete) has no SAC reference.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sb3_curve_files import Merge, Read, Short, WriteReport  # noqa: E402

# The Engine reads it to draw the reference on the RL Training Dashboard
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "resources" / "rl_reference" / "sac_sb3_curves.json"
# One row per report: these SB3 logger values, in this order
COLUMNS = ["time/total_timesteps", "rollout/ep_rew_mean", "train/critic_loss", "train/actor_loss",
           "train/ent_coef"]
REPORT_EVERY = 1000

# The stability gate's SAC tasks: the settings, the copy count, the target and the step budget
TASKS = [
    # No RL Zoo SAC entry: SB3's SAC defaults (the SAC node's). The target is the mean of the last
    # 100 episodes, so a run that balances needs ~100,000 steps of 1,000-step episodes to show it.
    {"id": "InvertedPendulum-v5", "copies": 1, "target": 950.0, "total_timesteps": 150000, "sac": {}},
    # RL Zoo's HalfCheetah (sac.yml): learning_starts 10000, SB3 defaults otherwise. No target return in
    # the zoo: the gate compares the final returns.
    {"id": "HalfCheetah-v5", "copies": 1, "target": None, "total_timesteps": 1000000,
     "sac": {"learning_starts": 10000}},
]


def run(task, seed, rl):
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env

    rows = []
    reached = None
    envs = make_vec_env(lambda: rl.Env(task["id"]), n_envs=task["copies"], seed=seed)
    model = SAC("MlpPolicy", envs, seed=seed, device="cpu", verbose=0, **task["sac"])
    start = time.time()

    class Record(BaseCallback):
        def _on_step(self):
            nonlocal reached
            steps = int(self.model.num_timesteps)
            buffer = self.model.ep_info_buffer
            mean = sum(e["r"] for e in buffer) / len(buffer) if len(buffer) else float("nan")
            if reached is None and task["target"] is not None and len(buffer) >= 100 and mean >= task["target"]:
                reached = steps
            if steps % REPORT_EVERY != 0:
                return True
            # SAC.train records into the logger each gradient step; the values stand until the next dump
            logged = self.model.logger.name_to_value
            rows.append([steps, Short(mean)] +
                        [Short(logged.get(key, float("nan"))) for key in COLUMNS[2:]])
            if steps % 50000 == 0:  # progress of a long run
                print(f"{task['id']} seed {seed}: {steps} steps, mean return {rows[-1][1]} "
                      f"({time.time() - start:.0f} s)", flush=True)
            return True

    model.learn(total_timesteps=task["total_timesteps"], callback=Record())
    envs.close()
    return {"seed": seed, "reached_target_at": reached, "seconds": round(time.time() - start, 1), "updates": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?")
    parser.add_argument("plugin", nargs="?")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--tasks", nargs="+", default=[t["id"] for t in TASKS])
    # SB3 SAC's 256-wide networks with a batch of 256 ran fastest on one thread here (42 steps a second against
    # 23 on two and 29 on four), so parallel seeds are separate processes
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads of this process")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--merge", type=Path, nargs="+", help="part files to merge into --output")
    args = parser.parse_args()
    order = [t["id"] for t in TASKS]
    if args.merge:
        report = Read(args.output)
        for part in args.merge:
            report = Merge(report, Read(part), order)
        WriteReport(args.output, report)
        print(f"merged {len(args.merge)} part(s) into {args.output}")
        return
    if not args.folder or not args.plugin:
        parser.error("the folder of pycyxwiz and the MuJoCo plugin folder are needed to train")

    folder = os.path.abspath(args.folder)
    for d in (folder, r"C:\Program Files\ArrayFire\v3\lib"):
        if os.path.isdir(d) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(d)
    os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, folder)

    import gymnasium
    import mujoco
    import stable_baselines3
    import torch
    from pycyxwiz import rl

    torch.set_num_threads(args.threads)
    rl.load_plugin(os.path.abspath(args.plugin))
    results = []
    for task in TASKS:
        if task["id"] not in args.tasks:
            continue
        runs = []
        for seed in args.seeds:
            outcome = run(task, seed, rl)
            print(f"{task['id']} seed {seed}: target {task['target']} reached at "
                  f"{outcome['reached_target_at']} steps ({outcome['seconds']} s), "
                  f"final return {outcome['updates'][-1][1]:.1f}", flush=True)
            runs.append(outcome)
        results.append({key: task[key] for key in ("id", "copies", "target", "total_timesteps", "sac")} |
                       {"runs": runs})

    report = {
        "generator": "cyxwiz-engine/tests/computation_truth/reference/generate_sac_sb3_curves.py",
        "versions": {"stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__,
                     "gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__},
        "columns": COLUMNS,
        "report_every": REPORT_EVERY,
        "tasks": results,
    }
    WriteReport(args.output, Merge(Read(args.output), report, order))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
