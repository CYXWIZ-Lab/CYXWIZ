#!/usr/bin/env python3
"""Reference learning curves for the Engine's PPO trainer (core/rl/ppo_trainer).

Trains Stable-Baselines3 PPO on the Engine's own environments through
pycyxwiz.rl (so the physics is the same MuJoCo build and the same CartPole),
with the settings and copy counts test_ppo_trainer uses, for a few seeds, and
writes every update's return, losses and explained variance:

    python generate_ppo_sb3_curves.py <folder of pycyxwiz and _cyxwiz_envs> <MuJoCo plugin folder>
        [--seeds 1 2 3] [--output fixtures/ppo_sb3_curves.json]

Differences that remain on purpose: SB3 initialises its layers orthogonally
(gain sqrt(2), 0.01 for the policy output, 1 for the value), the Engine with
its own layer init and the policy output scaled by 0.01.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# The Engine reads it to draw the reference on the RL Training Dashboard
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "resources" / "rl_reference" / "ppo_sb3_curves.json"
# One row per update: these SB3 logger values, in this order
COLUMNS = ["time/total_timesteps", "rollout/ep_rew_mean", "train/policy_gradient_loss", "train/value_loss",
           "train/approx_kl", "train/clip_fraction", "train/explained_variance"]


def Short(value):
    return float(f"{float(value):.5g}")


def WriteReport(path, report):
    """JSON with one update per line."""
    rows = {}
    for task in report["tasks"]:
        for run in task["runs"]:
            token = f"@rows{len(rows)}@"
            rows[token] = run["updates"]
            run["updates"] = token
    text = json.dumps(report, indent=1)
    for token, updates in rows.items():
        text = text.replace(json.dumps(token), "[\n" + ",\n".join(json.dumps(r) for r in updates) + "]")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")

# test_ppo_trainer's tasks: RL Zoo settings, the copy count, the target and the step budget
TASKS = [
    {"id": "CartPole-v1", "copies": 8, "target": 475.0, "total_timesteps": 200000,
     "ppo": {"n_steps": 32, "batch_size": 256, "n_epochs": 20, "learning_rate": 1e-3, "gamma": 0.98,
             "gae_lambda": 0.8, "ent_coef": 0.0}},
    {"id": "InvertedPendulum-v5", "copies": 4, "target": 950.0, "total_timesteps": 300000,
     "ppo": {"n_steps": 512, "batch_size": 64, "n_epochs": 10, "learning_rate": 3e-4, "gamma": 0.999,
             "gae_lambda": 0.9}},
]


def run(task, seed, rl):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.logger import KVWriter, Logger

    updates = []

    class Recorder(KVWriter):
        def write(self, key_values, key_excluded, step=0):
            if "train/explained_variance" not in key_values:
                return
            updates.append([int(key_values[COLUMNS[0]])] +
                           [Short(key_values.get(key, float("nan"))) for key in COLUMNS[1:]])

    envs = make_vec_env(lambda: rl.Env(task["id"]), n_envs=task["copies"], seed=seed)
    model = PPO("MlpPolicy", envs, seed=seed, device="cpu", verbose=0, **task["ppo"])
    model.set_logger(Logger(None, [Recorder()]))
    start = time.time()
    reached = None

    # first step count at which the last 100 episodes average the target
    class Watch(BaseCallback):
        def _on_step(self):
            return True

        def _on_rollout_start(self):
            nonlocal reached
            buffer = self.model.ep_info_buffer
            if reached is None and len(buffer) >= 100 and \
                    sum(e["r"] for e in buffer) / len(buffer) >= task["target"]:
                reached = int(self.model.num_timesteps)

    model.learn(total_timesteps=task["total_timesteps"], callback=Watch())
    envs.close()
    return {"seed": seed, "reached_target_at": reached, "seconds": round(time.time() - start, 1),
            "updates": updates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("plugin")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--tasks", nargs="+", default=[t["id"] for t in TASKS])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

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
                  f"final return {outcome['updates'][-1][1]:.1f}, "
                  f"last explained variance {outcome['updates'][-1][6]:.3f}", flush=True)
            runs.append(outcome)
        results.append({key: task[key] for key in ("id", "copies", "target", "total_timesteps", "ppo")} |
                       {"runs": runs})

    report = {
        "generator": "cyxwiz-engine/tests/computation_truth/reference/generate_ppo_sb3_curves.py",
        "versions": {"stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__,
                     "gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__},
        "columns": COLUMNS,
        "tasks": results,
    }
    WriteReport(args.output, report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
