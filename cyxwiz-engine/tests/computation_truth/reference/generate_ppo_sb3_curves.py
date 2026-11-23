#!/usr/bin/env python3
"""Reference learning curves for the Engine's PPO trainer (core/rl/ppo_trainer).

Trains Stable-Baselines3 PPO on the Engine's own environments through
pycyxwiz.rl (so the physics is the same MuJoCo build and the same CartPole),
with the settings and copy counts test_ppo_trainer uses, for a few seeds, and
writes every update's return, losses and explained variance:

    python generate_ppo_sb3_curves.py <folder of pycyxwiz and _cyxwiz_envs> <MuJoCo plugin folder>
        [--seeds 1 2 3] [--tasks HalfCheetah-v5] [--output <file>]
    python generate_ppo_sb3_curves.py --merge <part.json> [<part.json> ...] [--output <file>]

The output keeps the tasks and seeds it already holds that this run did not
train, so one task or one seed can be run on its own (seeds as parallel
processes, each with its own --output, then --merge into the reference).

Both initialise the layers orthogonally as SB3's MlpPolicy does (gain sqrt(2),
0.01 for the policy output, 1 for the value); RL Zoo's HalfCheetah turns that
off (ortho_init False), the reference keeps it on to match the Engine, which
always uses it.
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


def Read(path):
    """A report written by WriteReport, or None when there is none."""
    if not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def Merge(old, new):
    """`new` over `old`: its tasks replace the same seeds of the same task, the rest stays."""
    if old is None:
        return new
    tasks = {t["id"]: t for t in old["tasks"]}
    for task in new["tasks"]:
        if task["id"] in tasks:
            kept = [r for r in tasks[task["id"]]["runs"] if r["seed"] not in {n["seed"] for n in task["runs"]}]
            task = dict(task, runs=sorted(kept + task["runs"], key=lambda r: r["seed"]))
        tasks[task["id"]] = task
    order = [t["id"] for t in TASKS]
    merged = dict(new, tasks=sorted(tasks.values(), key=lambda t: order.index(t["id"]) if t["id"] in order else 99))
    return merged


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
    # RL Zoo's HalfCheetah (ppo.yml): VecNormalize, two layers of 256 with ReLU, log std from -2.
    # No target return in the zoo: the gate compares the final returns (TOFIX195 stability gate).
    {"id": "HalfCheetah-v5", "copies": 1, "target": 4800.0, "total_timesteps": 1000000,
     "ppo": {"n_steps": 512, "batch_size": 64, "n_epochs": 20, "learning_rate": 2.0633e-05, "gamma": 0.98,
             "gae_lambda": 0.92, "clip_range": 0.1, "ent_coef": 0.000401762, "vf_coef": 0.58096,
             "max_grad_norm": 0.8},
     "policy": {"net_arch": 256, "activation": "relu", "log_std_init": -2.0},
     "normalize": True},
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
            if len(updates) % 50 == 0:  # progress of a long run
                print(f"{task['id']} seed {seed}: {updates[-1][0]} steps, mean return {updates[-1][1]}", flush=True)

    envs = make_vec_env(lambda: rl.Env(task["id"]), n_envs=task["copies"], seed=seed)
    if task.get("normalize"):
        from stable_baselines3.common.vec_env import VecNormalize
        envs = VecNormalize(envs, norm_obs=True, norm_reward=True, gamma=task["ppo"]["gamma"])
    policy_kwargs = {}
    if "policy" in task:
        import torch
        p = task["policy"]
        policy_kwargs = {"net_arch": dict(pi=[p["net_arch"]] * 2, vf=[p["net_arch"]] * 2),
                         "activation_fn": torch.nn.ReLU if p["activation"] == "relu" else torch.nn.Tanh,
                         "log_std_init": p["log_std_init"]}
    model = PPO("MlpPolicy", envs, seed=seed, device="cpu", verbose=0, policy_kwargs=policy_kwargs, **task["ppo"])
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
    parser.add_argument("folder", nargs="?")
    parser.add_argument("plugin", nargs="?")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--tasks", nargs="+", default=[t["id"] for t in TASKS])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--merge", type=Path, nargs="+", help="part files to merge into --output")
    args = parser.parse_args()
    if args.merge:
        report = Read(args.output)
        for part in args.merge:
            report = Merge(report, Read(part))
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
                       {key: task[key] for key in ("policy", "normalize") if key in task} | {"runs": runs})

    report = {
        "generator": "cyxwiz-engine/tests/computation_truth/reference/generate_ppo_sb3_curves.py",
        "versions": {"stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__,
                     "gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__},
        "columns": COLUMNS,
        "tasks": results,
    }
    WriteReport(args.output, Merge(Read(args.output), report))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
