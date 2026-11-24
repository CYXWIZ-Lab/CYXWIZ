"""Reference learning curves of a Stable-Baselines3 off-policy learner (SAC, TD3, DQN).

Trains it on the Engine's own environments through pycyxwiz.rl (so the
physics is the same MuJoCo build), with the settings the stability gate gives
the learner's node, for a few seeds, and writes a row every 1,000 environment
steps (the node's report interval): the mean return of the last 100 episodes
and the last gradient step's logged values. generate_sac_sb3_curves.py,
generate_td3_sb3_curves.py and generate_dqn_sb3_curves.py give the learner,
its tasks and its columns.

The output keeps the tasks and seeds it already holds that a run did not
train, so one task or one seed can be run on its own (seeds as parallel
processes, each with its own --output, then --merge into the reference).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sb3_curve_files import Merge, Read, Short, WriteReport  # noqa: E402

REPORT_EVERY = 1000


def run(learner, task, seed, rl, settings_key, make_kwargs, columns):
    import stable_baselines3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env

    rows = []
    reached = None
    envs = make_vec_env(lambda: rl.Env(task["id"]), n_envs=task["copies"], seed=seed)
    kwargs = make_kwargs(task, envs) if make_kwargs else dict(task[settings_key])
    model = getattr(stable_baselines3, learner)("MlpPolicy", envs, seed=seed, device="cpu", verbose=0, **kwargs)
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
            # train() records into the logger each gradient step; the values stand until the next dump
            logged = self.model.logger.name_to_value
            rows.append([steps, Short(mean)] + [Short(logged.get(key, float("nan"))) for key in columns[2:]])
            if steps % 50000 == 0:  # progress of a long run
                print(f"{task['id']} seed {seed}: {steps} steps, mean return {rows[-1][1]} "
                      f"({time.time() - start:.0f} s)", flush=True)
            return True

    model.learn(total_timesteps=task["total_timesteps"], callback=Record())
    envs.close()
    return {"seed": seed, "reached_target_at": reached, "seconds": round(time.time() - start, 1), "updates": rows}


def Main(learner, tasks, columns, settings_key, default_output, generator, make_kwargs=None):
    """The command line of a learner's generator: train, or --merge part files. make_kwargs(task, envs):
    the learner's keyword arguments (default: the task's settings as they are)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?")
    parser.add_argument("plugin", nargs="?")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--tasks", nargs="+", default=[t["id"] for t in tasks])
    # SB3's small networks with a batch of 256 ran fastest on one thread here (SAC: 42 steps a second against
    # 23 on two and 29 on four), so parallel seeds are separate processes
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads of this process")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--merge", type=Path, nargs="+", help="part files to merge into --output")
    args = parser.parse_args()
    order = [t["id"] for t in tasks]
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
    versions = {"stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__,
                "gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__}
    for task in tasks:
        if task["id"] not in args.tasks:
            continue
        for seed in args.seeds:
            outcome = run(learner, task, seed, rl, settings_key, make_kwargs, columns)
            final = outcome["updates"][-1][1] if outcome["updates"] else None
            print(f"{task['id']} seed {seed}: target {task['target']} reached at "
                  f"{outcome['reached_target_at']} steps ({outcome['seconds']} s), final return {final}", flush=True)
            # Written after every seed, so a long run that stops later keeps what it trained
            report = {
                "generator": generator,
                "versions": versions,
                "columns": columns,
                "report_every": REPORT_EVERY,
                "tasks": [{key: task[key] for key in ("id", "copies", "target", "total_timesteps", settings_key)} |
                          {"runs": [outcome]}],
            }
            WriteReport(args.output, Merge(Read(args.output), report, order))
            print(f"wrote {args.output}", flush=True)
