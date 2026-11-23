#!/usr/bin/env python3
"""Reference learning curves for the Engine's TD3 trainer (core/rl/td3_trainer).

Stable-Baselines3 TD3 on the Engine's own environments, with the settings the
stability gate gives the TD3 node (sb3_offpolicy_curves.py trains and writes):
a row every 1,000 environment steps with the mean return of the last 100
episodes and the last gradient step's losses (the actor's on its last
policy update).

    python generate_td3_sb3_curves.py <folder of pycyxwiz and _cyxwiz_envs> <MuJoCo plugin folder>
        [--seeds 1 2 3] [--tasks HalfCheetah-v5] [--threads 1] [--output <file>]
    python generate_td3_sb3_curves.py --merge <part.json> [<part.json> ...] [--output <file>]

TD3 is continuous control only. The node explores with Gaussian noise of std
0.1 (RL Zoo's; SB3's own default is none), so every task here does too.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sb3_offpolicy_curves import Main  # noqa: E402

# The Engine reads it to draw the reference on the RL Training Dashboard
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "resources" / "rl_reference" / "td3_sb3_curves.json"
# One row per report: these SB3 logger values, in this order
COLUMNS = ["time/total_timesteps", "rollout/ep_rew_mean", "train/critic_loss", "train/actor_loss"]

# The stability gate's TD3 tasks: the settings, the copy count, the target and the step budget.
# "noise_std" is the node's Gaussian exploration (NormalActionNoise), made per run below.
TASKS = [
    # No RL Zoo TD3 entry for it: the TD3 node's defaults (SB3's, RL Zoo's noise). As SAC's.
    {"id": "InvertedPendulum-v5", "copies": 1, "target": 950.0, "total_timesteps": 150000,
     "td3": {"noise_std": 0.1}},
    # RL Zoo's HalfCheetah (td3.yml mujoco defaults): learning_starts 10000, noise normal 0.1, lr 1e-3,
    # batch 256, net_arch [400, 300] (SB3 TD3's default). No target: the gate compares the final returns.
    {"id": "HalfCheetah-v5", "copies": 1, "target": None, "total_timesteps": 1000000,
     "td3": {"learning_starts": 10000, "noise_std": 0.1}},
]


def settings(task, envs):
    """TD3's keyword arguments: the task's settings, the node's Gaussian noise as a NormalActionNoise of
    the task's actions."""
    import numpy as np
    from stable_baselines3.common.noise import NormalActionNoise

    kwargs = {k: v for k, v in task["td3"].items() if k != "noise_std"}
    std = task["td3"].get("noise_std")
    if std is not None:
        actions = envs.action_space.shape[-1]
        kwargs["action_noise"] = NormalActionNoise(mean=np.zeros(actions), sigma=std * np.ones(actions))
    return kwargs


if __name__ == "__main__":
    Main("TD3", TASKS, COLUMNS, "td3", DEFAULT_OUTPUT,
         "cyxwiz-engine/tests/computation_truth/reference/generate_td3_sb3_curves.py", settings)
