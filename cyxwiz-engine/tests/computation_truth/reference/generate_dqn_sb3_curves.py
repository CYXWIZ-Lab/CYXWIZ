#!/usr/bin/env python3
"""Reference learning curves for the Engine's DQN trainer (core/rl/dqn_trainer).

Stable-Baselines3 DQN on the Engine's own environments, with the settings the
stability gate gives the DQN node (sb3_offpolicy_curves.py trains and writes):
a row every 1,000 environment steps with the mean return of the last 100
episodes, the last gradient step's loss and the exploration rate.

    python generate_dqn_sb3_curves.py <folder of pycyxwiz and _cyxwiz_envs> <MuJoCo plugin folder>
        [--seeds 1 2 3] [--threads 1] [--output <file>]
    python generate_dqn_sb3_curves.py --merge <part.json> [<part.json> ...] [--output <file>]

DQN chooses among actions: CartPole-v1 (built in; the plugin folder is
loaded as for the other learners and not used).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sb3_offpolicy_curves import Main  # noqa: E402

# The Engine reads it to draw the reference on the RL Training Dashboard
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "resources" / "rl_reference" / "dqn_sb3_curves.json"
# One row per report: these SB3 logger values, in this order
COLUMNS = ["time/total_timesteps", "rollout/ep_rew_mean", "train/loss", "rollout/exploration_rate"]

# The stability gate's DQN task: RL Zoo's dqn.yml CartPole-v1 (the DQN node's Tuned for), all 50,000 steps;
# "net_arch" is the Q network's layers (policy_kwargs)
TASKS = [
    {"id": "CartPole-v1", "copies": 1, "target": 475.0, "total_timesteps": 50000,
     "dqn": {"learning_rate": 2.3e-3, "batch_size": 64, "buffer_size": 100000, "learning_starts": 1000,
             "gamma": 0.99, "target_update_interval": 10, "train_freq": 256, "gradient_steps": 128,
             "exploration_fraction": 0.16, "exploration_final_eps": 0.04, "net_arch": [256, 256]}},
]


def settings(task, envs):
    """DQN's keyword arguments: the task's settings, its layers as policy_kwargs."""
    kwargs = {k: v for k, v in task["dqn"].items() if k != "net_arch"}
    kwargs["policy_kwargs"] = {"net_arch": list(task["dqn"]["net_arch"])}
    return kwargs


if __name__ == "__main__":
    Main("DQN", TASKS, COLUMNS, "dqn", DEFAULT_OUTPUT,
         "cyxwiz-engine/tests/computation_truth/reference/generate_dqn_sb3_curves.py", settings)
