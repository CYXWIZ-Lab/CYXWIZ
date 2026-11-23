#!/usr/bin/env python3
"""Reference learning curves for the Engine's SAC trainer (core/rl/sac_trainer).

Stable-Baselines3 SAC on the Engine's own environments, with the settings the
stability gate gives the SAC node (sb3_offpolicy_curves.py trains and writes):
a row every 1,000 environment steps with the mean return of the last 100
episodes and the last gradient step's losses and entropy weight.

    python generate_sac_sb3_curves.py <folder of pycyxwiz and _cyxwiz_envs> <MuJoCo plugin folder>
        [--seeds 1 2 3] [--tasks HalfCheetah-v5] [--threads 1] [--output <file>]
    python generate_sac_sb3_curves.py --merge <part.json> [<part.json> ...] [--output <file>]

SAC is continuous control only: CartPole (discrete) has no SAC reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sb3_offpolicy_curves import Main  # noqa: E402

# The Engine reads it to draw the reference on the RL Training Dashboard
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "resources" / "rl_reference" / "sac_sb3_curves.json"
# One row per report: these SB3 logger values, in this order
COLUMNS = ["time/total_timesteps", "rollout/ep_rew_mean", "train/critic_loss", "train/actor_loss",
           "train/ent_coef"]

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

if __name__ == "__main__":
    Main("SAC", TASKS, COLUMNS, "sac", DEFAULT_OUTPUT,
         "cyxwiz-engine/tests/computation_truth/reference/generate_sac_sb3_curves.py")
