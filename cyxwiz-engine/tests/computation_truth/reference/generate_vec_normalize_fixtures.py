#!/usr/bin/env python3
"""Generate Stable-Baselines3 fixtures for the Engine's VecNormalize (core/rl/vec_normalize).

Replays a fixed stream of observations, rewards and episode ends from three
copies through SB3 2.7's own VecNormalize (observations and rewards on,
clips 10 / 1 so both clips are hit) and records what it hands the trainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "vec_normalize_sb3.json"
COPIES, WIDTH, STEPS = 3, 4, 40
GAMMA, CLIP_OBS, CLIP_REWARD = 0.97, 10.0, 1.0


class Replay(VecEnv):
    """A VecEnv that returns a recorded stream."""

    def __init__(self, first, steps):
        space = gym.spaces.Box(-np.inf, np.inf, (WIDTH,), np.float32)
        super().__init__(COPIES, space, gym.spaces.Discrete(2))
        self.first, self.steps, self.t = first, steps, 0

    def reset(self):
        return self.first.copy()

    def step_async(self, actions):
        pass

    def step_wait(self):
        s = self.steps[self.t]
        self.t += 1
        infos = [{"terminal_observation": s["final"][i].copy()} if s["done"][i] else {} for i in range(COPIES)]
        return s["obs"].copy(), s["rewards"].copy(), s["done"].copy(), infos

    def close(self): pass
    def get_attr(self, *a, **k): return [None] * COPIES
    def set_attr(self, *a, **k): pass
    def env_method(self, *a, **k): return [None] * COPIES
    def env_is_wrapped(self, *a, **k): return [False] * COPIES
    def seed(self, seed=None): return [None] * COPIES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    output = parser.parse_args().output
    rng = np.random.default_rng(11)
    # observations of different scales and offsets; one feature with a spike past the clip
    scale = np.array([0.1, 1.0, 5.0, 50.0], np.float32)
    offset = np.array([0.0, -2.0, 3.0, 100.0], np.float32)
    draw = lambda: (rng.standard_normal((COPIES, WIDTH)) * scale + offset).astype(np.float32)
    first = draw()
    steps = []
    for t in range(STEPS):
        obs = draw()
        if t == 38:
            obs[1, 0] = 40.0  # far outside: clipped to 10
        done = rng.random(COPIES) < 0.12
        steps.append({"obs": obs, "rewards": (rng.standard_normal(COPIES) * 3.0 + 1.0).astype(np.float32),
                      "done": done, "final": draw()})

    env = VecNormalize(Replay(first, steps), norm_obs=True, norm_reward=True, clip_obs=CLIP_OBS,
                       clip_reward=CLIP_REWARD, gamma=GAMMA)
    expected = {"reset": env.reset().astype(np.float32).tolist(), "steps": []}
    for _ in range(STEPS):
        env.step_async(np.zeros(COPIES))
        obs, rewards, dones, infos = env.step_wait()
        expected["steps"].append({
            "obs": obs.astype(np.float32).tolist(), "rewards": rewards.astype(np.float32).tolist(),
            "final": [infos[i]["terminal_observation"].astype(np.float32).tolist() if dones[i] else None
                      for i in range(COPIES)]})
    fixture = {
        "generator": "cyxwiz-engine/tests/computation_truth/reference/generate_vec_normalize_fixtures.py",
        "stable_baselines3": stable_baselines3.__version__,
        "copies": COPIES, "width": WIDTH, "gamma": GAMMA, "clip_observations": CLIP_OBS, "clip_rewards": CLIP_REWARD,
        "inputs": {"reset": first.tolist(), "steps": [{"obs": s["obs"].tolist(), "rewards": s["rewards"].tolist(),
                                                       "done": s["done"].astype(int).tolist(),
                                                       "final": s["final"].tolist()} for s in steps]},
        "expected": expected,
        "final_stats": {"obs_mean": env.obs_rms.mean.tolist(), "obs_var": env.obs_rms.var.tolist(),
                        "obs_count": float(env.obs_rms.count), "ret_var": float(env.ret_rms.var)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
