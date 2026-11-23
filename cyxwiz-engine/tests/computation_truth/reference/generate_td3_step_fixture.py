#!/usr/bin/env python3
"""Four gradient steps of a real Stable-Baselines3 TD3, for Td3Learner (TOFIX216 TD3 U3).

A tiny TD3 (3 observations, 2 actions, one hidden layer of 8 with ReLU in the
actor and both critics, policy_delay 2) runs TD3.train(gradient_steps=4)
with its replay buffer's sample and the target smoothing draw (the
in-place normal_ on the batch's actions) pinned to recorded batches and
noise. Four steps: the critics update every step, the actor and the targets
on steps 2 and 4. The fixture keeps the starting weights, the batches, the
noise (standard normal; TD3.train scales it by target_policy_noise), and
SB3's weights, target weights and logged losses after, so test_td3_trainer
can replay the same steps with Td3Learner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import stable_baselines3
from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.type_aliases import ReplayBufferSamples

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "td3_step_sb3.json"
OBS, ACT, HIDDEN, BATCH, STEPS = 3, 2, 8, 4, 4
TARGET_POLICY_NOISE, TARGET_NOISE_CLIP = 0.2, 0.5


class Tiny(gym.Env):
    observation_space = gym.spaces.Box(-10.0, 10.0, (OBS,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (ACT,), np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(OBS, np.float32), {}

    def step(self, action):
        return np.zeros(OBS, np.float32), 0.0, False, False, {}


def tensor_fixture(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous().float()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def weights(model: TD3) -> dict[str, Any]:
    """SB3's parameters under the Engine's names: actor / target_actor layer0 / layer2, q1 / q2 and targets."""
    out = {}
    for prefix, actor in (("", model.actor), ("target_", model.actor_target)):
        a = dict(actor.named_parameters())
        out[f"{prefix}actor.layer0.weight"] = a["mu.0.weight"]
        out[f"{prefix}actor.layer0.bias"] = a["mu.0.bias"]
        out[f"{prefix}actor.layer2.weight"] = a["mu.2.weight"]
        out[f"{prefix}actor.layer2.bias"] = a["mu.2.bias"]
    for prefix, critic in (("", model.critic), ("target_", model.critic_target)):
        c = dict(critic.named_parameters())
        for q, name in ((0, "q1"), (1, "q2")):
            out[f"{prefix}{name}.layer0.weight"] = c[f"qf{q}.0.weight"]
            out[f"{prefix}{name}.layer0.bias"] = c[f"qf{q}.0.bias"]
            out[f"{prefix}{name}.layer2.weight"] = c[f"qf{q}.2.weight"]
            out[f"{prefix}{name}.layer2.bias"] = c[f"qf{q}.2.bias"]
    return {k: tensor_fixture(v) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    output = parser.parse_args().output

    model = TD3("MlpPolicy", Tiny(), learning_rate=1e-3, buffer_size=100, learning_starts=0, batch_size=BATCH,
                tau=0.005, gamma=0.99, policy_delay=2, target_policy_noise=TARGET_POLICY_NOISE,
                target_noise_clip=TARGET_NOISE_CLIP, policy_kwargs=dict(net_arch=[HIDDEN]), device="cpu", seed=3)
    model.set_logger(configure(None, []))
    before = weights(model)

    generator = torch.Generator().manual_seed(910)
    batches, noises = [], []
    for _ in range(STEPS):
        batches.append(ReplayBufferSamples(
            observations=torch.randn(BATCH, OBS, generator=generator),
            actions=torch.rand(BATCH, ACT, generator=generator) * 2.0 - 1.0,
            next_observations=torch.randn(BATCH, OBS, generator=generator),
            dones=(torch.rand(BATCH, 1, generator=generator) < 0.3).float(),
            rewards=torch.randn(BATCH, 1, generator=generator)))
        noises.append(torch.randn(BATCH, ACT, generator=generator) * 3.0)  # some past the clip

    queue = list(noises)
    original_normal = torch.Tensor.normal_

    def pinned_normal(self, mean=0.0, std=1.0, *, generator=None):
        return self.copy_(queue.pop(0) * std + mean)

    torch.Tensor.normal_ = pinned_normal
    pending = list(batches)
    model.replay_buffer.sample = lambda batch_size, env=None: pending.pop(0)
    try:
        model.train(gradient_steps=STEPS, batch_size=BATCH)
    finally:
        torch.Tensor.normal_ = original_normal
    assert not queue and not pending, "TD3.train used every pinned draw and batch"
    logged = model.logger.name_to_value

    fixture = {
        "schema_version": SCHEMA_VERSION,
        "generator": "cyxwiz-engine/tests/computation_truth/reference/generate_td3_step_fixture.py",
        "torch_version": torch.__version__,
        "reference": f"stable_baselines3 {stable_baselines3.__version__} TD3.train",
        "settings": {"observations": OBS, "actions": ACT, "hidden": HIDDEN, "batch": BATCH, "steps": STEPS,
                     "learning_rate": 1e-3, "gamma": 0.99, "tau": 0.005, "policy_delay": 2,
                     "target_policy_noise": TARGET_POLICY_NOISE, "target_noise_clip": TARGET_NOISE_CLIP},
        "before": before,
        "batches": [{"observations": tensor_fixture(b.observations), "actions": tensor_fixture(b.actions),
                     "next_observations": tensor_fixture(b.next_observations), "dones": tensor_fixture(b.dones),
                     "rewards": tensor_fixture(b.rewards)} for b in batches],
        "noise": [tensor_fixture(n) for n in noises],
        "after": weights(model),
        "logged": {k.split("/")[-1]: float(v) for k, v in logged.items() if k.startswith("train/")},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {output}: logged {fixture['logged']}")


if __name__ == "__main__":
    main()
