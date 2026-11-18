#!/usr/bin/env python3
"""Three gradient steps of a real Stable-Baselines3 SAC, for SacLearner (TOFIX216 S3).

A tiny SAC (3 observations, 2 actions, one hidden layer of 8 with ReLU in the
actor and both critics, ent_coef "auto") runs SAC.train(gradient_steps=3)
with its replay buffer's sample and torch's Normal.rsample pinned to recorded
batches and noise. The fixture keeps the starting weights, the batches, the
noise, and SB3's weights, target weights, log entropy weight and logged
losses after, so test_sac_trainer can replay the same steps with SacLearner
and check the whole order (entropy weight, critics, actor on the updated
critics, Polyak) and every gradient through the Adam updates.
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
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.type_aliases import ReplayBufferSamples

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "sac_step_sb3.json"
OBS, ACT, HIDDEN, BATCH, STEPS = 3, 2, 8, 4, 3


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


def weights(model: SAC) -> dict[str, Any]:
    """SB3's parameters under the Engine's names: actor layer0 / layer2 (mean rows, then log std rows), q1 / q2."""
    a = dict(model.actor.named_parameters())
    out = {
        "actor.layer0.weight": a["latent_pi.0.weight"], "actor.layer0.bias": a["latent_pi.0.bias"],
        "actor.layer2.weight": torch.cat([a["mu.weight"], a["log_std.weight"]], 0),
        "actor.layer2.bias": torch.cat([a["mu.bias"], a["log_std.bias"]], 0),
    }
    for prefix, critic in (("", model.critic), ("target_", model.critic_target)):
        c = dict(critic.named_parameters())
        for q, name in ((0, "q1"), (1, "q2")):
            out[f"{prefix}{name}.layer0.weight"] = c[f"qf{q}.0.weight"]
            out[f"{prefix}{name}.layer0.bias"] = c[f"qf{q}.0.bias"]
            out[f"{prefix}{name}.layer2.weight"] = c[f"qf{q}.2.weight"]
            out[f"{prefix}{name}.layer2.bias"] = c[f"qf{q}.2.bias"]
    out["log_ent_coef"] = model.log_ent_coef
    return {k: tensor_fixture(v) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    output = parser.parse_args().output

    model = SAC("MlpPolicy", Tiny(), learning_rate=3e-4, buffer_size=100, learning_starts=0, batch_size=BATCH,
                tau=0.005, gamma=0.99, ent_coef="auto", target_entropy="auto", target_update_interval=1,
                policy_kwargs=dict(net_arch=[HIDDEN]), device="cpu", seed=3)
    model.set_logger(configure(None, []))
    before = weights(model)

    generator = torch.Generator().manual_seed(900)
    batches, noises = [], []
    for _ in range(STEPS):
        batches.append(ReplayBufferSamples(
            observations=torch.randn(BATCH, OBS, generator=generator),
            actions=torch.rand(BATCH, ACT, generator=generator) * 2.0 - 1.0,
            next_observations=torch.randn(BATCH, OBS, generator=generator),
            dones=(torch.rand(BATCH, 1, generator=generator) < 0.3).float(),
            rewards=torch.randn(BATCH, 1, generator=generator)))
        # SAC.train draws for the batch's observations first, then for the next ones
        noises.append((torch.randn(BATCH, ACT, generator=generator), torch.randn(BATCH, ACT, generator=generator)))

    queue = [n for pair in noises for n in pair]
    original_rsample = torch.distributions.Normal.rsample

    def pinned_rsample(self, sample_shape=torch.Size()):
        return self.loc + queue.pop(0) * self.scale

    torch.distributions.Normal.rsample = pinned_rsample
    pending = list(batches)
    model.replay_buffer.sample = lambda batch_size, env=None: pending.pop(0)
    try:
        model.train(gradient_steps=STEPS, batch_size=BATCH)
    finally:
        torch.distributions.Normal.rsample = original_rsample
    assert not queue and not pending, "SAC.train used every pinned draw and batch"
    logged = model.logger.name_to_value

    fixture = {
        "schema_version": SCHEMA_VERSION,
        "generator": "cyxwiz-engine/tests/computation_truth/reference/generate_sac_step_fixture.py",
        "torch_version": torch.__version__,
        "reference": f"stable_baselines3 {stable_baselines3.__version__} SAC.train",
        "settings": {"observations": OBS, "actions": ACT, "hidden": HIDDEN, "batch": BATCH, "steps": STEPS,
                     "learning_rate": 3e-4, "gamma": 0.99, "tau": 0.005, "target_entropy": float(model.target_entropy)},
        "before": before,
        "batches": [{"observations": tensor_fixture(b.observations), "actions": tensor_fixture(b.actions),
                     "next_observations": tensor_fixture(b.next_observations), "dones": tensor_fixture(b.dones),
                     "rewards": tensor_fixture(b.rewards)} for b in batches],
        "noise": [{"pi": tensor_fixture(pi), "next": tensor_fixture(nxt)} for pi, nxt in noises],
        "after": weights(model),
        "logged": {k.split("/")[-1]: float(v) for k, v in logged.items() if k.startswith("train/")},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {output}: logged {fixture['logged']}")


if __name__ == "__main__":
    main()
