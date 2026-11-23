#!/usr/bin/env python3
"""Four gradient steps and two target updates of a real Stable-Baselines3 DQN, for DqnLearner (TOFIX216 DQN V3).

A tiny DQN (3 observations, 3 choices, one hidden layer of 8 with ReLU,
max_grad_norm 0.5 so the clip acts, tau 0.5 so the target update is a true
average) runs DQN.train(gradient_steps=2) twice with its replay buffer's
sample pinned to recorded batches, and after each train call one _on_step
with target_update_interval 1, which moves the target network. The fixture
keeps the starting weights, the batches, and SB3's weights, target weights
and each train call's logged loss after, so test_dqn_trainer can replay the
same steps with DqnLearner.
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
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.type_aliases import ReplayBufferSamples

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "dqn_step_sb3.json"
OBS, CHOICES, HIDDEN, BATCH, CALLS, STEPS_PER_CALL = 3, 3, 8, 4, 2, 2
LEARNING_RATE, GAMMA, TAU, MAX_GRAD_NORM = 1e-2, 0.99, 0.5, 0.5


class Tiny(gym.Env):
    observation_space = gym.spaces.Box(-10.0, 10.0, (OBS,), np.float32)
    action_space = gym.spaces.Discrete(CHOICES)

    def reset(self, *, seed=None, options=None):
        return np.zeros(OBS, np.float32), {}

    def step(self, action):
        return np.zeros(OBS, np.float32), 0.0, False, False, {}


def tensor_fixture(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous().float()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def weights(model: DQN) -> dict[str, Any]:
    """SB3's parameters under the Engine's names: q / target_q layer0 / layer2."""
    out = {}
    for prefix, net in (("q", model.q_net), ("target_q", model.q_net_target)):
        p = dict(net.named_parameters())
        out[f"{prefix}.layer0.weight"] = p["q_net.0.weight"]
        out[f"{prefix}.layer0.bias"] = p["q_net.0.bias"]
        out[f"{prefix}.layer2.weight"] = p["q_net.2.weight"]
        out[f"{prefix}.layer2.bias"] = p["q_net.2.bias"]
    return {k: tensor_fixture(v) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    output = parser.parse_args().output

    model = DQN("MlpPolicy", Tiny(), learning_rate=LEARNING_RATE, buffer_size=100, learning_starts=0,
                batch_size=BATCH, tau=TAU, gamma=GAMMA, target_update_interval=1, max_grad_norm=MAX_GRAD_NORM,
                policy_kwargs=dict(net_arch=[HIDDEN]), device="cpu", seed=3)
    model.set_logger(configure(None, []))
    before = weights(model)

    generator = torch.Generator().manual_seed(920)
    batches = []
    for _ in range(CALLS * STEPS_PER_CALL):
        batches.append(ReplayBufferSamples(
            observations=torch.randn(BATCH, OBS, generator=generator) * 2.0,
            actions=torch.randint(0, CHOICES, (BATCH, 1), generator=generator),
            next_observations=torch.randn(BATCH, OBS, generator=generator) * 2.0,
            dones=(torch.rand(BATCH, 1, generator=generator) < 0.3).float(),
            rewards=torch.randn(BATCH, 1, generator=generator) * 3.0))

    pending = list(batches)
    model.replay_buffer.sample = lambda batch_size, env=None: pending.pop(0)
    losses = []
    for _ in range(CALLS):
        model.train(gradient_steps=STEPS_PER_CALL, batch_size=BATCH)
        losses.append(float(model.logger.name_to_value["train/loss"]))
        model._on_step()  # _n_calls 1, 2: target_update_interval 1 moves the target each time
    assert not pending, "DQN.train used every pinned batch"

    fixture = {
        "schema_version": SCHEMA_VERSION,
        "generator": "cyxwiz-engine/tests/computation_truth/reference/generate_dqn_step_fixture.py",
        "torch_version": torch.__version__,
        "reference": f"stable_baselines3 {stable_baselines3.__version__} DQN.train and DQN._on_step",
        "settings": {"observations": OBS, "choices": CHOICES, "hidden": HIDDEN, "batch": BATCH, "calls": CALLS,
                     "steps_per_call": STEPS_PER_CALL, "learning_rate": LEARNING_RATE, "gamma": GAMMA, "tau": TAU,
                     "max_grad_norm": MAX_GRAD_NORM},
        "before": before,
        "batches": [{"observations": tensor_fixture(b.observations), "actions": tensor_fixture(b.actions.float()),
                     "next_observations": tensor_fixture(b.next_observations), "dones": tensor_fixture(b.dones),
                     "rewards": tensor_fixture(b.rewards)} for b in batches],
        "after": weights(model),
        "logged_loss": losses,  # each train call's mean loss (SB3 logs np.mean(losses))
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {output}: losses {losses}")


if __name__ == "__main__":
    main()
