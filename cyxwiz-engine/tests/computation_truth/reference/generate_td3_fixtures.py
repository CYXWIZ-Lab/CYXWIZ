#!/usr/bin/env python3
"""Generate PyTorch fixtures for the backend's TD3 math (cyxwiz/rl/td3.h).

The pieces are Stable-Baselines3 2.7's TD3.train, line for line, with the
target smoothing noise passed in (standard normal, scaled by
target_policy_noise as normal_(0, target_policy_noise) draws it) instead of
drawn; exploration is OffPolicyAlgorithm._sample_action's clip of the scaled
action plus NormalActionNoise. Autograd gives the gradients the backend
writes by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import stable_baselines3

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "td3_pytorch.json"


def tensor_fixture(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def actions_case(name: str, seed: int, rows: int, act: int, target_policy_noise: float, target_noise_clip: float,
                 noise_std: float) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    # The actor's tanh outputs, some near and at the bounds
    actions = torch.tanh(torch.randn(rows, act, generator=generator) * 1.5)
    actions[0, 0] = 1.0
    actions[1, act - 1] = -1.0
    actions[2, 0] = 0.98
    noise = torch.randn(rows, act, generator=generator)
    noise[3, 0] = 6.0    # past the clip either way
    noise[4, act - 1] = -6.0
    # TD3.train: noise.normal_(0, target_policy_noise).clamp(-clip, clip); (actor_target(s') + noise).clamp(-1, 1)
    smoothing = (noise * target_policy_noise).clamp(-target_noise_clip, target_noise_clip)
    target_actions = (actions + smoothing).clamp(-1, 1)
    # _sample_action: np.clip(scaled_action + action_noise(), -1, 1), NormalActionNoise(0, sigma)
    explore = (actions + noise_std * noise).clamp(-1, 1)
    return {
        "name": name, "target_policy_noise": target_policy_noise, "target_noise_clip": target_noise_clip,
        "noise_std": noise_std,
        "inputs": {"actions": tensor_fixture(actions), "noise": tensor_fixture(noise)},
        "expected": {"target_actions": tensor_fixture(target_actions), "explore": tensor_fixture(explore)},
    }


def losses_case(name: str, seed: int, rows: int, gamma: float, tie: bool) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    next_q1 = torch.randn(rows, 1, generator=generator) * 5.0
    next_q2 = torch.randn(rows, 1, generator=generator) * 5.0
    if tie:
        next_q2[:3] = next_q1[:3]
    rewards = torch.randn(rows, 1, generator=generator)
    dones = (torch.rand(rows, 1, generator=generator) < 0.3).float()
    q1 = (torch.randn(rows, 1, generator=generator) * 5.0).requires_grad_()
    q2 = (torch.randn(rows, 1, generator=generator) * 5.0).requires_grad_()
    q1_pi = (torch.randn(rows, 1, generator=generator) * 5.0).requires_grad_()

    # TD3.train, line for line
    with torch.no_grad():
        next_q_values = torch.cat([next_q1, next_q2], dim=1)
        next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
        target_q_values = rewards + (1 - dones) * gamma * next_q_values
    critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in (q1, q2))
    critic_loss.backward()
    actor_loss = -q1_pi.mean()
    actor_loss.backward()

    return {
        "name": name, "gamma": gamma,
        "inputs": {"next_q1": tensor_fixture(next_q1), "next_q2": tensor_fixture(next_q2),
                   "rewards": tensor_fixture(rewards), "dones": tensor_fixture(dones),
                   "q1": tensor_fixture(q1), "q2": tensor_fixture(q2), "q1_pi": tensor_fixture(q1_pi)},
        "expected": {"target": tensor_fixture(target_q_values),
                     "critic_loss": critic_loss.item(), "grad_q1": tensor_fixture(q1.grad),
                     "grad_q2": tensor_fixture(q2.grad),
                     "actor_loss": actor_loss.item(), "mean_q": q1_pi.mean().item(),
                     "grad_q1_pi": tensor_fixture(q1_pi.grad)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    output = parser.parse_args().output
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "generator": "cyxwiz-engine/tests/computation_truth/reference/generate_td3_fixtures.py",
        "torch_version": torch.__version__,
        "reference": f"stable_baselines3 {stable_baselines3.__version__} TD3.train, "
                     "OffPolicyAlgorithm._sample_action with NormalActionNoise",
        "actions": [actions_case("actions_6", 800, 8, 6, 0.2, 0.5, 0.1),
                    actions_case("actions_1_wide", 801, 6, 1, 0.4, 0.3, 0.5)],
        "losses": [losses_case("losses", 900, 16, 0.99, False), losses_case("losses_ties", 901, 8, 0.98, True)],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
