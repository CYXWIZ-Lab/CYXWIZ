#!/usr/bin/env python3
"""Generate PyTorch fixtures for the backend's PPO math (cyxwiz/rl/ppo.h).

The loss is Stable-Baselines3 2.7's PPO.train, line for line (normalised
advantages, clipped surrogate, optional clipped value, entropy bonus), on
torch.distributions, with autograd giving the gradients the backend writes
by hand. GAE is SB3's RolloutBuffer.compute_returns_and_advantage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "ppo_pytorch.json"


def tensor_fixture(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def sb3_loss(log_prob, entropy, values, old_log_prob, advantages, returns, old_values, settings):
    values = values.flatten()
    if settings["normalize_advantage"] and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    ratio = torch.exp(log_prob - old_log_prob)
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - settings["clip_range"], 1 + settings["clip_range"])
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
    if settings["clip_range_vf"] <= 0:
        values_pred = values
    else:
        values_pred = old_values + torch.clamp(values - old_values, -settings["clip_range_vf"], settings["clip_range_vf"])
    value_loss = F.mse_loss(returns, values_pred)
    entropy_loss = -torch.mean(entropy)
    loss = policy_loss + settings["ent_coef"] * entropy_loss + settings["vf_coef"] * value_loss
    with torch.no_grad():
        log_ratio = log_prob - old_log_prob
        approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
        clip_fraction = torch.mean((torch.abs(ratio - 1) > settings["clip_range"]).float())
    return loss, policy_loss, value_loss, entropy_loss, approx_kl, clip_fraction


SETTINGS = [
    {"name": "default", "clip_range": 0.2, "clip_range_vf": 0.0, "ent_coef": 0.0, "vf_coef": 0.5,
     "normalize_advantage": True},
    {"name": "clipped_value_entropy", "clip_range": 0.1, "clip_range_vf": 0.2, "ent_coef": 0.01, "vf_coef": 0.5,
     "normalize_advantage": True},
    {"name": "raw_advantages", "clip_range": 0.3, "clip_range_vf": 0.0, "ent_coef": 0.05, "vf_coef": 1.0,
     "normalize_advantage": False},
]


def common_batch(generator, rows, log_prob):
    # old log-probabilities around the current ones, so some ratios fall
    # outside the clip range on both sides and some sit at exactly 1
    shift = torch.randn(rows, generator=generator) * 0.25
    shift[0] = 0.0
    old_log_prob = (log_prob.detach() + shift).float()
    advantages = torch.randn(rows, generator=generator) * 2.0 + 0.3
    returns = torch.randn(rows, generator=generator)
    old_values = torch.randn(rows, generator=generator) * 0.5
    return old_log_prob, advantages, returns, old_values


def result_fixture(outputs, inputs_with_grad):
    loss, policy_loss, value_loss, entropy_loss, approx_kl, clip_fraction = outputs
    loss.backward()
    expected = {
        "loss": loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
        "entropy_loss": entropy_loss.item(), "approx_kl": approx_kl.item(), "clip_fraction": clip_fraction.item(),
    }
    for name, tensor in inputs_with_grad.items():
        expected["grad_" + name] = tensor_fixture(tensor.grad)
    return expected


def gaussian_cases() -> list[dict[str, Any]]:
    cases = []
    for index, settings in enumerate(SETTINGS):
        generator = torch.Generator().manual_seed(100 + index)
        rows, act = 8, 2
        mean = (torch.randn(rows, act, generator=generator) * 0.7).requires_grad_()
        log_std = (torch.randn(act, generator=generator) * 0.3 - 0.2).requires_grad_()
        values = torch.randn(rows, 1, generator=generator).requires_grad_()
        actions = mean.detach() + torch.randn(rows, act, generator=generator) * log_std.detach().exp()
        dist = torch.distributions.Normal(mean, log_std.exp())
        log_prob = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        old_log_prob, advantages, returns, old_values = common_batch(generator, rows, log_prob)
        inputs = {"mean": mean, "log_std": log_std, "values": values}
        case = {
            "name": "gaussian_" + settings["name"], "settings": settings,
            "inputs": {k: tensor_fixture(v) for k, v in inputs.items()} | {
                "actions": tensor_fixture(actions), "old_log_prob": tensor_fixture(old_log_prob),
                "advantages": tensor_fixture(advantages), "returns": tensor_fixture(returns),
                "old_values": tensor_fixture(old_values)},
            "log_prob": tensor_fixture(log_prob),
        }
        case["expected"] = result_fixture(
            sb3_loss(log_prob, entropy, values, old_log_prob, advantages, returns, old_values, settings), inputs)
        cases.append(case)
    return cases


def categorical_cases() -> list[dict[str, Any]]:
    cases = []
    for index, settings in enumerate(SETTINGS):
        generator = torch.Generator().manual_seed(200 + index)
        rows, classes = 8, 3
        logits = (torch.randn(rows, classes, generator=generator) * 1.5).requires_grad_()
        values = torch.randn(rows, 1, generator=generator).requires_grad_()
        actions = torch.randint(0, classes, (rows,), generator=generator)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        old_log_prob, advantages, returns, old_values = common_batch(generator, rows, log_prob)
        inputs = {"logits": logits, "values": values}
        case = {
            "name": "categorical_" + settings["name"], "settings": settings,
            "inputs": {k: tensor_fixture(v) for k, v in inputs.items()} | {
                "actions": tensor_fixture(actions.float()), "old_log_prob": tensor_fixture(old_log_prob),
                "advantages": tensor_fixture(advantages), "returns": tensor_fixture(returns),
                "old_values": tensor_fixture(old_values)},
            "log_prob": tensor_fixture(log_prob),
        }
        case["expected"] = result_fixture(
            sb3_loss(log_prob, entropy, values, old_log_prob, advantages, returns, old_values, settings), inputs)
        cases.append(case)
    return cases


def gae_case() -> dict[str, Any]:
    # SB3 RolloutBuffer.compute_returns_and_advantage; episode_starts[t + 1]
    # = dones[t] (an episode ended at step t)
    generator = torch.Generator().manual_seed(300)
    steps, copies, gamma, gae_lambda = 6, 3, 0.99, 0.95
    rewards = torch.randn(steps, copies, generator=generator, dtype=torch.float64)
    values = torch.randn(steps, copies, generator=generator, dtype=torch.float64)
    dones = torch.zeros(steps, copies, dtype=torch.float64)
    dones[1, 0] = dones[4, 0] = dones[2, 2] = dones[5, 1] = 1.0
    last_values = torch.randn(copies, generator=generator, dtype=torch.float64)
    advantages = torch.zeros(steps, copies, dtype=torch.float64)
    last_gae = torch.zeros(copies, dtype=torch.float64)
    for step in reversed(range(steps)):
        next_non_terminal = 1.0 - dones[step]
        next_values = last_values if step == steps - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[step] = last_gae
    return {
        "gamma": gamma, "gae_lambda": gae_lambda,
        "rewards": tensor_fixture(rewards.float()), "values": tensor_fixture(values.float()),
        "dones": tensor_fixture(dones.float()), "last_values": tensor_fixture(last_values.float()),
        "advantages": tensor_fixture(advantages.float()), "returns": tensor_fixture((advantages + values).float()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    output = parser.parse_args().output
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "generator": "cyxwiz-engine/tests/computation_truth/reference/generate_ppo_fixtures.py",
        "torch_version": torch.__version__,
        "reference": "stable_baselines3 2.7 PPO.train and RolloutBuffer.compute_returns_and_advantage",
        "gaussian": gaussian_cases(),
        "categorical": categorical_cases(),
        "gae": gae_case(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
