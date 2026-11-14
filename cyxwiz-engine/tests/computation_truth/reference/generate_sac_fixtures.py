#!/usr/bin/env python3
"""Generate PyTorch fixtures for the backend's SAC math (cyxwiz/rl/sac.h).

The actor's distribution is Stable-Baselines3 2.7's own
SquashedDiagGaussianDistribution (log std clamped to [-20, 2] as SAC's Actor
does), fed fixed noise in place of rsample's draw; the losses are SB3's
SAC.train, line for line; autograd gives the gradients the backend writes by
hand. polyak_update is SB3's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import LOG_STD_MAX, LOG_STD_MIN
import stable_baselines3

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "sac_pytorch.json"


def tensor_fixture(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def squash_case(name: str, seed: int, rows: int, act: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    mean = torch.randn(rows, act, generator=generator) * 1.2
    log_std = torch.randn(rows, act, generator=generator) * 0.8 - 0.5
    # Held by the clamp: below -20 (with a 0 mean, so u - mean is exact) and above 2
    log_std[0, 0] = -25.0
    mean[0, 0] = 0.0
    log_std[1, act - 1] = 3.0
    # Near saturation: tanh(u) close to +-1, where the epsilon matters
    mean[2, 0] = 4.5
    mean[3, act - 1] = -4.0
    mean.requires_grad_()
    log_std.requires_grad_()
    noise = torch.randn(rows, act, generator=generator)
    # At the low clamp std is 2e-9: torch's float32 gradient to the mean there
    # cancels two terms of about noise / std (1e8) and keeps only rounding; a 0
    # noise makes u - mean exactly 0, so the clamp is tested and nothing cancels
    noise[0, 0] = 0.0
    # Upstream gradients, as the actor loss and the critics send them
    grad_actions = torch.randn(rows, act, generator=generator)
    grad_log_prob = torch.randn(rows, generator=generator) * 0.1

    # SAC's Actor.get_action_dist_params, then log_prob_from_params with this noise
    clamped = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    dist = SquashedDiagGaussianDistribution(act)
    dist.proba_distribution(mean, clamped)
    gaussian_actions = mean + noise * clamped.exp()  # torch.distributions.Normal.rsample
    actions = torch.tanh(gaussian_actions)
    log_prob = dist.log_prob(actions, gaussian_actions)
    surrogate = (actions * grad_actions).sum() + (log_prob * grad_log_prob).sum()
    surrogate.backward()
    return {
        "name": name,
        "inputs": {"mean": tensor_fixture(mean), "log_std": tensor_fixture(log_std), "noise": tensor_fixture(noise),
                   "grad_actions": tensor_fixture(grad_actions), "grad_log_prob": tensor_fixture(grad_log_prob)},
        "expected": {"actions": tensor_fixture(actions), "log_prob": tensor_fixture(log_prob),
                     "deterministic": tensor_fixture(torch.tanh(mean)),
                     "grad_mean": tensor_fixture(mean.grad), "grad_log_std": tensor_fixture(log_std.grad)},
    }


def losses_case(name: str, seed: int, rows: int, gamma: float, target_entropy: float, tie: bool) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    log_ent_coef = (torch.randn(1, generator=generator) * 0.5 - 1.0).requires_grad_()
    log_prob = torch.randn(rows, 1, generator=generator) * 2.0 - 3.0
    next_log_prob = torch.randn(rows, 1, generator=generator) * 2.0 - 3.0
    next_q1 = torch.randn(rows, 1, generator=generator) * 5.0
    next_q2 = torch.randn(rows, 1, generator=generator) * 5.0
    rewards = torch.randn(rows, 1, generator=generator)
    dones = (torch.rand(rows, 1, generator=generator) < 0.3).float()
    q1 = (torch.randn(rows, 1, generator=generator) * 5.0).requires_grad_()
    q2 = (torch.randn(rows, 1, generator=generator) * 5.0).requires_grad_()
    q1_pi = (torch.randn(rows, 1, generator=generator) * 5.0)
    q2_pi = (torch.randn(rows, 1, generator=generator) * 5.0)
    if tie:  # equal Q values: the minimum's gradient goes to the first
        q2_pi[:3] = q1_pi[:3]
    q1_pi.requires_grad_()
    q2_pi.requires_grad_()
    log_prob_pi = log_prob.clone().requires_grad_()

    # SAC.train, line for line
    ent_coef = torch.exp(log_ent_coef.detach())
    ent_coef_loss = -(log_ent_coef * (log_prob + target_entropy).detach()).mean()
    ent_coef_loss.backward()
    with torch.no_grad():
        next_q_values = torch.cat([next_q1, next_q2], dim=1)
        next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
        next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
        target_q_values = rewards + (1 - dones) * gamma * next_q_values
    critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in (q1, q2))
    critic_loss.backward()
    q_values_pi = torch.cat([q1_pi, q2_pi], dim=1)
    min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
    actor_loss = (ent_coef * log_prob_pi - min_qf_pi).mean()
    actor_loss.backward()

    return {
        "name": name, "gamma": gamma, "target_entropy": target_entropy,
        "inputs": {"log_ent_coef": tensor_fixture(log_ent_coef), "ent_coef": tensor_fixture(ent_coef),
                   "log_prob": tensor_fixture(log_prob.flatten()), "next_log_prob": tensor_fixture(next_log_prob.flatten()),
                   "next_q1": tensor_fixture(next_q1), "next_q2": tensor_fixture(next_q2),
                   "rewards": tensor_fixture(rewards), "dones": tensor_fixture(dones),
                   "q1": tensor_fixture(q1), "q2": tensor_fixture(q2),
                   "q1_pi": tensor_fixture(q1_pi), "q2_pi": tensor_fixture(q2_pi)},
        "expected": {"ent_coef_loss": ent_coef_loss.item(), "grad_log_ent_coef": tensor_fixture(log_ent_coef.grad),
                     "target": tensor_fixture(target_q_values),
                     "critic_loss": critic_loss.item(), "grad_q1": tensor_fixture(q1.grad),
                     "grad_q2": tensor_fixture(q2.grad),
                     "actor_loss": actor_loss.item(), "mean_min_q": min_qf_pi.mean().item(),
                     "grad_q1_pi": tensor_fixture(q1_pi.grad), "grad_q2_pi": tensor_fixture(q2_pi.grad),
                     "grad_log_prob": tensor_fixture(log_prob_pi.grad.flatten())},
    }


def polyak_case() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(500)
    params = [torch.randn(4, 3, generator=generator), torch.randn(3, generator=generator)]
    targets = [torch.randn(4, 3, generator=generator), torch.randn(3, generator=generator)]
    before = [t.clone() for t in targets]
    tau = 0.005
    polyak_update(params, targets, tau)
    return {"tau": tau, "params": [tensor_fixture(p) for p in params],
            "targets": [tensor_fixture(t) for t in before], "expected": [tensor_fixture(t) for t in targets]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    output = parser.parse_args().output
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "generator": "cyxwiz-engine/tests/computation_truth/reference/generate_sac_fixtures.py",
        "torch_version": torch.__version__,
        "reference": f"stable_baselines3 {stable_baselines3.__version__} SAC.train, SquashedDiagGaussianDistribution, "
                     "polyak_update",
        "squash": [squash_case("squash_6_actions", 600, 8, 6), squash_case("squash_1_action", 601, 5, 1)],
        "losses": [losses_case("losses_auto", 700, 16, 0.99, -6.0, False),
                   losses_case("losses_ties", 701, 8, 0.98, -1.0, True)],
        "polyak": polyak_case(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
