#!/usr/bin/env python3
"""Generate PyTorch fixtures for the backend's DQN math (cyxwiz/rl/dqn.h).

The pieces are Stable-Baselines3 2.7's DQN.train, line for line: the target
from the target network's next Q values (their max), the Q values gathered at
the stored actions, smooth_l1_loss (Huber, beta 1) and its gradient by
autograd, which the backend writes by hand; the greedy choice is QNetwork's
argmax and the exploration rate SB3's own LinearSchedule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import stable_baselines3
from stable_baselines3.common.utils import LinearSchedule

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "dqn_pytorch.json"


def tensor_fixture(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def losses_case(name: str, seed: int, rows: int, choices: int, gamma: float) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    next_q = torch.randn(rows, choices, generator=generator) * 3.0
    next_q[0, :] = next_q[0, 0]  # a tie in the max
    rewards = torch.randn(rows, 1, generator=generator)
    dones = (torch.rand(rows, 1, generator=generator) < 0.3).float()
    actions = torch.randint(0, choices, (rows, 1), generator=generator)
    q = torch.randn(rows, choices, generator=generator) * 3.0

    # DQN.train, line for line
    with torch.no_grad():
        next_q_values, _ = next_q.max(dim=1)
        next_q_values = next_q_values.reshape(-1, 1)
        target_q_values = rewards + (1 - dones) * gamma * next_q_values
    # Differences within and past the Huber's 1, both signs, and one at 0
    q[1, actions[1, 0]] = target_q_values[1, 0] + 0.3
    q[2, actions[2, 0]] = target_q_values[2, 0] - 0.7
    q[3, actions[3, 0]] = target_q_values[3, 0] + 4.0
    q[4, actions[4, 0]] = target_q_values[4, 0] - 2.5
    q[5, actions[5, 0]] = target_q_values[5, 0]
    q.requires_grad_()
    current_q_values = torch.gather(q, dim=1, index=actions.long())
    loss = F.smooth_l1_loss(current_q_values, target_q_values)
    loss.backward()

    return {
        "name": name, "gamma": gamma,
        "inputs": {"next_q": tensor_fixture(next_q), "rewards": tensor_fixture(rewards),
                   "dones": tensor_fixture(dones), "actions": tensor_fixture(actions.float()),
                   "q": tensor_fixture(q)},
        "expected": {"target": tensor_fixture(target_q_values), "loss": loss.item(),
                     "mean_q": current_q_values.mean().item(), "grad_q": tensor_fixture(q.grad)},
    }


def greedy_case(name: str, seed: int, rows: int, choices: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(rows, choices, generator=generator)
    q[0, :] = 1.5             # all tied: the first
    q[1, 1] = q[1, choices - 1] = q[1].max() + 1.0  # tied best: the first of them
    return {"name": name, "inputs": {"q": tensor_fixture(q)},
            "expected": {"greedy": tensor_fixture(q.argmax(dim=1).float())}}


def schedule_cases() -> list[dict[str, Any]]:
    cases = []
    for start, end, fraction in ((1.0, 0.05, 0.1), (1.0, 0.04, 0.16), (0.5, 0.1, 1.0), (1.0, 0.05, 0.0)):
        schedule = LinearSchedule(start, end, fraction)
        for progress in (1.0, 0.99, 0.95, 0.9, 0.84, 0.5, 0.0):
            if fraction == 0.0 and progress == 1.0:
                continue  # SB3 divides 0 by 0 there; DQN never asks at progress 1 (it steps first)
            cases.append({"start": start, "end": end, "fraction": fraction, "progress_remaining": progress,
                          "rate": schedule(progress)})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    output = parser.parse_args().output
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "generator": "cyxwiz-engine/tests/computation_truth/reference/generate_dqn_fixtures.py",
        "torch_version": torch.__version__,
        "reference": f"stable_baselines3 {stable_baselines3.__version__} DQN.train, QNetwork argmax, "
                     "LinearSchedule",
        "losses": [losses_case("losses_2", 1000, 16, 2, 0.99), losses_case("losses_5", 1001, 12, 5, 0.98)],
        "greedy": [greedy_case("greedy_2", 1100, 6, 2), greedy_case("greedy_4", 1101, 9, 4)],
        "schedule": schedule_cases(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
