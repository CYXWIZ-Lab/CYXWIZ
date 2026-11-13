#!/usr/bin/env python3
"""Reference of CyxWiz multi-node training in rounds (TOFIX166 phase 0).

The algorithm of docs/architecture/distributed-training.md: time-based
DiLoCo rounds. This file is the contract the Engine, the server
node (worker) and the coordinator must reproduce; PyTorch is the oracle,
as for single-node layers.

One round r:

1. The coordinator sends the global weights theta_r to every worker.
2. Worker i sets its model to theta_r, keeps its own inner optimizer state
   from earlier rounds, and trains `steps_i` steps on its chunks (a round has
   a deadline; a faster worker does more steps). It reports the
   pseudo-gradient delta_i = theta_r - theta_i and n_i, the samples trained.
3. The update travels as int8 (`quantize_int8`); reduction is in float32.
4. The coordinator screens the updates (`screen_by_norm`), aggregates them
   (`aggregate`: sample-weighted mean, or a trimmed mean that resists one bad
   worker), and takes a Nesterov outer step (`OuterOptimizer`).

Everything is float32, the one type every CyxWiz device has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import torch

DTYPE = torch.float32

# Starting values from DiLoCo (Douillard et al., 2023) and INTELLECT-1
# (Prime Intellect, 2024): outer SGD with Nesterov momentum.
OUTER_LR = 0.7
OUTER_MOMENTUM = 0.9

# int8 wire format (INTELLECT-1): values clipped to mean +- 6 std, the range
# split into 256 equal buckets, each code decoded as the mean of the values
# that fell in its bucket (the table travels with the codes).
INT8_SIGMAS = 6.0
INT8_LEVELS = 256


# ---------------------------------------------------------------- parameters

def flatten(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """The model's parameters as one float32 vector, in the given order."""
    return torch.cat([t.detach().reshape(-1).to(DTYPE) for t in tensors])


def unflatten_into(vector: torch.Tensor, tensors: Sequence[torch.Tensor]) -> None:
    """Writes `vector` back into `tensors` (same order as `flatten`)."""
    offset = 0
    with torch.no_grad():
        for t in tensors:
            count = t.numel()
            t.copy_(vector[offset:offset + count].view_as(t))
            offset += count
    if offset != vector.numel():
        raise ValueError(f"vector has {vector.numel()} values, parameters take {offset}")


# ------------------------------------------------------------------- int8 wire

@dataclass
class Int8Payload:
    codes: torch.Tensor   # uint8, one per value
    table: torch.Tensor   # float32[256], the decoded value of each code


def quantize_int8(values: torch.Tensor) -> Int8Payload:
    """Encodes a float32 vector as 256-level codes plus a decode table."""
    values = values.to(DTYPE).reshape(-1)
    mean = values.mean()
    std = values.std(unbiased=False)
    if values.numel() == 0 or float(std) == 0.0:
        codes = torch.zeros(values.numel(), dtype=torch.uint8)
        table = torch.full((INT8_LEVELS,), float(mean) if values.numel() else 0.0, dtype=DTYPE)
        return Int8Payload(codes, table)
    low = mean - INT8_SIGMAS * std
    high = mean + INT8_SIGMAS * std
    clipped = values.clamp(low, high)
    scaled = (clipped - low) / (high - low) * (INT8_LEVELS - 1)
    codes_long = scaled.round().clamp(0, INT8_LEVELS - 1).to(torch.long)
    # Each code decodes to the mean of the (clipped) values that got it; an
    # empty bucket decodes to its centre.
    sums = torch.zeros(INT8_LEVELS, dtype=DTYPE).index_add_(0, codes_long, clipped)
    counts = torch.zeros(INT8_LEVELS, dtype=DTYPE).index_add_(0, codes_long, torch.ones_like(clipped))
    centres = low + (high - low) * torch.arange(INT8_LEVELS, dtype=DTYPE) / (INT8_LEVELS - 1)
    table = torch.where(counts > 0, sums / counts.clamp_min(1), centres)
    return Int8Payload(codes_long.to(torch.uint8), table)


def dequantize_int8(payload: Int8Payload) -> torch.Tensor:
    return payload.table[payload.codes.to(torch.long)]


# --------------------------------------------------------------- aggregation

def screen_by_norm(deltas: Sequence[torch.Tensor], factor: float = 3.0) -> list[bool]:
    """Keeps an update whose L2 norm is at most `factor` x the median norm.

    A cheap first defence against a scaled or random update; with fewer than
    three updates there is no meaningful median and every update is kept.
    """
    if len(deltas) < 3:
        return [True] * len(deltas)
    norms = torch.stack([d.norm() for d in deltas])
    limit = factor * norms.median()
    return [bool(n <= limit) for n in norms]


def aggregate(deltas: Sequence[torch.Tensor], samples: Sequence[int],
              method: str = "weighted_mean", trim: int = 1) -> torch.Tensor:
    """Combines the workers' pseudo-gradients.

    weighted_mean: sum_i n_i * delta_i / sum_i n_i.
    trimmed_mean:  per coordinate, the `trim` smallest and `trim` largest
                   values are dropped, and the rest are averaged weighted by
                   their workers' n_i. Needs more than 2 * trim updates;
                   with fewer it falls back to weighted_mean.
    """
    if not deltas:
        raise ValueError("no updates to aggregate")
    stacked = torch.stack([d.to(DTYPE) for d in deltas])          # [k, d]
    weights = torch.tensor(samples, dtype=DTYPE)
    if bool((weights <= 0).any()):
        raise ValueError("every update needs a positive sample count")
    if method == "weighted_mean" or len(deltas) <= 2 * trim:
        return (weights[:, None] * stacked).sum(0) / weights.sum()
    if method != "trimmed_mean":
        raise ValueError(f"unknown aggregation method {method!r}")
    order = stacked.argsort(dim=0)                                 # per coordinate
    keep = torch.zeros_like(stacked, dtype=torch.bool)
    kept_rows = order[trim:len(deltas) - trim]                     # [k - 2t, d]
    keep.scatter_(0, kept_rows, True)
    w = weights[:, None].expand_as(stacked) * keep
    return (w * stacked).sum(0) / w.sum(0)


# ------------------------------------------------------------ outer optimizer

@dataclass
class OuterOptimizer:
    """SGD with Nesterov momentum on the aggregated pseudo-gradient.

    momentum = mu * momentum + delta
    theta    = theta - lr * (delta + mu * momentum)
    (torch.optim.SGD(nesterov=True, dampening=0) on the same vectors.)
    """
    lr: float = OUTER_LR
    mu: float = OUTER_MOMENTUM
    momentum: torch.Tensor | None = None

    def step(self, theta: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if self.momentum is None:
            self.momentum = torch.zeros_like(theta)
        self.momentum = self.mu * self.momentum + delta
        return theta - self.lr * (delta + self.mu * self.momentum)


# ----------------------------------------------------------------- one round

@dataclass
class Worker:
    """A server node in a group: its model copy and inner optimizer."""
    name: str
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    batches: Callable[[int], Iterable[tuple[torch.Tensor, torch.Tensor]]]
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = field(
        default=torch.nn.functional.cross_entropy)

    def parameters(self) -> list[torch.Tensor]:
        return list(self.model.parameters())


@dataclass
class WorkerReport:
    name: str
    delta: torch.Tensor
    samples: int
    steps: int
    last_loss: float


def train_round(worker: Worker, theta: torch.Tensor, steps: int) -> WorkerReport:
    """Worker side of one round: start from theta, train `steps` steps.

    The inner optimizer keeps its state between rounds (DiLoCo: syncing it
    did not help and tripled the traffic).
    """
    unflatten_into(theta, worker.parameters())
    worker.model.train()
    samples = 0
    loss_value = float("nan")
    done = 0
    for x, y in worker.batches(steps):
        worker.optimizer.zero_grad(set_to_none=True)
        loss = worker.loss_fn(worker.model(x), y)
        loss.backward()
        worker.optimizer.step()
        samples += int(x.shape[0])
        loss_value = float(loss.detach())
        done += 1
    delta = theta - flatten(worker.parameters())
    return WorkerReport(worker.name, delta, samples, done, loss_value)


@dataclass
class RoundResult:
    theta: torch.Tensor
    used: list[str]
    rejected: list[str]
    aggregated: torch.Tensor


def coordinator_round(theta: torch.Tensor, reports: Sequence[WorkerReport],
                      outer: OuterOptimizer, *, int8: bool = True,
                      method: str = "trimmed_mean", trim: int = 1,
                      norm_factor: float = 3.0) -> RoundResult:
    """Coordinator side: decode, screen, aggregate, outer step.

    A worker that missed the deadline simply has no report this round.
    """
    reports = [r for r in reports if r.samples > 0]
    if not reports:
        return RoundResult(theta, [], [], torch.zeros_like(theta))
    received = [dequantize_int8(quantize_int8(r.delta)) if int8 else r.delta for r in reports]
    keep = screen_by_norm(received, norm_factor)
    used = [r for r, k in zip(reports, keep) if k]
    rejected = [r.name for r, k in zip(reports, keep) if not k]
    deltas = [d for d, k in zip(received, keep) if k]
    aggregated = aggregate(deltas, [r.samples for r in used], method, trim)
    return RoundResult(outer.step(theta, aggregated), [r.name for r in used], rejected, aggregated)
