#!/usr/bin/env python3
"""Reference of CyxWiz multi-node training in rounds (TOFIX166).

The algorithm of docs/architecture/distributed-training.md: time-based
DiLoCo rounds. This file is the contract the Engine, the server
node (worker) and the coordinator must reproduce; PyTorch is the oracle,
as for single-node layers.

The exchanged state is one float32 vector cut into segments (`Segment`), one
per stored tensor. A segment is trainable (it has a gradient) or a statistic
(BatchNorm's running mean and variance: training moves it without a gradient).

One round r:

1. The coordinator sends the global state theta_r to every worker.
2. Worker i sets its model to theta_r, keeps its own inner optimizer state
   from earlier rounds, and trains `steps_i` steps on its data (a round has
   a deadline; a faster worker does more steps). It reports
   delta_i = theta_r - theta_i, n_i (the samples trained) and steps_i.
3. The update travels encoded per segment (`encode_update`): large trainable
   segments as int8, small ones and statistics as float32.
4. The coordinator rejects an update that is not finite, screens the rest by
   norm per step (`screen_by_norm`), aggregates the trainable segments
   (`aggregate`: sample-weighted mean, or a trimmed mean that resists one bad
   worker) and takes a Nesterov outer step on them (`OuterOptimizer`).
   Statistics are averaged by samples and applied as they are: no outer rate,
   no momentum.

Everything is float32, the one type every CyxWiz device has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence, Union

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
# A segment travels as int8 only from this many values: below it the 1 KiB
# table costs more than it saves, and a handful of values has no spread worth
# 256 buckets.
INT8_MIN_VALUES = 1024

TRAINABLE = "trainable"
STATISTIC = "statistic"


# ---------------------------------------------------------------- parameters

@dataclass(frozen=True)
class Segment:
    """One stored tensor's place in the exchanged state."""
    name: str
    count: int
    kind: str = TRAINABLE


def single_segment(count: int) -> list[Segment]:
    """A layout of one trainable segment: the whole vector."""
    return [Segment("all", count)]


def layout_count(layout: Sequence[Segment]) -> int:
    return sum(s.count for s in layout)


def trainable_mask(layout: Sequence[Segment]) -> torch.Tensor:
    """True where the exchanged state is trainable, False where it is a statistic."""
    return torch.cat([torch.full((s.count,), s.kind == TRAINABLE, dtype=torch.bool) for s in layout]) \
        if layout else torch.zeros(0, dtype=torch.bool)


def model_state(model: torch.nn.Module) -> tuple[list[Segment], list[torch.Tensor]]:
    """A model's exchanged state: per module, its parameters then its float
    buffers (BatchNorm's batch counter is not a float and is not exchanged)."""
    layout: list[Segment] = []
    tensors: list[torch.Tensor] = []
    for module_name, module in model.named_modules():
        prefix = module_name + "." if module_name else ""
        for name, parameter in module.named_parameters(recurse=False):
            layout.append(Segment(prefix + name, parameter.numel(), TRAINABLE))
            tensors.append(parameter)
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.is_floating_point():
                layout.append(Segment(prefix + name, buffer.numel(), STATISTIC))
                tensors.append(buffer)
    return layout, tensors


def flatten(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """The tensors as one float32 vector, in the given order."""
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
    """Encodes a float32 vector as 256-level codes plus a decode table.

    Mean, standard deviation and the bucket means are summed in float64 and
    stored as float32; the codes are computed in float32. An implementation
    that follows the same widths reproduces the codes exactly.
    """
    values = values.to(DTYPE).reshape(-1)
    if values.numel() == 0:
        return Int8Payload(torch.zeros(0, dtype=torch.uint8), torch.zeros(INT8_LEVELS, dtype=DTYPE))
    wide = values.double()
    mean = wide.mean()
    std = wide.std(unbiased=False)
    low = (mean - INT8_SIGMAS * std).to(DTYPE)
    high = (mean + INT8_SIGMAS * std).to(DTYPE)
    if not bool(high > low):
        # No spread: every code is 0 and decodes to the value itself.
        codes = torch.zeros(values.numel(), dtype=torch.uint8)
        return Int8Payload(codes, torch.full((INT8_LEVELS,), float(mean.to(DTYPE)), dtype=DTYPE))
    clipped = values.clamp(low, high)
    scaled = (clipped - low) / (high - low) * (INT8_LEVELS - 1)
    codes_long = scaled.round().clamp(0, INT8_LEVELS - 1).to(torch.long)   # round half to even
    # Each code decodes to the mean of the (clipped) values that got it; an
    # empty bucket decodes to its centre.
    sums = torch.zeros(INT8_LEVELS, dtype=torch.float64).index_add_(0, codes_long, clipped.double())
    counts = torch.zeros(INT8_LEVELS, dtype=torch.float64).index_add_(0, codes_long, torch.ones_like(wide))
    centres = low + (high - low) * torch.arange(INT8_LEVELS, dtype=DTYPE) / (INT8_LEVELS - 1)
    table = torch.where(counts > 0, (sums / counts.clamp_min(1)).to(DTYPE), centres)
    return Int8Payload(codes_long.to(torch.uint8), table)


def dequantize_int8(payload: Int8Payload) -> torch.Tensor:
    return payload.table[payload.codes.to(torch.long)]


# ------------------------------------------------------------ update encoding

def segment_travels_as_int8(segment: Segment) -> bool:
    """Large trainable segments only. A statistic stays float32 (a rounded
    variance can turn negative), and so does a small segment."""
    return segment.kind == TRAINABLE and segment.count >= INT8_MIN_VALUES


SegmentPayload = Union[torch.Tensor, Int8Payload]


def encode_update(delta: torch.Tensor, layout: Sequence[Segment], int8: bool = True) -> list[SegmentPayload]:
    """A worker's update on the wire: one payload per segment, in layout order.

    A worker never sends an update that is not finite; it reports the failure
    instead.
    """
    delta = delta.to(DTYPE).reshape(-1)
    if delta.numel() != layout_count(layout):
        raise ValueError(f"the update has {delta.numel()} values, the layout takes {layout_count(layout)}")
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("the update holds a value that is not a finite number")
    payloads: list[SegmentPayload] = []
    offset = 0
    for segment in layout:
        part = delta[offset:offset + segment.count]
        payloads.append(quantize_int8(part) if int8 and segment_travels_as_int8(segment) else part.clone())
        offset += segment.count
    return payloads


def decode_update(payloads: Sequence[SegmentPayload], layout: Sequence[Segment]) -> torch.Tensor:
    if len(payloads) != len(layout):
        raise ValueError(f"the update has {len(payloads)} segments, the layout has {len(layout)}")
    parts = []
    for payload, segment in zip(payloads, layout):
        part = dequantize_int8(payload) if isinstance(payload, Int8Payload) else payload.to(DTYPE)
        if part.numel() != segment.count:
            raise ValueError(f"segment {segment.name} has {part.numel()} values, the layout says {segment.count}")
        parts.append(part)
    return torch.cat(parts) if parts else torch.zeros(0, dtype=DTYPE)


def wire_bytes(payloads: Sequence[SegmentPayload]) -> int:
    """Bytes on the wire: a code is 1 byte, a float 4, a table 1 KiB."""
    total = 0
    for payload in payloads:
        if isinstance(payload, Int8Payload):
            total += payload.codes.numel() + 4 * INT8_LEVELS
        else:
            total += 4 * payload.numel()
    return total


# --------------------------------------------------------------- aggregation

def screen_by_norm(deltas: Sequence[torch.Tensor], steps: Sequence[int], factor: float = 3.0,
                   mask: torch.Tensor | None = None) -> list[bool]:
    """Keeps an update whose L2 norm **per step** is at most `factor` x the
    median of the group's.

    A fast worker takes more steps in a round and so sends a larger update;
    dividing by its steps compares like with like (a raw-norm screen rejects
    the fastest honest worker when speeds differ by 10). `mask` selects the
    trainable values. An update that is not finite is never kept. With fewer
    than three finite updates there is no meaningful median and every finite
    one is kept. The median of an even count is the mean of the middle two.
    """
    if len(deltas) != len(steps):
        raise ValueError("every update needs its step count")
    if any(s <= 0 for s in steps):
        raise ValueError("every update needs a positive step count")
    finite = [bool(torch.isfinite(d).all()) for d in deltas]
    rates = [float((d if mask is None else d[mask]).norm()) / s if ok else float("nan")
             for d, s, ok in zip(deltas, steps, finite)]
    known = sorted(r for r, ok in zip(rates, finite) if ok)
    if len(known) < 3:
        return finite
    middle = len(known) // 2
    median = known[middle] if len(known) % 2 else 0.5 * (known[middle - 1] + known[middle])
    limit = factor * median
    return [ok and r <= limit for r, ok in zip(rates, finite)]


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

    def state(self) -> list[torch.Tensor]:
        return model_state(self.model)[1]


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
    unflatten_into(theta, worker.state())
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
    delta = theta - flatten(worker.state())
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
                      norm_factor: float = 3.0,
                      layout: Sequence[Segment] | None = None) -> RoundResult:
    """Coordinator side: decode, reject, screen, aggregate, outer step.

    A worker that missed the deadline simply has no report this round. When
    no update is kept the round is abandoned: theta and the momentum stay as
    they were. `layout` defaults to one trainable segment.
    """
    layout = list(layout) if layout is not None else single_segment(theta.numel())
    trainable = trainable_mask(layout)
    reports = [r for r in reports if r.samples > 0 and r.steps > 0]
    if not reports:
        return RoundResult(theta, [], [], torch.zeros_like(theta))
    received = []
    for r in reports:
        if bool(torch.isfinite(r.delta).all()):
            received.append(decode_update(encode_update(r.delta, layout, int8), layout))
        else:
            received.append(r.delta)   # rejected by the screen below
    keep = screen_by_norm(received, [r.steps for r in reports], norm_factor, trainable)
    used = [r for r, k in zip(reports, keep) if k]
    rejected = [r.name for r, k in zip(reports, keep) if not k]
    if not used:
        return RoundResult(theta, [], rejected, torch.zeros_like(theta))
    deltas = [d for d, k in zip(received, keep) if k]
    samples = [r.samples for r in used]
    aggregated = torch.zeros_like(theta)
    aggregated[trainable] = aggregate([d[trainable] for d in deltas], samples, method, trim)
    step = aggregated.clone()
    step[~trainable] = 0.0
    momentum_before = outer.momentum
    new_theta = outer.step(theta, step)
    if bool((~trainable).any()):
        statistics = aggregate([d[~trainable] for d in deltas], samples, "weighted_mean")
        aggregated[~trainable] = statistics
        new_theta[~trainable] = theta[~trainable] - statistics
    if not bool(torch.isfinite(new_theta).all()):
        outer.momentum = momentum_before
        return RoundResult(theta, [], [r.name for r in reports], torch.zeros_like(theta))
    return RoundResult(new_theta, [r.name for r in used], rejected, aggregated)
