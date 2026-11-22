#!/usr/bin/env python3
"""Simulation study of multi-node training in rounds (TOFIX166 step 0.2).

Runs round_training_reference.py on a simulated clock to compare, for the
same wall-clock budget:

- one node alone (the fastest);
- every-step synchronisation (data parallel: the slowest node and two model
  transfers set the time of every step);
- rounds of several lengths, float32 or int8 updates;
- rounds with a node that misses rounds;
- rounds with a node that sends a bad update, with and without the screen
  and the trimmed mean.

The model trained here is small so the study runs in minutes; the network
cost is charged for a notional model size (`--model-mb`) at a measured link
speed (`--mbit`). Node speeds are steps per second.

    py -3.12 round_training_study.py --output study.json

`--many-rounds` runs the second question instead: with data that does not
repeat (`--rows 2000000`) and short rounds, does a group beat one of its own
nodes? Equal nodes, the lab mix and six nodes, rounds of 30, 100 and 300 s.

    py -3.12 round_training_study.py --many-rounds --rows 2000000 --model-mb 40

`--contract` runs the third: the questions the design review left open. For
each seed, with data that does not repeat and 30 s rounds: the best single
node; that node alone with the outer step (the control: how much of a group's
gain is the outer step and not the extra nodes); equal nodes; a group inside
the group contract (speeds within a factor of 4) and one outside it (10/1/1);
one whole-vector table against a table per segment. It prints the final loss
with its spread over the seeds, and the time each case needs to reach the
loss the best single node ends at.

    py -3.12 round_training_study.py --contract --rows 2000000 --model-mb 40 --seeds 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

import round_training_reference as ref

BATCH = 32
FEATURES = 20
CLASSES = 10
TRAIN_ROWS = 60_000
HELD_OUT_ROWS = 5_000
INNER_LR = 1e-3


class Student(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(FEATURES, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_data(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Labels from a fixed random teacher network, so the task is learnable."""
    g = torch.Generator().manual_seed(seed)
    teacher = torch.nn.Sequential(
        torch.nn.Linear(FEATURES, 96), torch.nn.Tanh(), torch.nn.Linear(96, CLASSES))
    with torch.no_grad():
        for p in teacher.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.6)
        x = torch.randn(TRAIN_ROWS + HELD_OUT_ROWS, FEATURES, generator=g)
        y = teacher(x).argmax(dim=1)
    return x[:TRAIN_ROWS], y[:TRAIN_ROWS], x[TRAIN_ROWS:], y[TRAIN_ROWS:]


class WorkQueue:
    """Rows handed out in order from a shuffled epoch; reshuffled when used up."""

    def __init__(self, rows: int, seed: int) -> None:
        self.rows = rows
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(rows, generator=self.generator)
        self.at = 0
        self.epochs = 0.0

    def take(self, count: int) -> torch.Tensor:
        parts = []
        while count > 0:
            if self.at == self.rows:
                self.order = torch.randperm(self.rows, generator=self.generator)
                self.at = 0
            n = min(count, self.rows - self.at)
            parts.append(self.order[self.at:self.at + n])
            self.at += n
            count -= n
            self.epochs += n / self.rows
        return torch.cat(parts)


@dataclass
class Link:
    model_mb: float
    mbit: float

    def seconds(self, fraction: float = 1.0) -> float:
        return self.model_mb * 8.0 * fraction / self.mbit


def held_out_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(torch.nn.functional.cross_entropy(model(x), y))


def run_single(seed: int, speed: float, budget: float, data) -> list[tuple[float, float]]:
    x, y, hx, hy = data
    torch.manual_seed(seed)
    model = Student()
    optimizer = torch.optim.Adam(model.parameters(), lr=INNER_LR)
    queue = WorkQueue(TRAIN_ROWS, seed)
    steps = int(budget * speed)
    every = max(1, steps // 60)
    curve = [(0.0, held_out_loss(model, hx, hy))]
    for step in range(1, steps + 1):
        rows = queue.take(BATCH)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x[rows]), y[rows]).backward()
        optimizer.step()
        if step % every == 0 or step == steps:
            curve.append((step / speed, held_out_loss(model, hx, hy)))
    return curve


def run_every_step(seed: int, speeds: list[float], budget: float, link: Link, data) -> list[tuple[float, float]]:
    """Data parallel: one step = the slowest node's step plus gradients out and back."""
    x, y, hx, hy = data
    torch.manual_seed(seed)
    model = Student()
    optimizer = torch.optim.Adam(model.parameters(), lr=INNER_LR)
    queue = WorkQueue(TRAIN_ROWS, seed)
    step_seconds = 1.0 / min(speeds) + 2.0 * link.seconds()
    steps = int(budget / step_seconds)
    every = max(1, steps // 60)
    curve = [(0.0, held_out_loss(model, hx, hy))]
    for step in range(1, steps + 1):
        rows = queue.take(BATCH * len(speeds))   # averaged gradient of every node's batch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x[rows]), y[rows]).backward()
        optimizer.step()
        if step % every == 0 or step == steps:
            curve.append((step * step_seconds, held_out_loss(model, hx, hy)))
    return curve


def run_rounds(seed: int, speeds: list[float], budget: float, link: Link, data, *,
               round_seconds: float, int8: bool, method: str = "trimmed_mean",
               screen: bool = True, miss: dict[int, float] | None = None,
               bad: set[int] | None = None,
               outer_lr: float = ref.OUTER_LR, outer_mu: float = ref.OUTER_MOMENTUM,
               per_segment: bool = False) -> dict:
    """Rounds on a simulated clock. `miss[i]` is worker i's chance of missing a
    round; a worker in `bad` reports 50x a random update instead of its own."""
    x, y, hx, hy = data
    miss = miss or {}
    bad = bad or set()
    torch.manual_seed(seed)
    template = Student()
    theta = ref.flatten(list(template.parameters()))
    # One table for the whole vector, or the wire format: a table per segment.
    layout = ref.model_state(template)[0] if per_segment else None
    queue = WorkQueue(TRAIN_ROWS, seed)
    chance = torch.Generator().manual_seed(seed + 7)
    workers = []
    for i in range(len(speeds)):
        model = Student()
        workers.append(ref.Worker(f"w{i}", model, torch.optim.Adam(model.parameters(), lr=INNER_LR),
                                  batches=lambda n: iter(())))
    outer = ref.OuterOptimizer(lr=outer_lr, mu=outer_mu)
    # One round: weights down in float32, training, the update back up.
    transfer = link.seconds() + link.seconds(0.25 if int8 else 1.0)
    round_total = round_seconds + transfer
    rounds = int(budget / round_total)
    curve = [(0.0, held_out_loss(template, hx, hy))]
    rejected = 0
    for r in range(1, rounds + 1):
        reports = []
        for i, worker in enumerate(workers):
            if float(torch.rand(1, generator=chance)) < miss.get(i, 0.0):
                continue
            steps = max(1, int(round_seconds * speeds[i]))
            rows = queue.take(BATCH * steps).view(steps, BATCH)
            worker.batches = (lambda picked: (lambda n: ((x[b], y[b]) for b in picked[:n])))(rows)
            report = ref.train_round(worker, theta, steps)
            if i in bad:
                report.delta = torch.randn(theta.shape, generator=chance) * report.delta.norm() * 50.0 / math.sqrt(theta.numel())
            reports.append(report)
        result = ref.coordinator_round(theta, reports, outer, int8=int8, method=method,
                                       norm_factor=3.0 if screen else float("inf"), layout=layout)
        rejected += len(result.rejected)
        theta = result.theta
        if not bool(torch.isfinite(theta).all()):
            curve.append((r * round_total, float("inf")))
            break
        ref.unflatten_into(theta, list(template.parameters()))
        curve.append((r * round_total, held_out_loss(template, hx, hy)))
    return {"curve": curve, "rounds": rounds, "rejected": rejected, "epochs": queue.epochs,
            "network_share": transfer / round_total}


def time_to(curve: list[tuple[float, float]], target: float) -> float | None:
    for t, loss in curve:
        if loss <= target:
            return t
    return None


def many_rounds(args: argparse.Namespace) -> None:
    link = Link(args.model_mb[0], args.mbit)
    results = {}
    for seed in args.seeds:
        data = make_data(100 + seed)

        def show(name: str, curve: list[tuple[float, float]], extra: str = "") -> None:
            results[f"{name} (seed {seed})"] = curve
            half = min(loss for t, loss in curve if t <= args.budget / 2)
            print(f"{name:52s} half {half:.4f} final {curve[-1][1]:.4f} {extra}", flush=True)

        show("one node 3 steps/s", run_single(seed, 3.0, args.budget, data))
        show("one node 10 steps/s", run_single(seed, 10.0, args.budget, data))
        for speeds, label in (([3.0, 3.0, 3.0], "equal 3/3/3"), ([10.0, 3.0, 1.0], "lab 10/3/1"),
                              ([3.0] * 6, "equal x6")):
            for seconds in (30.0, 100.0, 300.0):
                for method in ("weighted_mean", "trimmed_mean"):
                    if method == "trimmed_mean" and seconds != 100.0:
                        continue
                    run = run_rounds(seed, speeds, args.budget, link, data, round_seconds=seconds,
                                     int8=True, method=method)
                    show(f"{label}, rounds {seconds:.0f} s, {method}", run["curve"],
                         f"rounds {run['rounds']} network {run['network_share']:.2f}")
    if args.output:
        args.output.write_text(json.dumps({"budget": args.budget, "mbit": args.mbit, "curves": results}, indent=1),
                               encoding="utf-8")


def contract_study(args: argparse.Namespace) -> None:
    link = Link(args.model_mb[0], args.mbit)
    cases = [
        ("one node 10 steps/s (best single node)", None, {}),
        ("one node 10 steps/s + outer step (control)", [10.0], {}),
        ("one node 3 steps/s", None, {"speed": 3.0}),
        ("one node 3 steps/s + outer step (control)", [3.0], {}),
        ("equal 3/3/3", [3.0, 3.0, 3.0], {}),
        ("inside the contract 8/4/2", [8.0, 4.0, 2.0], {}),
        ("inside the contract 8/4/2, table per segment", [8.0, 4.0, 2.0], {"per_segment": True}),
        ("outside the contract 10/1/1", [10.0, 1.0, 1.0], {}),
        ("outside the contract 10/1/1, table per segment", [10.0, 1.0, 1.0], {"per_segment": True}),
    ]
    curves: dict[str, list] = {name: [] for name, _, _ in cases}
    rejected: dict[str, int] = {name: 0 for name, _, _ in cases}
    for seed in args.seeds:
        data = make_data(100 + seed)
        for name, speeds, options in cases:
            if speeds is None:
                curve = run_single(seed, options.get("speed", 10.0), args.budget, data)
            else:
                run = run_rounds(seed, speeds, args.budget, link, data, round_seconds=30.0, int8=True,
                                 method="weighted_mean", **options)
                curve = run["curve"]
                rejected[name] += run["rejected"]
            curves[name].append(curve)
            print(f"seed {seed} {name:50s} final {curve[-1][1]:.4f}", flush=True)
    targets = [curve[-1][1] for curve in curves[cases[0][0]]]
    print()
    print(f"{'case':52s} {'final (mean)':>12s} {'spread':>8s} {'rejected':>8s}  time to the best single node's final loss, per seed")
    for name, _, _ in cases:
        finals = [curve[-1][1] for curve in curves[name]]
        mean = sum(finals) / len(finals)
        spread = max(finals) - min(finals)
        times = [time_to(curve, target) for curve, target in zip(curves[name], targets)]
        shown = " ".join("never" if t is None else f"{t:5.0f}" for t in times)
        print(f"{name:52s} {mean:12.4f} {spread:8.4f} {rejected[name]:8d}  {shown}", flush=True)
    if args.output:
        args.output.write_text(json.dumps({"budget": args.budget, "mbit": args.mbit, "seeds": args.seeds,
                                           "curves": curves, "rejected": rejected}, indent=1), encoding="utf-8")


def main() -> None:
    global TRAIN_ROWS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=3600.0, help="simulated seconds")
    parser.add_argument("--mbit", type=float, default=100.0, help="link speed (lab: 65-130)")
    parser.add_argument("--model-mb", type=float, nargs="+", default=[40.0, 400.0],
                        help="notional model sizes charged to the network")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--rows", type=int, default=TRAIN_ROWS, help="training rows")
    parser.add_argument("--many-rounds", action="store_true",
                        help="short rounds on data that does not repeat (see the module text)")
    parser.add_argument("--contract", action="store_true",
                        help="the review's open questions (see the module text)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    TRAIN_ROWS = args.rows
    if args.many_rounds:
        many_rounds(args)
        return
    if args.contract:
        contract_study(args)
        return

    lab = [10.0, 3.0, 1.0]       # steps per second: fast, middle, slow (about 10x apart)
    equal = [3.0, 3.0, 3.0]
    rows = []

    def record(name: str, mb: float, curves: list[list[tuple[float, float]]], extra: dict | None = None) -> None:
        finals = [c[-1][1] for c in curves]
        rows.append({"case": name, "model_mb": mb, "final_loss": sum(finals) / len(finals),
                     "curves": curves, **(extra or {})})
        print(f"{name:58s} {mb:6.0f} MB  final {rows[-1]['final_loss']:.4f}", flush=True)

    data = {seed: make_data(100 + seed) for seed in args.seeds}

    for mb in args.model_mb:
        link = Link(mb, args.mbit)
        record("one node (fast, 10 steps/s)", mb,
               [run_single(s, lab[0], args.budget, data[s]) for s in args.seeds])
        record("one node (3 steps/s)", mb,
               [run_single(s, 3.0, args.budget, data[s]) for s in args.seeds])
        record("every-step sync, lab mix 10/3/1", mb,
               [run_every_step(s, lab, args.budget, link, data[s]) for s in args.seeds])

        def rounds_case(name: str, speeds: list[float], **kw) -> None:
            runs = [run_rounds(s, speeds, args.budget, link, data[s], **kw) for s in args.seeds]
            record(name, mb, [r["curve"] for r in runs],
                   {"rounds": runs[0]["rounds"], "rejected": sum(r["rejected"] for r in runs),
                    "epochs": runs[0]["epochs"], "network_share": runs[0]["network_share"]})

        for seconds in (60.0, 300.0, 900.0):
            rounds_case(f"rounds {seconds:.0f} s, float32, lab mix", lab, round_seconds=seconds, int8=False)
        rounds_case("rounds 300 s, int8, lab mix", lab, round_seconds=300.0, int8=True)
        rounds_case("rounds 300 s, int8, plain mean, lab mix", lab, round_seconds=300.0, int8=True,
                    method="weighted_mean")
        rounds_case("rounds 300 s, int8, middle node misses 30 %", lab, round_seconds=300.0, int8=True,
                    miss={1: 0.3})
        rounds_case("rounds 300 s, int8, equal nodes 3/3/3", equal, round_seconds=300.0, int8=True)
        rounds_case("rounds 300 s, int8, equal nodes, plain mean", equal, round_seconds=300.0, int8=True,
                    method="weighted_mean")
        four = [10.0, 3.0, 1.0, 3.0]
        rounds_case("rounds 300 s, int8, 4 nodes, none bad", four, round_seconds=300.0, int8=True)
        rounds_case("rounds 300 s, int8, 4th node bad, plain mean no screen", four, round_seconds=300.0,
                    int8=True, method="weighted_mean", screen=False, bad={3})
        rounds_case("rounds 300 s, int8, 4th node bad, screen + trimmed", four, round_seconds=300.0,
                    int8=True, bad={3})
        rounds_case("rounds 300 s, int8, 4th node bad, trimmed only", four, round_seconds=300.0,
                    int8=True, screen=False, bad={3})

    # Time to reach the loss the fast node alone has at half the budget.
    print()
    for mb in args.model_mb:
        base = next(r for r in rows if r["model_mb"] == mb and r["case"].startswith("one node (fast"))
        targets = [min(loss for t, loss in c if t <= args.budget / 2) for c in base["curves"]]
        for r in (r for r in rows if r["model_mb"] == mb):
            times = [time_to(c, target) for c, target in zip(r["curves"], targets)]
            r["seconds_to_target"] = times
            shown = ", ".join("never" if t is None else f"{t:6.0f} s" for t in times)
            print(f"{r['case']:58s} {mb:6.0f} MB  to target: {shown}")

    if args.output:
        args.output.write_text(json.dumps({"budget": args.budget, "mbit": args.mbit, "rows": rows}, indent=1),
                               encoding="utf-8")


if __name__ == "__main__":
    main()
