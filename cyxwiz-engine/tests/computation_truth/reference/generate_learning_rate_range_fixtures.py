"""Truth for the learning rate range test (TOFIX129 D11), for
test_learning_rate_range.

The test as fastai's lr_find / Smith's range test runs it in PyTorch: the rate
lr_i = start * (end / start) ** (i / (steps - 1)) for step i, one optimizer
step per batch, the smoothed loss avg_i = b * loss_i + (1 - b) * avg_{i-1}
(avg_0 = loss_0), and a stop once a loss is not finite or avg_i > factor *
(lowest avg before it). The suggestions: the lowest smoothed loss's rate, a
tenth of it, and the steepest fall: argmin of numpy.gradient(avg,
log10(lr)) up to the lowest point.

Each case records the raw losses torch saw, so the C++ test replays them
through LearningRateRangeTest and must reproduce the rates, the smoothing, the
stop and the suggestions exactly; the losses depend only on torch.

  mnist_like   a small CNN on mock 12x12 images (seed 7), SGD momentum 0.9,
               1e-5 .. 1000, 100 steps: stops early when the loss blows up.
  mlp_short    an MLP on mock rows, SGD, 1e-4 .. 0.5, 30 steps: runs all.

Run with torch 2.10, numpy 2.3:
  python generate_learning_rate_range_fixtures.py
Writes ../fixtures/learning_rate_range_pytorch.json.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def range_test(model, optimizer, batches, start, end, steps, beta=0.05, factor=4.0):
    rates, losses, smooth = [], [], []
    avg, best = 0.0, math.inf
    stop = "steps"
    for i in range(steps):
        lr = start * (end / start) ** (i / (steps - 1))
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y, loss_fn = batches(i)
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        v = loss.item()
        avg = beta * v + (1 - beta) * avg if i else v
        rates.append(lr)
        losses.append(v)
        smooth.append(avg)
        if not math.isfinite(v):
            stop = "not_finite"
            break
        if avg > factor * best:
            stop = "factor"
            break
        best = min(best, avg)
    s = np.array(smooth)
    lg = np.log10(np.array(rates))
    finite = np.isfinite(s)
    s, lg = s[finite], lg[finite]
    slope = np.gradient(s, lg)
    lowest = int(np.argmin(s))
    steepest = int(np.argmin(slope[: lowest + 1])) if lowest > 0 else 0
    return {"start": start, "end": end, "steps": steps, "smoothing": beta, "stop_factor": factor,
            "rates": rates, "losses": losses, "smoothed": smooth, "stop": stop,
            "lowest": lowest, "steepest": steepest, "lowest_rate": rates[lowest],
            "steepest_rate": rates[steepest], "lowest_over_ten": rates[lowest] / 10,
            "slopes": slope.tolist()}


def mnist_like():
    torch.manual_seed(7)
    x = torch.rand(512, 1, 12, 12)
    y = (x.mean(dim=(1, 2, 3)) * 37).long() % 10
    model = nn.Sequential(nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(200, 10))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7, momentum=0.9)
    g = torch.Generator().manual_seed(1)

    def batches(i):
        idx = torch.randint(0, 512, (32,), generator=g)
        return x[idx], y[idx], F.cross_entropy

    return range_test(model, optimizer, batches, 1e-5, 1000.0, 100)


def mlp_short():
    torch.manual_seed(11)
    x = torch.randn(256, 6)
    y = x @ torch.randn(6, 1) + 0.1 * torch.randn(256, 1)
    model = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    def batches(i):
        b = (i * 32) % 256
        return x[b:b + 32], y[b:b + 32], F.mse_loss

    return range_test(model, optimizer, batches, 1e-4, 0.5, 30)


fixture = {"torch": torch.__version__, "numpy": np.__version__,
           "cases": {"mnist_like": mnist_like(), "mlp_short": mlp_short()}}
out = Path(__file__).resolve().parent.parent / "fixtures" / "learning_rate_range_pytorch.json"
out.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
for name, case in fixture["cases"].items():
    print(name, len(case["rates"]), case["stop"], case["lowest_rate"], case["steepest_rate"])
