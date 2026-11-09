"""Truth for pycyxwiz.model (TOFIX129 D11), for test_python_model.py.

The models a Python Sequential can hold, with fixed random weights in the
backend's layouts (Linear [out, in]):

  mlp        6 features -> Linear 16 -> ReLU -> Linear 3. The scores (logits),
             the predicted class, saliency d score / d x for it, Integrated
             Gradients from a zero baseline in 64 midpoint steps (as
             generate_model_explain_fixtures.py), and the summary: each
             module's parameters and multiply-adds (torch.utils.flop_counter's
             FLOPs / 2).
  attention  one self-attention module, length 10, width 16, 4 heads: its
             parameters and multiply-adds (the MATH attention kernel, as
             generate_model_summary_fixtures.py counts it).
  range      the mlp's range test as torch runs it: 32 rows of x and y, batches
             of 8 in order (round again at the end), mean squared error, plain
             SGD whose rate rises from 1e-3 to 3.0 over 24 steps
             (start x (end / start)^(step / (steps - 1))). Each step's loss.

Run with torch 2.10:
  python generate_model_python_fixtures.py
Writes ../fixtures/model_python_torch.json.
"""
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

STEPS = 64


def plain(t):
    return {"shape": list(t.shape), "values": [float(v) for v in t.detach().reshape(-1)]}


def flops(module, *inputs):
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
        module(*inputs)
    return counter.get_total_flops()


def integrated(score_of, x, baseline):
    alphas = (torch.arange(STEPS, dtype=torch.float32) + 0.5) / STEPS
    path = baseline + alphas.view(-1, *[1] * (x.dim() - 1)) * (x - baseline)
    path.requires_grad_(True)
    score_of(path).sum().backward()
    return (x - baseline) * path.grad.mean(dim=0, keepdim=True)


def mlp_model(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))


def weights_of(model):
    return {"layer0.weight": plain(model[0].weight), "layer0.bias": plain(model[0].bias),
            "layer2.weight": plain(model[2].weight), "layer2.bias": plain(model[2].bias)}


def mlp_case():
    model = mlp_model(7)
    x = torch.randn(1, 6)
    scores = model(x)
    target = int(scores.argmax())
    xs = x.clone().requires_grad_(True)
    model(xs)[0, target].backward()
    ig = integrated(lambda p: model(p)[:, target], x, torch.zeros_like(x))
    one = torch.zeros(1, 6)
    return {
        "seed": 7,
        "input": [float(v) for v in x.reshape(-1)],
        "weights": weights_of(model),
        "scores": [float(v) for v in scores.detach().reshape(-1)],
        "target": target,
        "saliency": [float(v) for v in xs.grad.reshape(-1)],
        "integrated": [float(v) for v in ig.reshape(-1)],
        "score": float(scores.detach()[0, target]),
        "baseline_score": float(model(torch.zeros_like(x))[0, target]),
        "summary": {
            "parameters": [sum(p.numel() for p in m.parameters()) for m in model],
            "multiply_adds": [flops(model[0], one) // 2, 0, flops(model[2], torch.zeros(1, 16)) // 2],
        },
    }


def attention_case():
    torch.manual_seed(11)
    attention = torch.nn.MultiheadAttention(16, 4, batch_first=True)
    x = torch.zeros(1, 10, 16)
    return {"length": 10, "width": 16, "heads": 4,
            "parameters": sum(p.numel() for p in attention.parameters()),
            "multiply_adds": flops(lambda t: attention(t, t, t, need_weights=False), x) // 2}


def range_case():
    model = mlp_model(5)
    start_weights = weights_of(model)
    torch.manual_seed(13)
    x = torch.randn(32, 6)
    y = torch.randn(32, 3)
    start, end, steps, batch = 1e-3, 3.0, 24, 8
    optimizer = torch.optim.SGD(model.parameters(), lr=start)
    losses, rates = [], []
    at = 0
    for step in range(steps):
        rate = start * (end / start) ** (step / (steps - 1))
        for group in optimizer.param_groups:
            group["lr"] = rate
        xb, yb = x[at:at + batch], y[at:at + batch]
        at = 0 if at + batch >= 32 else at + batch
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(xb), yb)
        loss.backward()
        optimizer.step()
        rates.append(rate)
        losses.append(float(loss))
    return {"x": plain(x), "y": plain(y), "weights": start_weights, "start": start, "end": end, "steps": steps,
            "batch": batch, "rates": rates, "losses": losses}


fixture = {"torch": torch.__version__, "steps": STEPS,
           "cases": {"mlp": mlp_case(), "attention": attention_case(), "range": range_case()}}
out = Path(__file__).resolve().parent.parent / "fixtures" / "model_python_torch.json"
out.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
mlp = fixture["cases"]["mlp"]
print("mlp target", mlp["target"], "ig sum", sum(mlp["integrated"]), "moved", mlp["score"] - mlp["baseline_score"])
print("attention", fixture["cases"]["attention"])
print("range losses", [round(v, 4) for v in fixture["cases"]["range"]["losses"]])
