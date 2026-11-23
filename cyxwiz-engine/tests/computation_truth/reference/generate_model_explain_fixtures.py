"""Truth for the Model window's Explain tab (TOFIX129 D11), for
test_model_explain.

Two models with fixed random weights, written in the backend's
layouts (Conv2d weights [kh, kw, Cin, Cout], Linear [out, in]); inputs and
maps in the Engine's [H, W, C] row order.

  cnn       8x8x1 -> Conv 4 (3x3, same) -> ReLU -> Conv 6 (3x3, valid) -> ReLU
            -> global average pool -> Dense 3. Grad-CAM at "Conv 6" (its
            output, before the ReLU, as the Engine's Conv module gives it):
            alpha_c = mean over h, w of d score / d A, ReLU(sum alpha_c A_c),
            resized to 8x8 by F.interpolate(bilinear, align_corners=False).
            The seed is the first from 3 whose map has kept and dropped parts.
  residual  6 features -> Dense a (5) and Dense b (5) -> Add -> ReLU ->
            Dense 3: a graph with branches (no Grad-CAM); seed 3.

For both: the scores (logits), the predicted class, saliency d score / d x
for that class, and Integrated Gradients from a zero baseline in 64 midpoint
steps: (x - x0) * mean_k grad(x0 + (k + 0.5) / 64 (x - x0)), with the scores
at the input and the baseline (the completeness check).

Run with torch 2.10:
  python generate_model_explain_fixtures.py
Writes ../fixtures/model_explain_pytorch.json.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

STEPS = 64


def plain(t):
    return {"shape": list(t.shape), "values": [float(v) for v in t.detach().reshape(-1)]}


def conv_weight(t):
    return plain(t.detach().permute(2, 3, 1, 0))


def hwc(t):
    # [1, C, H, W] -> [H, W, C] rows
    return [float(v) for v in t.detach()[0].permute(1, 2, 0).reshape(-1)]


def integrated(score_of, x, baseline):
    alphas = (torch.arange(STEPS, dtype=torch.float32) + 0.5) / STEPS
    path = baseline + alphas.view(-1, *[1] * (x.dim() - 1)) * (x - baseline)
    path.requires_grad_(True)
    score_of(path).sum().backward()
    return (x - baseline) * path.grad.mean(dim=0, keepdim=True)


def cnn_case(seed):
    torch.manual_seed(seed)
    w1 = torch.randn(4, 1, 3, 3) * 0.5
    b1 = torch.randn(4) * 0.1
    w2 = torch.randn(6, 4, 3, 3) * 0.3
    b2 = torch.randn(6) * 0.1
    wd = torch.randn(3, 6) * 0.5
    bd = torch.randn(3) * 0.1
    x = torch.rand(1, 1, 8, 8)

    def forward(t, keep=None):
        a = F.conv2d(F.relu(F.conv2d(t, w1, b1, padding=1)), w2, b2)
        if keep is not None:
            a.retain_grad()
            keep["A"] = a
        return F.linear(F.relu(a).mean(dim=(2, 3)), wd, bd)

    scores = forward(x)
    target = int(scores.argmax())
    xs = x.clone().requires_grad_(True)
    forward(xs)[0, target].backward()
    saliency = xs.grad
    ig = integrated(lambda p: forward(p)[:, target], x, torch.zeros_like(x))
    keep = {}
    forward(x.clone().requires_grad_(True), keep)[0, target].backward()
    a, g = keep["A"], keep["A"].grad
    cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(dim=1, keepdim=True))
    up = F.interpolate(cam, size=(8, 8), mode="bilinear", align_corners=False)
    return {
        "seed": seed,
        "input_shape": [8, 8, 1],
        "input": hwc(x),
        "weights": {"Conv 4": {"weights": conv_weight(w1), "bias": plain(b1)},
                    "Conv 6": {"weights": conv_weight(w2), "bias": plain(b2)},
                    "Head": {"weight": plain(wd), "bias": plain(bd)}},
        "scores": [float(v) for v in scores.detach()[0]],
        "target": target,
        "saliency": hwc(saliency),
        "integrated": hwc(ig),
        "score": float(scores[0, target]),
        "baseline_score": float(forward(torch.zeros_like(x))[0, target]),
        "grad_cam_layer": [float(v) for v in cam.detach()[0, 0].reshape(-1)],
        "grad_cam": [float(v) for v in up.detach()[0, 0].reshape(-1)],
    }


def residual_case():
    wa, ba = torch.randn(5, 6) * 0.5, torch.randn(5) * 0.1
    wb, bb = torch.randn(5, 6) * 0.5, torch.randn(5) * 0.1
    wh, bh = torch.randn(3, 5) * 0.5, torch.randn(3) * 0.1
    x = torch.randn(1, 6)

    def forward(t):
        return F.linear(F.relu(F.linear(t, wa, ba) + F.linear(t, wb, bb)), wh, bh)

    scores = forward(x)
    target = int(scores.argmax())
    xs = x.clone().requires_grad_(True)
    forward(xs)[0, target].backward()
    ig = integrated(lambda p: forward(p)[:, target], x, torch.zeros_like(x))
    return {
        "input_shape": [6],
        "input": [float(v) for v in x[0]],
        "weights": {"Dense a": {"weight": plain(wa), "bias": plain(ba)},
                    "Dense b": {"weight": plain(wb), "bias": plain(bb)},
                    "Head": {"weight": plain(wh), "bias": plain(bh)}},
        "scores": [float(v) for v in scores.detach()[0]],
        "target": target,
        "saliency": [float(v) for v in xs.grad[0]],
        "integrated": [float(v) for v in ig[0]],
        "score": float(scores[0, target]),
        "baseline_score": float(forward(torch.zeros_like(x))[0, target]),
    }


def cnn_with_a_map():
    # The first seed whose map is not all zero and not all positive: the
    # ReLU of Grad-CAM must both keep and drop parts of it.
    for seed in range(3, 200):
        case = cnn_case(seed)
        cam = case["grad_cam_layer"]
        if max(cam) > 0.01 and min(cam) == 0.0:
            return case
    raise SystemExit("no seed gives a Grad-CAM map with kept and dropped parts")


cnn = cnn_with_a_map()
torch.manual_seed(3)
fixture = {"torch": torch.__version__, "steps": STEPS, "cases": {"cnn": cnn, "residual": residual_case()}}
out = Path(__file__).resolve().parent.parent / "fixtures" / "model_explain_pytorch.json"
out.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
for name, case in fixture["cases"].items():
    print(name, "target", case["target"], "ig sum", sum(case["integrated"]), "moved", case["score"] - case["baseline_score"])
