#!/usr/bin/env python3
"""PyTorch fixture for the Studio Debugger's saliency (TOFIX178 6.3c).

Three small models in eval mode, each with the output weighting saliency
uses (1 at the target's logit; -1 at a single sigmoid logit for class 0),
the gradient PyTorch computes, and the two per-element values the view
shows:

- image: Conv2d 3->4 (3x3, padding 1), ReLU, flatten, Linear 100->3; the
  gradient at the input; size = the largest |gradient| over a pixel's
  channels, times_input = gradient x input summed over the channels;
- table: Linear 4->5, ReLU, Linear 5->1 (one logit through a sigmoid),
  for class 0 and class 1; size = |gradient|, times_input = gradient x
  value;
- tokens: Embedding 20x4, flatten, Linear 20->3; the gradient at the
  Embedding's output; size = its length per token, times_input =
  gradient . embedding.

Parameters are stored in the backend's layouts: Conv2d weights
[kh,kw,Cin,Cout], Linear [out,in], Embedding [V,D]. An image is [H,W,C,N] and
the backend flattens [H,W,C] row-major, so the flatten here is over
[N,H,W,C]. Layer keys are the Sequential model's ("layer<i>.<name>").

    py -3.12 generate_saliency_fixtures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "saliency_pytorch.json"


def plain(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def image_case() -> dict[str, Any]:
    torch.manual_seed(1783)
    conv_w = torch.randn(4, 3, 3, 3) * 0.4
    conv_b = torch.randn(4) * 0.1
    dense_w = torch.randn(3, 100) * 0.2
    dense_b = torch.randn(3) * 0.1
    x = (torch.randn(1, 3, 5, 5) * 0.8).requires_grad_(True)  # [N,C,H,W]
    h = functional.relu(functional.conv2d(x, conv_w, conv_b, padding=1))
    flat = h.permute(0, 2, 3, 1).reshape(1, -1)  # the backend's [H,W,C] order
    logits = functional.linear(flat, dense_w, dense_b)
    target = 2
    weights = torch.zeros_like(logits)
    weights[0, target] = 1.0
    logits.backward(weights)
    g = x.grad.permute(2, 3, 1, 0)  # [H,W,C,1]
    xv = x.detach().permute(2, 3, 1, 0)
    return {
        "name": "image",
        "kind": "image",
        "rule": "softmax",
        "target": target,
        "layers": {"layer0.weights": plain(conv_w.permute(2, 3, 1, 0)), "layer0.bias": plain(conv_b),
                   "layer3.weight": plain(dense_w), "layer3.bias": plain(dense_b)},
        "input": plain(xv),
        "output": plain(logits),
        "weights": plain(weights),
        "gradient": plain(g),
        "size": plain(g.abs().amax(dim=2).reshape(-1)),
        "times_input": plain((g * xv).sum(dim=2).reshape(-1)),
    }


def table_cases() -> list[dict[str, Any]]:
    torch.manual_seed(1784)
    w1 = torch.randn(5, 4) * 0.6
    b1 = torch.randn(5) * 0.1
    w2 = torch.randn(1, 5) * 0.6
    b2 = torch.randn(1) * 0.1
    x0 = torch.tensor([[0.7, -1.2, 0.3, 2.0]])
    cases = []
    for target in (0, 1):
        x = x0.clone().requires_grad_(True)
        logit = functional.linear(functional.relu(functional.linear(x, w1, b1)), w2, b2)
        weights = torch.full_like(logit, -1.0 if target == 0 else 1.0)
        logit.backward(weights)
        cases.append({
            "name": f"table class {target}",
            "kind": "features",
            "rule": "sigmoid",
            "target": target,
            "layers": {"layer0.weight": plain(w1), "layer0.bias": plain(b1),
                       "layer2.weight": plain(w2), "layer2.bias": plain(b2)},
            "input": plain(x.detach()),
            "output": plain(logit),
            "weights": plain(weights),
            "gradient": plain(x.grad),
            "size": plain(x.grad.abs().reshape(-1)),
            "times_input": plain((x.grad * x.detach()).reshape(-1)),
        })
    return cases


def token_case() -> dict[str, Any]:
    torch.manual_seed(1785)
    table = torch.randn(20, 4) * 0.5
    dense_w = torch.randn(3, 20) * 0.3
    dense_b = torch.randn(3) * 0.1
    ids = torch.tensor([[3, 7, 7, 12, 0]])  # a repeated id and a pad (0)
    embedded = functional.embedding(ids, table).detach().requires_grad_(True)  # [1,T,D]
    logits = functional.linear(embedded.reshape(1, -1), dense_w, dense_b)
    target = 1
    weights = torch.zeros_like(logits)
    weights[0, target] = 1.0
    logits.backward(weights)
    g = embedded.grad
    return {
        "name": "tokens",
        "kind": "tokens",
        "rule": "softmax",
        "target": target,
        "layers": {"layer0.weight": plain(table), "layer2.weight": plain(dense_w), "layer2.bias": plain(dense_b)},
        "input": plain(ids.float()),
        "output": plain(logits),
        "weights": plain(weights),
        "embedded": plain(embedded.detach()),
        "gradient": plain(g),
        "size": plain(g.norm(dim=2).reshape(-1)),
        "times_input": plain((g * embedded.detach()).sum(dim=2).reshape(-1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "generator": "generate_saliency_fixtures.py",
        "torch_version": torch.__version__,
        "tolerance": {"atol": 1e-5, "rtol": 1e-4},
        "layout": {"image": "[H,W,C,N] row-major", "conv_weights": "[kh,kw,Cin,Cout]",
                   "linear_weights": "[out,in]", "embedding": "[V,D]"},
        "cases": [image_case(), *table_cases(), token_case()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1), encoding="utf-8")
    print(f"wrote {args.output} ({len(fixture['cases'])} cases)")


if __name__ == "__main__":
    main()
