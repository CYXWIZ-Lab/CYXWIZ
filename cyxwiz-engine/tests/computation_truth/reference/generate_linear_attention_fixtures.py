#!/usr/bin/env python3
"""PyTorch fixtures for the Linear Attention node (TOFIX140 Group C): a
sequence input [N, T, E] goes through linear (kernel) self-attention
(Katharopoulos et al. 2020), then Flatten and a Linear head.

Per head, with phi = elu(x) + 1 or relu(x) and no softmax or 1/sqrt(d) scale:
    out_i = phi(q_i) . sum_j phi(k_j) v_j^T / (phi(q_i) . sum_j phi(k_j) + eps)
where j runs over every position, or j <= i when causal. The reference below
computes it the quadratic way (the full T x T kernel matrix), independent of
the Engine's O(T d^2) summary form, so the two only agree if both are right.

Each case stores the input, the attention parameters in the Engine's layout
(W_q / W_k / W_v / W_o [out, in] as torch.nn.Linear, b_*), the Linear head, a
fixed upstream gradient, the forward output and every gradient.

    py -3.12 generate_linear_attention_fixtures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "linear_attention_pytorch.json"
EPS = 1e-6


def plain(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


class LinearAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, heads: int, feature_map: str, causal: bool, bias: bool) -> None:
        super().__init__()
        self.heads = heads
        self.feature_map = feature_map
        self.causal = causal
        self.proj = torch.nn.ModuleDict({n: torch.nn.Linear(embed_dim, embed_dim, bias=bias) for n in "qkvo"})

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) if self.feature_map == "relu" else torch.nn.functional.elu(x) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, t, e = x.shape
        d = e // self.heads

        def heads(y: torch.Tensor) -> torch.Tensor:
            return y.reshape(n, t, self.heads, d).transpose(1, 2)  # [N, H, T, d]

        q = self.phi(heads(self.proj["q"](x)))
        k = self.phi(heads(self.proj["k"](x)))
        v = heads(self.proj["v"](x))
        kernel = q @ k.transpose(-1, -2)  # [N, H, T, T]
        if self.causal:
            kernel = kernel * torch.tril(torch.ones(t, t))
        out = (kernel @ v) / (kernel.sum(-1, keepdim=True) + EPS)
        return self.proj["o"](out.transpose(1, 2).reshape(n, t, e))

    def parameters_in_engine_layout(self, grads: bool) -> dict[str, Any]:
        result = {}
        for name, linear in self.proj.items():
            result["W_" + name] = plain(linear.weight.grad if grads else linear.weight)
            if linear.bias is not None:
                result["b_" + name] = plain(linear.bias.grad if grads else linear.bias)
        return result


def case(name: str, x: torch.Tensor, heads: int, feature_map: str, causal: bool, bias: bool) -> dict[str, Any]:
    n, t, e = x.shape
    attention = LinearAttention(e, heads, feature_map, causal, bias)
    head = torch.nn.Linear(t * e, 3)
    x = x.clone().requires_grad_(True)
    output = head(attention(x).flatten(1))
    grad = torch.randn(output.shape)
    output.backward(grad)
    return {
        "name": name,
        "embed_dim": e,
        "num_heads": heads,
        "feature_map": feature_map,
        "causal": causal,
        "use_bias": bias,
        "eps": EPS,
        "input": plain(x),
        "output": plain(output),
        "grad_output": plain(grad),
        "grad_input": plain(x.grad),
        "attention": attention.parameters_in_engine_layout(False),
        "attention_grad": attention.parameters_in_engine_layout(True),
        "head": {"weight": plain(head.weight), "bias": plain(head.bias),
                 "grad_weight": plain(head.weight.grad), "grad_bias": plain(head.bias.grad)},
        "tolerance": {"absolute": 1e-5, "relative": 1e-4},
    }


def build() -> list[dict[str, Any]]:
    torch.manual_seed(140)
    return [
        case("elu_two_heads", torch.randn(3, 5, 4), 2, "elu", False, True),
        case("elu_causal", torch.randn(3, 5, 4), 2, "elu", True, True),
        case("relu_one_head_no_bias", torch.randn(2, 6, 4), 1, "relu", False, False),
        case("relu_causal_four_heads", torch.randn(2, 4, 8), 4, "relu", True, True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generator": "generate_linear_attention_fixtures.py",
        "torch_version": torch.__version__,
        "cases": build(),
    }
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()
