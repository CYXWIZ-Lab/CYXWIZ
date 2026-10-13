#!/usr/bin/env python3
"""PyTorch fixtures for the Cross Attention node (TOFIX140 Group C): a sequence
input [N, T, E] is split into branches; Cross Attention attends from the Query
branch over the Key / Value branch(es) (torch.nn.MultiheadAttention(q, k, v),
batch_first, no mask, no dropout), then Flatten and a Linear head.

Each case stores the input, the attention parameters in the Engine's layout
(W_q / W_k / W_v / W_o [out, in] = in_proj_weight chunks and out_proj.weight;
b_* = in_proj_bias chunks and out_proj.bias), the Linear weight / bias, a fixed
upstream gradient, the forward output and every gradient PyTorch computes.

    py -3.12 generate_cross_attention_graph_fixtures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "fixtures" / "cross_attention_graph_pytorch.json"
)
EMBED_DIM = 4
HEADS = 2


def plain(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def attention_parameters(mha: torch.nn.MultiheadAttention, grads: bool) -> dict[str, Any]:
    weight = mha.in_proj_weight.grad if grads else mha.in_proj_weight
    bias = mha.in_proj_bias.grad if grads else mha.in_proj_bias
    out_weight = mha.out_proj.weight.grad if grads else mha.out_proj.weight
    out_bias = mha.out_proj.bias.grad if grads else mha.out_proj.bias
    e = EMBED_DIM
    return {
        "W_q": plain(weight[0:e]), "W_k": plain(weight[e:2 * e]), "W_v": plain(weight[2 * e:3 * e]),
        "b_q": plain(bias[0:e]), "b_k": plain(bias[e:2 * e]), "b_v": plain(bias[2 * e:3 * e]),
        "W_o": plain(out_weight), "b_o": plain(out_bias),
    }


def case(name: str, splits: list[int], x: torch.Tensor, kv_separate: bool) -> dict[str, Any]:
    mha = torch.nn.MultiheadAttention(EMBED_DIM, HEADS, batch_first=True)
    query_length = splits[0]
    head = torch.nn.Linear(query_length * EMBED_DIM, 3)
    x = x.clone().requires_grad_(True)
    first, rest = torch.split(x, [splits[0], x.shape[1] - splits[0]], 1)
    if kv_separate:
        key, value = torch.split(rest, [splits[1], rest.shape[1] - splits[1]], 1)
    else:
        key = value = rest
    attended, _ = mha(first, key, value, need_weights=False)
    output = head(attended.flatten(1))
    grad = torch.randn(output.shape)
    output.backward(grad)
    return {
        "name": name,
        "splits": splits,
        "kv_separate": kv_separate,
        "embed_dim": EMBED_DIM,
        "num_heads": HEADS,
        "input": plain(x),
        "output": plain(output),
        "grad_output": plain(grad),
        "grad_input": plain(x.grad),
        "attention": attention_parameters(mha, False),
        "attention_grad": attention_parameters(mha, True),
        "head": {"weight": plain(head.weight), "bias": plain(head.bias),
                 "grad_weight": plain(head.weight.grad), "grad_bias": plain(head.bias.grad)},
        "tolerance": {"absolute": 1e-5, "relative": 1e-4},
    }


def build() -> list[dict[str, Any]]:
    torch.manual_seed(141)
    return [
        # Query = rows 0-1, Key = Value = rows 2-5 (one branch to both pins).
        case("kv_shared", [2], torch.randn(3, 6, EMBED_DIM), False),
        # Query = rows 0-1, Key = rows 2-4, Value = rows 5-7 (three branches).
        case("kv_separate", [2, 3], torch.randn(3, 8, EMBED_DIM), True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generator": "generate_cross_attention_graph_fixtures.py",
        "torch_version": torch.__version__,
        "cases": build(),
    }
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()
