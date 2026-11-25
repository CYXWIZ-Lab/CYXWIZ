---
redirect_from:
  - ../../../usage/cross_attention.md
---

## How it computes

Every position of the **Query** sequence attends over the **Key / Value**
sequence and comes back as a weighted mix of the values. It is
`torch.nn.MultiheadAttention(query, key, value)` with `batch_first=True`:

```text
softmax(Q K^T / sqrt(d)) V   per head,  Q = query W_q,  K = key W_k,  V = value W_v,  out = concat(heads) W_o
```

Query, Key and Value all have the width `embed_dim`; the Query and Key
lengths may differ, and the output has the Query's shape. Compile checks
that each input is connected and is a `[length, embed_dim]` sequence, and
that Key and Value have the same length.

## Wire it

The inputs come from different branches of the graph. For example, split one
sequence into a "question" part and a "context" part:

```text
Data [8, 4] -> Split (split_size 2) --Output 1 [2, 4]--> Cross Attention.Query
                                    --Output 2 [6, 4]--> Cross Attention.Key
                                                     \-> Cross Attention.Value
Cross Attention [2, 4] -> Flatten -> Dense -> loss
```

or give each branch its own encoder layers (Embedding, Transformer Encoder,
Dense) before the attention. With token ids, split the ids and give each
part its own Embedding. For attention of a sequence over itself use
[Multi-Head Attention](multi-head-attention.md).

## Limits

- Training runs in the graph runtime: the three inputs are matched by pin,
  and the gradient goes back to each input (Key and Value linked from the
  same layer add up), as in PyTorch. Forward output, input gradient and
  every parameter gradient are checked against PyTorch, with Key and Value
  from one branch and from separate branches.
- There is no attention mask: every Query position sees every Key position.
- The code exports refuse it and say so; train it in Studio.
