---
redirect_from:
  - ../../../usage/linear_attention.md
---

## How it computes

Multi-Head Attention compares every position with every other one (a
`length x length` matrix per head), which runs out of memory on long
sequences. Linear Attention drops the softmax and uses a kernel feature map
`phi` instead, so each head summarises all keys and values once and every
query reads that summary (Katharopoulos et al. 2020, "Transformers are
RNNs"). Per head, with learned projections `q = x W_q`, `k = x W_k`,
`v = x W_v`:

```text
S = sum_j phi(k_j) v_j^T        (head_dim x head_dim summary, built once)
z = sum_j phi(k_j)              (normaliser)
out_i = phi(q_i) S / (phi(q_i) . z + eps)
output = concat(heads) W_o
```

There is no softmax and no `1/sqrt(d)` scale.

- **Feature map** `elu` is `phi(x) = elu(x) + 1`, always positive (the
  paper's choice). `relu` is `phi(x) = max(x, 0)`: sparser, but a query whose
  features are all zero returns zeros. Any other value is refused at
  compile.
- **Causal** lets each position attend to itself and earlier positions only,
  for next-token models.

## Wire it

Use it where you would use Multi-Head Attention:

```text
Data (token ids) -> Embedding -> Positional Encoding -> Linear Attention -> Flatten -> Dense -> loss
```

or on a numeric sequence directly (`Data [length, features]` with
`embed_dim = features`). Stack several, or mix them with LayerNorm and Dense
layers, as with any sequence layer.

## Cost

| | Work per head | Memory per head |
| --- | --- | --- |
| Multi-Head Attention | `length^2 x head_dim` | `length^2` |
| Linear Attention | `length x head_dim^2` | `head_dim^2` |
| Linear Attention, causal | `length^2 x head_dim` | `length^2` |

It pays off when `length` is larger than `head_dim`. Causal mode is computed
as a masked `length x length` product: the same result as the paper's
running sums, but quadratic.

## Limits

- The code exports refuse it (no framework has a built-in layer for it) and
  say so; train it in Studio.
- Forward output, input gradient and every parameter gradient are checked
  against PyTorch for `elu` and `relu`, causal and not, one to four heads,
  with and without bias. The reference computes the full kernel matrix, so
  it also confirms that the summary form gives the same numbers.
