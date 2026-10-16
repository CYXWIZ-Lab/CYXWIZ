# Linear Attention

Linear Attention is self-attention whose cost grows **linearly** with the
sequence length. Multi-Head Attention compares every position with every other
one (a `length x length` matrix per head), which runs out of memory on long
sequences. Linear Attention drops the softmax and uses a kernel feature map
`phi` instead, so each head can summarise all keys and values once and every
query just reads that summary (Katharopoulos et al. 2020, "Transformers are
RNNs").

Per head, with learned projections `q = x W_q`, `k = x W_k`, `v = x W_v`:

```
S = sum_j phi(k_j) v_j^T        (head_dim x head_dim summary, built once)
z = sum_j phi(k_j)              (normaliser)
out_i = phi(q_i) S / (phi(q_i) . z + eps)
output = concat(heads) W_o
```

There is no softmax and no `1/sqrt(d)` scale. It is a different (cheaper) model
from softmax attention, not an approximation of it: train it from scratch, do
not expect it to reproduce a Multi-Head Attention model's numbers.

## The node

| Pin / setting | Meaning |
| --- | --- |
| Input | `[batch, length, embed_dim]` sequence |
| Output | `[batch, length, embed_dim]`: the input's shape |
| `embed_dim` (512) | feature width of the input (must match it) |
| `num_heads` (8) | heads; `embed_dim` must divide evenly |
| `feature_map` (elu) | `elu`: `phi(x) = elu(x) + 1`, always positive (the paper's choice). `relu`: `phi(x) = max(x, 0)`; sparser, but a query whose features are all zero returns zeros |
| `causal` (off) | each position attends to itself and earlier positions only (for next-token models) |
| `eps` (1e-6, Advanced) | added to each normaliser so it can never divide by zero |
| `use_bias` (on, Advanced) | bias in the Q, K, V and output projections |

Compile checks: the input is a `[length, embed_dim]` sequence, `num_heads`
divides `embed_dim`, `feature_map` is `elu` or `relu`, `eps` is positive.

## Wiring it

Use it where you would use Multi-Head Attention:

```
Data (token ids) -> Embedding -> Positional Encoding -> Linear Attention -> Flatten -> Dense -> loss
```

or on a numeric sequence directly (`Data [length, features]` with
`embed_dim = features`). Stack several, or mix with Layer Norm / Dense layers,
as with any sequence layer.

## Cost

| | Work per head | Memory per head |
| --- | --- | --- |
| Multi-Head Attention | `length^2 x head_dim` | `length^2` |
| Linear Attention | `length x head_dim^2` | `head_dim^2` |
| Linear Attention, causal | `length^2 x head_dim` | `length^2` |

So it pays off when `length` is larger than `head_dim`. Causal mode is
computed as a masked `length x length` product: the same result as the
paper's running sums, but quadratic. A running-sum (prefix) kernel is a
follow-up.

## Notes

- Code export (PyTorch / TensorFlow / Keras / PyCyxWiz) does not support
  Linear Attention yet (no framework has a built-in layer for it) and says so;
  train it in Studio.
- Older graphs that used the `favor+` feature map (random features, never
  implemented) are refused at compile: pick `elu` or `relu`.
- Checked against PyTorch: `linear_attention_pytorch_parity` (forward, input
  gradient and every parameter gradient; elu and relu, causal and not, one to
  four heads, with and without bias). The reference computes the full kernel
  matrix, so it also confirms the summary form gives the same numbers.
