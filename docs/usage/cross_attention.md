# Cross Attention

Cross Attention lets one sequence look at another. Every position of the
**Query** sequence attends over the **Key / Value** sequence and comes back as
a weighted mix of the values: the encoder-decoder pattern (a decoder reading
the encoder's output), or any model where one branch should consult another.

It is `torch.nn.MultiheadAttention(query, key, value)` with `batch_first=True`:

```
softmax(Q K^T / sqrt(d)) V   per head,  Q = query W_q,  K = key W_k,  V = value W_v,  out = concat(heads) W_o
```

## The node

| Pin / setting | Meaning |
| --- | --- |
| Query (input) | `[batch, query length, embed_dim]` |
| Key (input) | `[batch, key length, embed_dim]` |
| Value (input) | `[batch, key length, embed_dim]`, the same length as Key. Usually the same tensor as Key: link one layer to both pins. |
| Output | `[batch, query length, embed_dim]`: the Query's shape |
| `embed_dim` (512) | the feature width of Query, Key and Value (all three must have it) |
| `num_heads` (8) | attention heads; `embed_dim` must divide evenly |
| `dropout` (0.0) | dropout on the attention weights while training |
| `use_bias` (on) | bias in the four projections |

The Query and Key lengths may differ. Compile checks the three inputs: each must
be connected, be a `[length, embed_dim]` sequence, and Key and Value must have
the same length.

## Wiring it

The inputs come from different branches of the graph. For example, split one
sequence into a "question" part and a "context" part:

```
Data [8, 4] -> Split (split_size 2) --Output 1 [2, 4]--> Cross Attention.Query
                                    --Output 2 [6, 4]--> Cross Attention.Key
                                                     \-> Cross Attention.Value
Cross Attention [2, 4] -> Flatten -> Dense -> loss
```

or give each branch its own encoder layers (Embedding, Transformer Encoder,
Dense ...) before the attention. With token ids, split the ids and give each
part its own Embedding.

For attention of a sequence over itself use **Multi-Head Attention** (one
input); its Key / Value pins stay unused.

## Notes

- Training runs in the graph runtime: the three inputs are matched by pin, and
  the gradient goes back to each input (Key and Value linked from the same layer
  add up), exactly as in PyTorch.
- Code export (PyTorch / TensorFlow / Keras / PyCyxWiz) does not support Cross
  Attention yet and says so; train it in Studio.
- No attention mask yet: every Query position sees every Key position.
- Checked against PyTorch: `cross_attention_graph_pytorch_parity` (forward,
  input gradient and every parameter gradient, Key = Value from one branch and
  from separate branches).
