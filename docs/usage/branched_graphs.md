# Branched graphs: Split and merge

A model does not have to be one straight chain. **Split** sends one tensor
down two branches, and a merge node (Concatenate, Add, Multiply, Average)
joins branches again. Each branch has its own layers and its own shapes.

## Split

| Setting | Meaning |
| --- | --- |
| Split size (`split_size`) | Entries in **Output 1**. **Output 2** takes the rest. |
| Dimension (`dim`) | Batched dimension to split. `1` = the features of a row, `-1` = the last dimension. `0` is the batch and is refused. |

Split matches PyTorch's `torch.split(x, [split_size, n - split_size], dim)`.

Example: rows of 6 features, Split size 2, Dimension 1:

```
Data [6] -> Split -> Output 1 [2] -> Dense A (3) --\
                  -> Output 2 [4] -> Dense B (3) ---> Concatenate [6] -> Dense C (2) -> MSE
```

Properties shows each Dense with its own input (`[2]` and `[4]`), and the
Concatenate output is `[6]`.

## Two-input operations

Besides the merges, two nodes combine branches in a way that needs their
inputs in order (Input 1 / A first):

| Node | Per sample | PyTorch |
| --- | --- | --- |
| Tensor Dot | A · B, one value per row | `(a * b).sum(-1)` |
| Tensor Batch MatMul | A `[rows, inner]` × B `[inner, columns]` → `[rows, columns]` | `torch.bmm(a, b)` |
| Cross Attention | Query over Key / Value (see [cross_attention.md](cross_attention.md)) | `nn.MultiheadAttention(q, k, v)` |

Example: rows of `[5, 3]`, Split size 2 → A `[2, 3]` and B `[3, 3]`;
Tensor Batch MatMul gives `[2, 3]`; Flatten → `[6]` → Dense.

Inputs are taken by pin, never by the order the links were drawn: linking
Input 2 before Input 1 changes nothing.

## Rules the compiler checks

- Split needs its input connected. Split size must be at least 1 and less
  than the split dimension, so both outputs keep entries.
- Add, Multiply and Average need inputs of one shape. Concatenate inputs must
  agree on every dimension except `dim`.
- Tensor Batch MatMul needs two matrices per sample, with A's columns equal
  to B's rows.
- An output you leave unconnected is fine: its branch passes a zero gradient
  back.
- In a CNN, Split runs after Flatten: before it the tensors are images
  `[H, W, C]` and Split cuts rows. Fan-out and Add do work on images: that
  is a residual block (see [cnn_layers.md](cnn_layers.md#residual-blocks)).

Each problem is reported on the node, with the shapes involved.

## Fan-out without Split

A node's output can feed several nodes at once: link the Data Loader (or any
layer) to each branch. Every branch receives the whole tensor, and the
gradients that come back from the branches are summed. Split is only needed
to give the branches different parts of a tensor.

## Training

A branched graph trains through the graph executable model: every layer
runs once, tensors are kept per output pin, and gradients flow back to each
pin separately. A Split's backward pass concatenates its two branch
gradients. The data steps before the layers (Data Split, Data Loader) hand
the batch on unchanged. The results match PyTorch (`branch_graph_pytorch_parity`,
including a Data Loader feeding three branches).

## Example: multi-branch classifier

`examples/cyxgraph/branching/blobs_multibranch_classifier.cyxgraph`
(17 nodes, 19 links) classifies `examples/datasets/metric_blobs.csv`
(480 rows, 8 features, 4 classes). Every batch goes three ways:

```
Data Input -> Data Split -> Data Loader -+-> Dense 32 -> ReLU -> Dense 32 -+
                                         |                                 +-> Add -> ReLU -+
                                         +-> Dense 32 (projection) --------+                +-> Concatenate [48]
                                         |                                                  |
                                         +-> Dense 16 -> Tanh ------------------------------+

Concatenate -> Dropout -> Dense 4 -> Cross Entropy -> Adam -> Output
```

The deep branch and the projection are added (a residual connection, so they
have the same width), and the wide Tanh branch is concatenated beside them.
Open it, double-click **Metric blobs** and **Apply**, then **Train**.
Measured 2026-10-15 (GTX 1050 Ti, CUDA): early stop after 11 of 30 epochs
in 8 s, training accuracy 97.9%, validation accuracy 98.6%.

Code export: the pycyxwiz export writes the split; the PyTorch, TensorFlow
and Keras exports still write a straight chain, so a branched graph does not
export to them yet.
