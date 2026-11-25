---
title: Split and merge branches in a graph
description: Send a tensor down several branches with Split or fan-out, and join them again with Concatenate, Add, Multiply or Average.
type: how-to
area: graph-editor
audience: user
status: stable
since: 1.0.7
owner: graph-editor
last_reviewed: 2026-11-25
related:
  - connect-nodes.md
  - compile-and-run.md
  - ../../concepts/model-formats.md
redirect_from:
  - ../../usage/branched_graphs.md
---

Use this when a model should not be one straight chain: when parts of a
row need their own layers, or a residual (skip) connection adds a block's
input to its output. **Split** sends one tensor down two branches; a merge
node (**Concatenate**, **Add**, **Multiply**, **Average**) joins branches
again. Each branch has its own layers and its own shapes.

## Before you start

- A graph with a Data Input, a Data Split and a Data Loader, as in
  [Compile and run](compile-and-run.md).

## Give two branches different parts of a row

1. On the canvas, add a **Split** node after the Data Loader (or after any
   layer) and connect its input.
2. In **Properties**, set **Split size** (`split_size`): the entries that go
   to **Output 1**. **Output 2** takes the rest.
3. Optional: set **Dimension** (`dim`), the batched dimension to split.
   `1` is the features of a row (the default), `-1` the last dimension. `0`
   is the batch and is refused.
4. Connect each output to the first layer of its branch.
5. Add a merge node and connect the last layer of each branch to it.
6. Compile. **Properties** shows each branch layer with its own input shape.

Split matches PyTorch's `torch.split(x, [split_size, n - split_size], dim)`.
For example, rows of 6 features, Split size 2, Dimension 1:

```text
Data [6] -> Split -> Output 1 [2] -> Dense A (3) --\
                  -> Output 2 [4] -> Dense B (3) ---> Concatenate [6] -> Dense C (2) -> MSE
```

Properties shows Dense A with input `[2]`, Dense B with `[4]`, and the
Concatenate output `[6]`.

## Send the whole tensor down several branches

A node's output can feed several nodes at once: link the Data Loader (or
any layer) to each branch. Every branch receives the whole tensor, and the
gradients that come back from the branches are summed. Split is only needed
to give the branches different parts of a tensor.

## Combine two inputs in order

Besides the merges, two nodes combine branches in a way that needs their
inputs in order (Input 1, or A, first):

| Node | Per sample | PyTorch |
| --- | --- | --- |
| **Tensor Dot** | A · B, one value per row | `(a * b).sum(-1)` |
| **Tensor Batch MatMul** | A `[rows, inner]` × B `[inner, columns]` → `[rows, columns]` | `torch.bmm(a, b)` |
| **Cross Attention** | Query over Key / Value (see [Cross Attention](../../reference/nodes/attention/cross-attention.md)) | `nn.MultiheadAttention(q, k, v)` |

For example, rows of `[5, 3]`, Split size 2, give A `[2, 3]` and B
`[3, 3]`; Tensor Batch MatMul gives `[2, 3]`; Flatten gives `[6]` for a
Dense.

Inputs are taken by pin, never by the order the links were drawn: linking
Input 2 before Input 1 changes nothing.

## Try the example

`examples/cyxgraph/branching/blobs_multibranch_classifier.cyxgraph` (17
nodes, 19 links) classifies `examples/datasets/metric_blobs.csv` (480 rows,
8 features, 4 classes; made-up data, no download needed). Every batch goes
three ways:

```text
Data Input -> Data Split -> Data Loader -+-> Dense 32 -> ReLU -> Dense 32 -+
                                         |                                 +-> Add -> ReLU -+
                                         +-> Dense 32 (projection) --------+                +-> Concatenate [48]
                                         |                                                  |
                                         +-> Dense 16 -> Tanh ------------------------------+

Concatenate -> Dropout -> Dense 4 -> Cross Entropy -> Adam -> Output
```

The deep branch and the projection are added (a residual connection, so
they have the same width), and the wide Tanh branch is concatenated beside
them.

1. Open the graph from the Asset Browser or **File > Open**.
2. Double-click **Metric blobs** and click **Apply**. If the dialog does
   not find the file, choose `examples/datasets/metric_blobs.csv` in it.
3. Click **Train**.

It usually stops early, before its 30 epochs, at a validation accuracy near
98 %. Your numbers vary with the device.

## How a branched graph trains

Every layer runs once, tensors are kept per output pin, and gradients flow
back to each pin separately. A Split's backward pass concatenates its two
branch gradients. The data steps before the layers (Data Split, Data
Loader) hand the batch on unchanged. The results match PyTorch, including a
Data Loader feeding three branches.

## If something goes wrong

The compiler reports each problem on the node, with the shapes involved:

- Split needs its input connected. **Split size** must be at least 1 and
  less than the split dimension, so both outputs keep entries.
- **Add**, **Multiply** and **Average** need inputs of one shape.
  **Concatenate** inputs must agree on every dimension except `dim`.
- **Tensor Batch MatMul** needs two matrices per sample, with A's columns
  equal to B's rows.
- An output you leave unconnected is fine: its branch passes a zero
  gradient back.
- In a CNN, put Split after **Flatten**: before it the tensors are images
  `[H, W, C]` and Split cuts rows. Fan-out and **Add** do work on images:
  that is a residual block (see
  [Convolutional layers](../../reference/nodes/categories/ml-layers.md#residual-blocks)).

## Export a branched graph

- **Export Model** writes a branched graph as a CyxWiz package or weights
  only; see [Export a trained model](../deployment/export-a-model.md).
- **Code**: the PyCyxWiz export writes the branches, the merges and
  **Split**. The PyTorch export writes the branches and the merges but not
  **Split**: export a graph with a Split as PyCyxWiz. TensorFlow and Keras
  write a straight chain, so a branched graph does not export to them. See
  [What each framework writes](../../concepts/model-formats.md#what-each-framework-writes).

## Related

- [Connect nodes](connect-nodes.md)
- [Compile and run](compile-and-run.md)
- [Model formats](../../concepts/model-formats.md)
