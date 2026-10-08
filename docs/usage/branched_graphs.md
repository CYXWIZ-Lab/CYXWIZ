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

## Rules the compiler checks

- Split needs its input connected. Split size must be at least 1 and less
  than the split dimension, so both outputs keep entries.
- Add, Multiply and Average need inputs of one shape. Concatenate inputs must
  agree on every dimension except `dim`.
- An output you leave unconnected is fine: its branch passes a zero gradient
  back.
- In a CNN, split after Flatten. Before Flatten the tensors are images
  `[H, W, C]`, and Split runs on rows.

Each problem is reported on the node, with the shapes involved.

## Training

A branched graph trains through the graph executable model: every layer
runs once, tensors are kept per output pin, and gradients flow back to each
pin separately. A Split's backward pass concatenates its two branch
gradients. The results match PyTorch (`branch_graph_pytorch_parity`).

Code export: the pycyxwiz export writes the split; the PyTorch, TensorFlow
and Keras exports still write a straight chain, so a branched graph does not
export to them yet.
