# Regularization nodes

A regularization node adds a penalty on the model's parameters to the training
loss. Wire it between the loss and the optimizer:

```
Data -> Dense ... -> MSE / Cross Entropy ... --Loss--> L2 Regularization --Loss--> SGD / Adam ...
```

The penalty covers every trainable parameter, weights and biases, as
`model.parameters()` does in PyTorch.

## The nodes

| Node | Settings | Loss term | PyTorch |
| --- | --- | --- | --- |
| L1 Regularization | `lambda` (0.01) | lambda x sum(\|w\|) | `loss + lam * sum(p.abs().sum() for p in model.parameters())` |
| L2 Regularization | `lambda` (0.01) | lambda x sum(w^2) | `loss + lam * sum(p.pow(2).sum() for p in model.parameters())` |
| Elastic Net | `lambda` (0.01), `l1_ratio` (0.5) | lambda x (l1_ratio x sum(\|w\|) + (1 - l1_ratio) x sum(w^2)) | the two terms above, weighted |

Elastic Net with `l1_ratio` 1 is L1 Regularization; with 0 it is L2
Regularization.

## How it trains

Each optimizer step adds the penalty's gradient to every parameter's gradient:
`lambda x sign(w)` for L1 (0 at w = 0, as torch), `2 x lambda x w` for L2. It
is added once per step, after gradient accumulation and before gradient
clipping, which is where torch's `(loss + penalty).backward()` puts it.

- The training loss the Engine reports is the data loss; the penalty is not
  added to it, and validation and test losses never include it.
- L2 and weight decay: with SGD, L2 Regularization with `lambda` equals the
  optimizer's `weight_decay = 2 x lambda`. With Adam the penalty goes through
  the moment estimates, so it is not the same as Adam's `weight_decay`; AdamW's
  `weight_decay` is decoupled from the loss altogether. Using both is allowed:
  they add up.

Example: SGD with learning_rate 0.0625 and L1 Regularization lambda 0.05 moves
every weight 0.003125 towards 0 on each step, on top of the data gradient.

## Rules the compiler checks

- The node must sit on the wire from the training loss to the optimizer: the
  loss's output into its Loss input, its Loss output into the optimizer's Loss
  input.
- One regularization node per loss; Elastic Net combines L1 and L2.
- `lambda` is a number from 0 to 1000000; `l1_ratio` from 0 to 1.

Each refusal is reported on the regularization node.

## Export

The PyTorch export writes the penalty into the training loop
(`loss = loss + 0.01 * sum(p.pow(2).sum() for p in model.parameters())`). The
TensorFlow and Keras exports do not write it yet.

## Checked against PyTorch

`regularization_node_pytorch_parity` first checks that the Engine's model_seed
52 initialisation of a Dense(2 -> 1) model is the fixture's start, then trains
a compiled graph per node (none, L1, L2, Elastic Net) for 6 SGD steps and
compares every parameter with torch's
(`tests/computation_truth/fixtures/regularization_node_pytorch.json`, tolerance
1e-6).
