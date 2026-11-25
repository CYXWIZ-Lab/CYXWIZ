---
title: Losses and optimizers
description: Pick the loss and optimizer nodes for a model, wire them, and set their key values and learning-rate schedule.
type: how-to
area: training
audience: user
status: stable
since: 1.0.7
owner: training
last_reviewed: 2026-11-25
related:
  - train-and-watch.md
  - class-imbalance.md
  - checkpoints-and-early-stopping.md
  - ../../reference/nodes/categories/training.md
  - ../../reference/nodes/categories/regularization.md
---

Use this when you wire the end of a model: the loss that scores its output
and the optimizer that updates its weights. Both are nodes in the
**Training** group of the Nodes panel. A graph needs both to train; without them Compile reports "Graph must have a loss function
(MSELoss, CrossEntropyLoss, etc.)" or "Graph must have an optimizer (SGD,
Adam, AdamW, RMSprop, Adagrad, or NAdam)".

## Wire the loss and the optimizer

```text
Data Loader --Data--> layers ... --> Loss <--Labels-- Data Loader
                                      |
                                    Loss
                                      v
                                  Optimizer --State--> (optional) LR scheduler
```

1. Link the last layer's output to the loss's first input (**Logits**,
   **Predictions** or **Log Probabilities**, by loss).
2. Link the Data Loader's **Labels** output to the loss's second input
   (**Labels** or **Targets**).
3. Link the loss's **Loss** output to the optimizer's **Loss** input.
4. Optional: put one regularization node between the loss and the
   optimizer ([Regularization nodes](../../reference/nodes/categories/regularization.md)), or a
   learning-rate scheduler on the optimizer's **State** output
   ([Learning-rate schedulers](../../reference/nodes/categories/training.md#learning-rate-schedulers)).

The batch size and the epochs are set on the Data Loader, not on the
optimizer; a batch size on the optimizer is ignored, and a warning says so
at training start.

## Pick a loss

| Task | Loss node | Model output it expects |
| --- | --- | --- |
| Classes (one label per row, image or clip) | **CrossEntropy / Token CE** | Raw scores (logits), one per class |
| Classes, a few rare ones matter most | **Focal Loss** | Logits |
| Classes, from log-probabilities | **NLL Loss** | Log-probabilities |
| Two classes or several yes/no labels | **BCE with Logits** | One logit per label |
| Same, from probabilities | **BCE Loss** | Probabilities in 0 to 1 |
| Numbers (regression) | **MSE Loss**; **L1 Loss** or **Smooth L1 Loss** / **Huber Loss** when outliers would dominate | One value per target |
| Masks (segmentation) | **Soft Dice Loss**, **Tversky Loss**, **Jaccard / IoU Loss** | Probability masks, same shape as the targets |
| A label per token (tagging, language models) | **CrossEntropy / Token CE** | Token logits |
| Embeddings, pairs and triplets | **Contrastive Loss**, **Cosine Embedding Loss**, **Triplet Loss** | See [Metric learning](metric-learning.md) |

CrossEntropy and Focal Loss apply the softmax themselves. A **Softmax**
node in a model trained with them is skipped during training (the log says "Softmax
layer skipped"), so you can keep one for prediction without harming the
gradients. Likewise prefer **BCE with Logits** over a Sigmoid followed by
**BCE Loss**: it is the numerically stable form.

Every loss has **Reduction**: `mean` (default), `sum` or `none`. The
settings that matter most:

| Loss | Setting | Default | Effect |
| --- | --- | --- | --- |
| CrossEntropy | **Label Smoothing** | 0 | Trains on softened targets; 0.1 is common against overconfidence. |
| CrossEntropy, NLL | **Ignore Index** | -100 | Labels with this value (padding tokens) do not count. |
| CrossEntropy | **Class Weight**, **Class Weights** | none | Weights per class; see [Train on unbalanced classes](class-imbalance.md). |
| Focal Loss | `alpha`, `gamma` | 0.25, 2.0 | `gamma` lowers the weight of rows the model already classifies well; `alpha` scales the loss. |
| BCE with Logits | **Pos Weight** | 1.0 | Weight of the positive class. |
| BCE with Logits | **Label Smoothing** | 0 | Targets become `t x (1 - e) + e / 2`. |
| Smooth L1, Huber | `beta` | 1.0 | Where the loss changes from squared to linear. |
| Tversky | `alpha`, `beta` | 0.5, 0.5 | Penalty of false positives and false negatives. |

## Unbalanced classes

When one class has far more rows than another, weight the loss
(**Class Weight** on CrossEntropy, **Pos Weight** on BCE with Logits, or
**Focal Loss**) or balance the Data Loader's batches. See
[Train on unbalanced classes](class-imbalance.md).

## Pick an optimizer

| Node | Default learning rate | Use it for |
| --- | --- | --- |
| **Adam** | 0.001 | A safe first choice for most models. |
| **AdamW** | 0.001 | Adam with decoupled weight decay (`weight_decay` 0.01); the usual choice for transformers. |
| **SGD** | 0.01 | Convolutional networks with a schedule; `momentum` 0.9, optional **Nesterov** and **Weight Decay**. |
| **RMSprop** | 0.001 | Recurrent networks. |
| **Adagrad** | 0.01 | Sparse features. |
| **NAdam** | 0.002 | Adam with Nesterov momentum. |

The settings shared by every optimizer, under **Schedule** in Properties:

| Setting | Default | Effect |
| --- | --- | --- |
| `lr_schedule` | `none` | The learning rate over the optimizer's updates: `none`, `warmup_cosine`, `warmup_linear`, `warmup_constant` or `one_cycle`. |
| `warmup_ratio` | 0.02 | The share of the updates spent warming up (with `one_cycle`, the share spent rising; 0.3 is usual). |
| `min_lr_ratio` | 0.1 | Where the decay ends, as a fraction of the learning rate. |
| `grad_clip_norm` | 0 (off) | Clips the gradients by their global norm before each update; see [Gradient clipping](../../reference/nodes/categories/training.md#gradient-clipping). |
| `weight_decay_exclude` | `none` | AdamW and SGD: parameters that skip weight decay (`norms_and_biases`; AdamW also `norms_biases_embeddings`). |

An optimizer's `lr_schedule` and a scheduler node both set the learning
rate, so Compile refuses the pair: set `lr_schedule` to `none` or remove the
scheduler.

To jump to the graph's optimizer, choose **Train > Optimizer Settings...**:
it selects the optimizer on the canvas and shows its settings in
Properties. A graph without one says "No optimizer in this graph".

To find a good learning rate or other values automatically, see
[Hyperparameter search](hyperparameter-search.md).

## Limits

- Training runs in 32-bit floating point (float32) on every device. Half,
  mixed and double precision are not available.
- One scheduler per optimizer and one regularization node per loss.
- Sample weights (a weight per row) are not supported; Compile warns that
  such settings are ignored.

## Changes

- Changed in 1.0.8: `lr_schedule` `one_cycle`.

## Related

- [Train a model and watch it](train-and-watch.md)
- [Train on unbalanced classes](class-imbalance.md)
- [Checkpoints, early stopping and resume](checkpoints-and-early-stopping.md)
- [Learning-rate schedulers](../../reference/nodes/categories/training.md#learning-rate-schedulers)
- [Regularization nodes](../../reference/nodes/categories/regularization.md)
- [Gradient clipping](../../reference/nodes/categories/training.md#gradient-clipping)
