---
title: Data Split and Data Loader dialogs
description: The dialogs that cut the data into train, validation and test parts and set how training reads it in batches.
type: reference
area: graph-editor
audience: user
status: stable
since: 1.0.7
owner: graph-editor
last_reviewed: 2026-11-25
related:
  - node-dialogs.md
  - ../user-guide/training/checkpoints-and-early-stopping.md
  - ../user-guide/training/generation-previews.md
redirect_from: []
---

The **Data Split** and **Data Loader** nodes sit between the data and the
model: the split decides which rows train, validate and test the model, the
loader how training reads them. The buttons every dialog shares are in
[Node dialogs](node-dialogs.md).

## Data Split

| Part | What it shows or sets |
| --- | --- |
| The bar | The three parts with their rows from the graph's data (*train 80% about 42,145, val 5,268, test 5,268*) |
| **Train**, **Validation**, **Test** | The three shares and their sum; Apply scales them to 1.00 |
| **Seed** | The shuffled order the parts are cut from |
| **Stratified: keep each class's share in every part** | Keeps each class's share in every part |
| *From the graph* | The dataset and its samples, with **Refresh from graph** |
| **Details** | The resolved manifest: origins, schemas, policy, source ids, leakage checks |

An old graph using Data Boundary v1 shows a caution and **Migrate graph to
Dataset v2**. The split the model trained with is kept with the model and
its checkpoints, so a test reads the part it never saw; see
[Test a trained model](../user-guide/evaluation/test-a-trained-model.md).

## Data Loader

The Data Loader is the one place batching is set. Six tabs:

| Tab | What it sets |
| --- | --- |
| **Training** | Epochs, validate every, batch size (with about how many batches an epoch has), seed, model seed, shuffle, drop the last short batch |
| **Batches** | Gradient accumulation, prefetch |
| **Balance** | Class balancing for training batches: mode, target per class, seed |
| **Checkpoints** | Best validation checkpoint, early stop patience, checkpoint folder, start from an earlier checkpoint; see [Checkpoints, early stopping and resume](../user-guide/training/checkpoints-and-early-stopping.md) |
| **Generation** | Samples a language model writes during training: prompts, chat template, stop texts, decoding, when; see [Generation previews](../user-guide/training/generation-previews.md) |
| **Advanced** | Worker threads, pin memory, report interval |

## Related

- [Node dialogs](node-dialogs.md)
- [Checkpoints, early stopping and resume](../user-guide/training/checkpoints-and-early-stopping.md)
- [Losses, optimizers and class imbalance](../user-guide/training/losses-and-optimizers.md)
