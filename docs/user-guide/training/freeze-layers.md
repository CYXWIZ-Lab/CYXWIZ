---
title: Freeze layers while fine-tuning
description: Keep loaded layers as they are while the rest of the model trains, one node at a time or with the Freeze commands.
type: how-to
area: training
audience: user
status: experimental
since: 1.0.8
owner: training
last_reviewed: 2026-11-25
related:
  - start-from-pretrained-weights.md
  - fine-tune-language-models.md
---

A frozen layer keeps its weights while the rest of the model trains: it
gets no gradient step, no weight decay and no optimizer memory. Its weights
are still saved in checkpoints and exports. Freezing is how you train a new
head on top of a network that started from
[pretrained weights](start-from-pretrained-weights.md).

## Freeze one layer

1. Select the layer node on the canvas.
2. In **Properties**, group **Fine-tuning**, turn **Frozen** on (off by
   default). Every layer node that holds weights has it.
3. BatchNorm only: set **When frozen**:
   - **Keep its statistics** (the default): the layer normalises with its
     loaded running mean and variance, as in evaluation. The usual choice
     for fine-tuning on a small dataset.
   - **Keep updating statistics**: the weights stay, but the running mean
     and variance keep following the new data. This is what PyTorch does
     when only `requires_grad` is turned off.

On the canvas a frozen node is dimmed and carries a snowflake mark, and the
line under the canvas reads *Trainable N · Frozen M* once anything is
frozen. The node's **As compiled** card shows *Weights: N frozen* instead of
*Learnable parameters*.

The default of a new BatchNorm node's **When frozen** is set in
**Preferences > Training**. Embedding keeps its own **Freeze Loaded
Weights** setting for a table loaded from a file.

## Freeze many layers at once

Use **Nodes > Freeze**, or right-click the canvas > **Actions > Freeze**:

| Command | What it does |
| --- | --- |
| **Freeze selected** / **Unfreeze selected** | The selected nodes that hold weights. |
| **Freeze every layer loaded from the file** | The layers the Data Loader's weights file fills (the *Loaded* rows of the match report). |
| **Freeze all but the head** | Every layer with weights except the last one of the model. |
| **Freeze all but the last N layers...** | Asks for N; layers are counted from the output back, and the last N are unfrozen. |
| **Unfreeze all** | Every layer trains again. |

Each command is one undo step. **Freeze every layer loaded from the file**,
**Freeze all but the head** and **Freeze all but the last N layers** need a
graph that compiles without errors, because the file is matched to the
compiled model and the order of the layers comes from it. None of the
commands runs while training runs.

## Fine-tune in two runs

1. Set the Data Loader's **Start from** to **A weights file**, select
   **Freeze every layer loaded from the file**, and train the new head.
2. Select **Unfreeze all**, set **Start from** to **A checkpoint of an
   earlier run** (the first run's `best` checkpoint), lower the learning
   rate, and train everything.

## Related

- [Start from pretrained weights](start-from-pretrained-weights.md)
- [Fine-tune a language model](fine-tune-language-models.md)
- [Training](index.md)
