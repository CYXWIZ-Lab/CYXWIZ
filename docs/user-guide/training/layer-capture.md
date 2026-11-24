---
title: Capture what every layer holds
description: Turn on layer capture in the Studio Debugger to keep each layer's activations, weights and gradients from Local Debug and training runs.
type: how-to
area: training
audience: user
status: experimental
since: 1.0.8
owner: training
last_reviewed: 2026-11-25
related:
  - studio-debugger.md
  - inspect-layers.md
  - where-the-model-looks.md
---

Local Debug and training can keep what every layer held: its input and
output, its weights (with BatchNorm's running statistics as kept state) and
the gradients of the weights. The layer views of the Studio Debugger read
these captures. Capture is off until you turn it on.

## Turn on layer capture

1. In the Studio Debugger's toolbar, select **Layer capture**. The button
   names what the next runs keep, for example *Layer capture: every 5
   epochs*.
2. Set **When**, **What** and **On** (the table that follows). The panel
   gives the size of one snapshot from the compiled model on the canvas
   (the input and every layer's output for the captured samples, the
   parameters with their kept state, and the gradients), plus the total for
   a training run of the graph's epochs.
3. Run **Local Debug** (F6) or **Train** (F5). A run in progress keeps the
   settings it started with.
4. Optional: select **Save as my default** to write the session's settings
   to Preferences, or **Use my defaults** to drop the session's changes.

Changes in the toolbar apply to this session's next Local Debug and
training runs. To set the defaults for every project, use **Preferences >
Studio Debugger**; Preferences applies and saves at once (the
`studio_debugger` section of the Engine's settings file).

| Setting | Choices |
| --- | --- |
| **When** | **Off** (Local Debug and training run exactly as without it), **Local Debug runs**, or **Training, every N epochs** (N = 5 to start). Training also captures after the last epoch, an early stop's included, and Local Debug runs are captured too. |
| **What**, for every layer | Activations, parameters, gradients (all ticked to start). |
| **On** | **The picked sample** (the debugger's **Sample**) or **A batch of 8**: the picked item and the seven after it (the picked item alone when the data ends before them). |
| **Keep the captures of the last N runs** (Preferences) | 10 to start. Older runs keep their report; their captures are deleted. |

**Preferences > Studio Debugger > Layer views** also holds the defaults of
the layer views: feature-map colour and order, the Where it looks method,
Top images, Embedding and Unit weights choices. See
[Inspect the layers](inspect-layers.md) and
[See where the model looks](where-the-model-looks.md).

## What a training snapshot is

A training snapshot runs the sample once, forward and backward, with no
weight update and with the model in evaluation mode (BatchNorm on its
running statistics, no dropout). Training therefore comes out exactly the
same with capture on or off. The snapshot's gradients are those of that
evaluation pass, not of a training batch.

## Where captures are kept

A capture stays on the compute device while the run is open, up to 256 MB
per step; tensors past that are left out and counted. It is also saved with
the run in the project's debug run store, under
`debug_runs/studio/<run>/layers/`: Local Debug as `local_debug`, training
snapshots as `epoch_0005` and so on. On the ResNet example one step takes
25 MB for all 36 layers.

A training run with snapshots also saves `training.json` beside them,
rewritten at each snapshot: every epoch so far (training loss and accuracy,
or the MAE for a regression, and the validation loss and metric when
validation ran), the graph's name, and the snapshot items' names, labels
and classes.

## Training runs and their snapshots

**Overview > Runs** lists the saved runs, newest first: Local Debug runs and
training runs that kept layer snapshots. A training run's line names its
graph, its epochs, its best validation (and at which epoch) and its snapshot
epochs, for example *20 epochs · best validation 92.97 % (epoch 18) · 4
layer snapshots (epochs 5, 10, 15, 20)*. A Local Debug run's line says
whether it passed and its issues; **Details** gives its traces, events,
graph, device, transfers and where it is saved. Click a run to show it.

**Overview > Compare**, for a shown training run, is its snapshot table:
each snapshot's epoch, training loss, validation loss, validation accuracy
(or MAE) and size on disk, with **Open in Model > Layers**. For a Local
Debug run, Compare sets the run against the latest one.

## Limits

- Data that cannot hand over an item by number (token tagging sequences,
  sparse features in training) has no training snapshots; the log says
  why.
- Snapshots saved before the epochs were kept show *not recorded* in
  Compare.

## Related

- [Check a graph with Local Debug](studio-debugger.md)
- [Inspect the layers](inspect-layers.md)
- [See where the model looks](where-the-model-looks.md)
- [Training](index.md)
