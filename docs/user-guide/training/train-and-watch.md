---
title: Train a model and watch it
description: Start training with Train, follow the run in the Training Dashboard, pause, resume or stop it, and read how it ended.
type: how-to
area: training
audience: user
status: stable
since: 1.0.7
owner: training
last_reviewed: 2026-11-25
related:
  - ../../main-window/training-dashboard.md
  - checkpoints-and-early-stopping.md
  - losses-and-optimizers.md
  - ../graph-editor/compile-and-run.md
  - studio-debugger.md
---

Use this when your graph compiles and you want to train it on this
computer and follow the run. Every control named here is described in the
[Training Dashboard](../../main-window/training-dashboard.md) reference.

## Before you start

- The graph has a Data Input with its data loaded (or loadable), a Data
  Split, a Data Loader, the model's layers, a loss and an optimizer. See
  [Build a model in the graph editor](../graph-editor/index.md).
- **Compile** (F7) reports **Ready to train**, or only warnings. See
  [Compile and run a graph](../graph-editor/compile-and-run.md).
- Optional: run **Local Debug** (F6) first. It runs one sample forward and
  backward and shows what every layer holds, so a wiring mistake shows in
  seconds instead of after an epoch. See
  [The Studio Debugger](studio-debugger.md).
- The run length and checkpoints are set on the **Data Loader** (double-click
  it): **Epochs**, **Batch size** and **Validate every** on its **Training**
  tab; the best checkpoint and early stopping on its **Checkpoints** tab.

## Start training

1. On the Studio bar, click **Train**, press **F5**, or choose **Train >
   Start Training**. When the Script Editor has the focus, F5 runs the
   script instead.
2. The Engine compiles the graph, loads the data, runs the data steps and
   checks for missing values and memory before the first batch. See
   [What Train does first](../graph-editor/compile-and-run.md#what-train-does-first).
   If something stops it, **Cannot start training** says why and, where it
   can, offers the fix.
3. The Training Dashboard opens. Its state reads *Preparing*, then
   *Training*. If another run holds the GPU, it reads *Waiting for the GPU*
   and starts when that run ends.
4. On the canvas, the links pulse while the run trains, and Train's place
   on the Studio bar shows the run: **Epoch 3 / 10** with **Pause** and
   **Stop**.

**Train** is greyed out, with the reason on hover, while it cannot start:
another run or a loading Data Input keeps the graph busy, the graph is not
ready to train, or the project requires a passing Local Debug first. Only
one training run runs at a time: **Train > Start Training** says
"Training is already running. Stop it first."

## Follow the run

The Training Dashboard shows, from top to bottom:

- **The header**: the state, the epoch and batch ("Epoch 6 of 10 · batch
  10 of 12"), where the run computes (CPU or a named GPU, and a fallback to
  the CPU when the GPU failed its check), the dataset with its training,
  validation and test rows and the batch size, and the latest batches' loss
  and accuracy.
- **The cards**: training and validation loss and accuracy of the last
  finished epoch, each with its change and its best epoch. The validation
  loss card says how far it is above the training loss; a large gap means
  the model is overfitting.
- **The charts**: **Loss** and **Accuracy** (and **Metrics** when the run
  reports others), with points inside each epoch and at its end. Hold
  **Ctrl** and turn the mouse wheel over a chart to zoom it; **View** holds
  Auto scale, Follow current epoch, Log loss axis and Smooth curves.

How often the charts get a point inside an epoch is the Data Loader's
**Report every** (Advanced tab, in batches; 0 = only the first and the
last). Validation runs every **Validate every** epochs, and always on the
last epoch.

The Console shows the same run as log lines, one per epoch
("Epoch 3/10: loss=..., acc=...").

## Pause, resume or stop

| To | Do | What happens |
| --- | --- | --- |
| Pause | **Pause** on the dashboard or the Studio bar, or **Train > Pause Training** | The run stops between two batches and waits. The state reads *Paused*. |
| Resume | **Resume** in the same places, or **Train > Resume Training** | The run goes on from the batch where it paused. |
| Stop | **Stop**, or **Train > Stop Training** | The run ends at the next batch and keeps the model as it is then. The state reads *Stopped*. |

A paused run lives in the Engine's memory only. Closing the Engine ends
it; there is no way to continue a local run after the Engine closes or
crashes. To keep training a model later, start a new run from its
checkpoint; see
[Checkpoints, early stopping and resume](checkpoints-and-early-stopping.md).

While a graph trains, the graph and project are locked: loading another
graph, clearing the canvas, and **File > New Project**, **Open Project** and
**Close Project** wait until the run ends. Freezing or unfreezing layers
also waits ("Training is running. Freeze or unfreeze layers once it has
finished.").

## Read how it ended

When the run ends, the dashboard's header says:

- **How it ended**: "Finished: every epoch ran.", "Stopped early: the
  validation loss did not improve for 3 epochs.", "Stopped: you stopped the
  run before its last epoch." or "Failed:" with the reason.
- **The active model**: the model the Engine keeps for testing and export.
  After a finished or early-stopped run with validation rows, it is the
  best validation checkpoint ("the best validation checkpoint, epoch 10
  (validation loss 0.2554, accuracy 92.61%)"). After a stopped run, or with
  no checkpoint, it is "the final state of the run".
- **The held-out test**, when the Data Split has test rows: the accuracy
  and loss on rows the model never saw (the loss and mean absolute error
  for a regression).

The run is added to **Runs**, the table of this session's runs, best
held-out test accuracy first. **Export** saves the metrics or the runs as
CSV files, in the project's `exports` folder unless you pick another.

## Next steps

- Test the model in detail: **Train > Test > Run Test** (F8); the result
  opens in Data Studio's **Results > Test**.
- Tune the run: [Losses and optimizers](losses-and-optimizers.md),
  [Hyperparameter search](hyperparameter-search.md).

## If something goes wrong

**Cannot start training**: the compile, the data load or the missing-value
check found a problem. The report names the node and, where it can, offers
the fix. See [Compile and run a graph](../graph-editor/compile-and-run.md).

**Failed: missing values**: the run met empty cells or NaN while it read the
data. The dashboard offers the same fixes as the report, then **Train
again**. See [Fix missing values before training](../graph-editor/fix-missing-values.md).

**The run moved to the CPU**: the chosen GPU failed its check, and the
header says so. See [What runs where on a GPU](../../concepts/gpu-providers.md).

## Related

- [Training Dashboard](../../main-window/training-dashboard.md)
- [Checkpoints, early stopping and resume](checkpoints-and-early-stopping.md)
- [Losses and optimizers](losses-and-optimizers.md)
- [Compile and run a graph](../graph-editor/compile-and-run.md)
- [The Studio Debugger](studio-debugger.md)
