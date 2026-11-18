# The Training Dashboard

The Training Dashboard follows a model's training run: where it is, how the
curves go, how it ended and which model the Engine holds after it. Open it
with **Train > Training Dashboard** or the **Training** button in the
sidebar; it also opens by itself when you press **Train**. It sits beside the
Console by default and can have a window of its own (right-click its tab:
**Move to its own window**; see [Windows of their own](windows.md)).

## The header

The first card says what the run is doing, in plain words.

- **State**: *Training*, *Paused*, *Waiting for the GPU* (another run holds
  it; this one starts when that one ends), *Preparing*, *Completed*,
  *Early stopped*, *Stopped*, *Failed*, *Preparation failed*, *Model loaded*
  (a checkpoint loaded for testing) or *Idle*. Next to it: "Epoch 6 of 10 ·
  batch 10 of 12" while it trains, "All 10 epochs in 2.2s" when it is done,
  or "7 of 10 epochs in 1.5s, stopped during epoch 8".
- **Pause** / **Resume** and **Stop** while a run trains. Stop ends the run
  at the next batch and keeps the model as it is. **More** holds Export
  metrics (CSV) and Clear all.
- The bar shows the share of all epochs done (hover for the percentage).
- **Where it runs**: "Runs on the CPU (ArrayFire CPU) · compatibility policy",
  or a GPU with its name ("CUDA (GeForce GTX 1050 Ti) · strict policy"). When
  the chosen GPU failed its check and the run moved to the CPU, the line says
  so. Hover it for the backend and device codes.
- **The data**: the dataset, its training, validation and test rows and the
  batch size ("digits_8x8 · 1,433 training, 176 validation, 188 test rows ·
  batches of 128").
- Warnings from the run (a data transfer, a device fallback, memory) show
  under these lines.
- **While it trains**: "Latest batches: loss 0.5287 · accuracy 90.70%
  (batch 10 of 12, 204 batches/s)", the mean of the batches since the last
  chart point.
- **When it ends**:
  - how it ended: "Finished: every epoch ran.", "Stopped early: the
    validation loss did not improve for 3 epochs.", "Stopped: you stopped
    the run before its last epoch." or "Failed:" with the reason;
  - the active model, the one the Engine keeps for testing and export: "the
    best validation checkpoint, epoch 10 (validation loss 0.2554, accuracy
    92.61%)", or "the final state of the run" when no checkpoint was kept;
  - the held-out test, when the split has test rows: "Held-out test:
    accuracy 94.15%, loss 0.2220 (188 rows the model never saw)". A
    regression run shows the loss and the mean absolute error.

## Run details

**Run details**, under the header, holds the codes and figures behind those
lines. It is folded at first; the line beside it says what it holds. When the
dashboard is wide its groups sit side by side, otherwise one under the other.

- **The run**: the stop code (`completed_all_epochs`,
  `validation_loss_plateau_patience_3`, ...), the checkpoint's epoch and step
  and its path, the batches (samples in each, how often the optimizer
  updates, how often the charts get a point), the last batch's loss while it
  trains, and the engine tasks running beside the run.
- **Where it ran**: everything the run recorded about its device: residency,
  the requested and effective backends and their qualification, the device's
  identity, the execution check and policy, placement, transfers and
  synchronizations with their reasons, the placement of each stage, and the
  latest pin-memory transfer.
- **Data**: for a graph with preparation steps, the prepared data's status,
  file, manifest, cache key and folder, the memory estimate and what the
  process used, and the cache limit. The Data preparation card below the
  charts keeps the steps, the cache size and the cache actions.

Long paths and keys wrap inside the dashboard instead of running past its
edge.

## The cards

The cards show the last finished epoch: training loss and accuracy, then
validation loss and accuracy, each with its change since the epoch before and
its best epoch. While the next epoch runs, the training cards add the latest
batches as a second figure. The validation loss card says how far it is from
the training loss at the same epoch. The last card is the time left while the
run trains (with the samples per second) and the total time once it is done.
