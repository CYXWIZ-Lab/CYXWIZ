---
title: Checkpoints, early stopping and resume
description: Keep the best model of a run, stop when validation stops improving, find the checkpoint files, and continue training from a checkpoint.
type: how-to
area: training
audience: user
status: stable
since: 1.0.7
owner: training
last_reviewed: 2026-11-25
related:
  - train-and-watch.md
  - start-from-pretrained-weights.md
  - ../../main-window/training-dashboard.md
  - ../../concepts/seeds-and-reproducibility.md
---

Use this when you want a run to keep its best model rather than its last
one, to stop by itself once the model stops improving, or to continue
training a model you trained before. All three are set on the graph's
**Data Loader** node, on the **Checkpoints** tab of its dialog
(double-click the node, or **Open Dialog...** in Properties).

## How the best checkpoint works

During a run the Engine measures the validation loss every **Validate
every** epochs (Data Loader, **Training** tab; the last epoch always
validates). Each time the validation loss is lower than every value before
it, the Engine saves the model's weights as the run's **best** checkpoint,
replacing the previous one.

When the run finishes or stops early, the Engine loads the best checkpoint
back into the model. That model is the **active model**: Run Test (F8),
export and the Model window use it, and the Training Dashboard names it
("the best validation checkpoint, epoch 3 (validation loss 0.5120, ...)").
So a run that overfits after epoch 3 still leaves you the epoch-3 model.

Limits:

- The best checkpoint needs validation rows. With no validation part in the
  Data Split, no checkpoint is written and the active model is the final
  state of the run.
- When you **Stop** a run, the best checkpoint is kept on disk but not
  loaded back: the active model is the model as it was when you stopped.
- A run keeps one checkpoint, `best`. It does not keep one per epoch.

## Keep the best model of a run

1. Double-click the graph's **Data Loader** and open the **Checkpoints**
   tab.
2. Check **Save the best validation checkpoint** (on by default).
3. Make sure the Data Split gives a validation part (its validation share
   above 0).
4. Click **Apply**, then **Train** (F5).

## Stop when validation stops improving

Early stopping ends a run before its last epoch when the validation loss
has not improved for a number of validation checks in a row.

1. On the Data Loader's **Checkpoints** tab, set **Early stop patience**:
   the number of validation checks without a new best validation loss
   before the run stops. The default is 5; 0 turns early stopping off.
2. Click **Apply** and train.

With **Validate every** at 1, patience counts epochs. With **Validate
every** at 2, a patience of 3 means three checks, that is six epochs.

When it triggers, the run ends after that epoch. The Training Dashboard
reads *Early stopped*, "Stopped early: the validation loss did not improve
for 3 epochs.", and Run details shows the stop code
`validation_loss_plateau_patience_3`. The epoch line keeps the plan, for
example "7 of 10 epochs". An early stop is a controlled end, not a failure:
the best checkpoint is loaded back as for a finished run.

A plugin can also end a run early; Run details then shows
`plugin_requested_early_stop`.

To lower the learning rate when the validation loss stalls, instead of
stopping, use the **Reduce LR** scheduler node; see
[Learning-rate schedulers](../../reference/nodes/categories/training.md#learning-rate-schedulers).

## Where the checkpoint files are

| Setting | Checkpoints go to |
| --- | --- |
| **Checkpoint folder** empty, a project open | `<project>/checkpoints/<run>/best/` |
| **Checkpoint folder** empty, no project | `.cyxwiz/checkpoints/<run>/best/` in the folder the Engine was started from |
| **Checkpoint folder** set | `<folder>/<run>/best/` |

Give **Checkpoint folder** as a full path: training reads a relative
folder from the folder the Engine was started from, not from the project.

`<run>` is the run's id, `train-` followed by a number, for example
`train-1791984271862`. Run details in the Training Dashboard shows the
checkpoint's epoch, step and path.

A checkpoint folder holds:

| File | Holds |
| --- | --- |
| `metadata.json` | The epoch and step, the training and validation loss and accuracy (MAE and RMSE for a regression), the learning rate, the history of every metric, the time it was saved, the seeds, the Data Split it was trained on, and the checkpoint it started from, if any |
| `model/` | One `.bin` file per weight tensor and `manifest.json`, which maps each file to its layer's parameter name |
| The tokenizer | For a language model, the tokenizer the run used |

The best checkpoint stores the model's weights, not the optimizer's state.

## Continue training from a checkpoint

A local run cannot be resumed after it ends or after the Engine closes:
**Pause** and **Resume** work only while the run is still in the Engine's
memory. To keep training a model you trained before, start a new run from
its checkpoint (a warm start):

1. On the Data Loader's **Checkpoints** tab, set **Start from** to **A
   checkpoint of an earlier run**.
2. Click **Browse...** and pick the checkpoint folder, for example
   `checkpoints/train-1791984271862/best` in the project.
3. Click **Apply** and train.

Every layer starts from the checkpoint's weights and keeps training on this
run's data. The optimizer, the learning-rate schedule, the epochs and the
random seed start fresh, so the run is a new run, not the old one
continued. Layer names and shapes, and for a language model the tokenizer,
must match the checkpoint, or the run stops with the reason and the model
is left untouched. The new run's checkpoints record the folder they started
from.

**Start from** can also be **A weights file** (a PyTorch `.pth` / `.pt`
state-dict or a `.safetensors` file); see
[Start from pretrained weights](start-from-pretrained-weights.md). Use one
of the two, not both.

## Test a checkpoint without training

With the graph the checkpoint was trained with open and its Data Inputs
applied, choose **Train > Test > Load Checkpoint for Testing...** and pick
the checkpoint folder (or the run's folder). It loads the weights into the
graph's model for testing; it refuses while a training or testing task
runs. The Training Dashboard then
reads *Model loaded*; a loaded checkpoint is not a run and gets no row in
Runs. Then run **Train > Test > Run Test** (F8).

## Changes

- Changed in 1.0.8: **Start from** offers **A weights file**.

## Related

- [Train a model and watch it](train-and-watch.md)
- [Start from pretrained weights](start-from-pretrained-weights.md)
- [Training Dashboard](../../main-window/training-dashboard.md)
- [Seeds and reproducibility](../../concepts/seeds-and-reproducibility.md)
