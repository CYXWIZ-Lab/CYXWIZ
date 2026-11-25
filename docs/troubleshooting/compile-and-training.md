---
title: Compile and training problems
description: Fix compile errors, a greyed Train button, the Cannot start training window and Local Debug failures.
type: troubleshooting
area: training
audience: user
status: stable
since: 1.0.7
owner: training
last_reviewed: 2026-11-25
related:
  - ../user-guide/graph-editor/compile-and-run.md
  - ../user-guide/training/studio-debugger.md
  - ../reference/error-messages.md
  - data-loading.md
---

The compile report (**F7**) lists every finding with its node and, where it
has one, a code such as `CW-C-0201`. How to read the report and what Train
does first are in
[Compile and run a graph](../user-guide/graph-editor/compile-and-run.md).
Every code is in [Error messages](../reference/error-messages.md). This page
lists the messages people meet most.

## Train is greyed out

Hover **Train** for the reason:

| Reason | Fix |
| --- | --- |
| "Training is not available yet (no dataset loaded?)" | A Data Input has not loaded its data. Open it and click **Apply**, or press **F7** for the findings. |
| "Train waits while ... It can start when that ends." | Another job is running (a Data Input loading, a pipeline run). Wait for it. |
| "The graph is not ready to train. It needs a dataset input, model layers and a loss." | Press **F7** and fix the errors in the report. |
| "This project requires a passing Local Debug (F6) on this graph" | The Engine's settings require a passing Local Debug. Press **F6** and fix what it finds. |

## "Training did not start: ..."

The **Cannot start training** window opens with "Nothing was trained and
nothing changed in the graph."

**Cause:** Train compiled the graph and found errors, could not load a Data
Input, or found missing values in the table the Data Split reads.

**Fix:** read the reason after the colon and the findings under it. Buttons
on a finding fix the common cases (**Open its dialog**, **Add a Text
Tokenizer**, **Add a Resize**). Missing values are explained in
[Fix missing values before training](../user-guide/graph-editor/fix-missing-values.md).

## "Graph is empty - add nodes to create a model"

**Fix:** add a Data Input, a Data Split, a Data Loader, model layers, a loss
and an optimizer. A task starter graph on the start page has all of them.

## "Graph must have a DataInput node", "... a loss function ...", "... an optimizer ...", "... at least one model layer ..."

Code `[CW-C-0101]`.

**Cause:** the training path misses one of the nodes every training needs:
a Data Input, a loss (MSELoss, CrossEntropyLoss and others), an optimizer
(SGD, Adam, AdamW, RMSprop, Adagrad or NAdam) and at least one layer.

**Fix:** add the missing node and connect it. See
[Connect nodes](../user-guide/graph-editor/connect-nodes.md).

## "Graph contains a cycle - remove circular connections"

**Fix:** find the link that goes back to an earlier node and delete it. A
training graph flows in one direction.

## "batch_size (...) is larger than the train split (... rows)"

Code `[CW-M-0703]`.

**Fix:** lower **Batch size** on the Data Loader's **Training** tab, or give the Train split
more rows on the Data Split. A batch size above half the train rows is a
warning: each epoch then has few steps.

## "Validation split is 0 - training will run without validation metrics"

A warning. Train runs, but the dashboard has no validation curve and early
stopping has nothing to watch. Give the Data Split a validation ratio.

## "Scheduler '...' is not connected to the training optimizer"

Code `[CW-C-0103]`.

**Fix:** link the optimizer's output to the scheduler's **Optimizer** input.
An optimizer takes one scheduler; when its own **lr_schedule** is also set,
the report asks you to set it to `none` or remove the scheduler.

## "Extension node '...' is not installed. Install and approve the extension that provides it, or remove the node."

Code `[CW-X-0601]`.

**Cause:** the graph uses a node from a plugin or Python bundle that is not
installed or not approved here.

**Fix:** install and approve the plugin in the **Plugin Manager**, or delete
the node. See [Troubleshooting extension nodes](../extending/troubleshooting.md).

## "The data steps before the Data Split failed: ..."

**Cause:** Train runs the data steps before the Data Split first, and one
failed. The reason follows the colon.

**Fix:** run **Execute Pipeline** in Data Pipeline mode to see which step
fails and why, fix it, then train. A step stopped by the memory guard is in
[Data loading problems](data-loading.md).

## "The data steps before the Data Split could not be started (a data pipeline run is in progress, or nothing feeds the Split)."

**Fix:** wait for the running pipeline to end, or connect the Data Input
(through its steps) to the Data Split.

## "Local Debug did not run: ..."

**Fix:** the same as for "Training did not start": read the reason and fix
the findings. Local Debug compiles the graph first.

## "Local Debug loss was not finite." or "Local Debug forward output contained non-finite values."

**Cause:** one sample gave NaN or infinite values: often unscaled inputs, a
learning rate far too high, a log of zero, or a wrong loss for the labels.

**Fix:** open the Studio Debugger and find the first layer whose values
turn non-finite. Scale the inputs (Normalize or Standard Scaler), check the
loss matches the task, and lower the learning rate. See
[The Studio Debugger](../user-guide/training/studio-debugger.md).

## "The checkpoint was not loaded: the open graph must compile first"

**Fix:** open the graph the checkpoint was trained with, fix what the
report lists, then load the checkpoint again.

## "Graph replacement locked: ..."

**Cause:** a training, test, pipeline or simulation run is using the graph.
While it runs, the graph and project cannot be replaced or closed.

**Fix:** stop the run or wait for it to end.

## Code says "Graph must have at least one Output node."

**Cause:** the **Code** button runs its own, older check, with its own
wording. A graph the report calls ready can still be refused by Code.

**Fix:** do what Code says before you generate code.

## "Compilation threw an exception: ..."

Code `[CW-C-0901]`.

**Cause:** a defect in the Engine, not in your graph.

**Fix:** save the graph and report it with the message and the log. See
[What to attach to a bug report](crash-dumps-and-logs.md#what-to-attach-to-a-bug-report).

## Related

- [Compile and run a graph](../user-guide/graph-editor/compile-and-run.md)
- [Train a model and watch it](../user-guide/training/train-and-watch.md)
- [Error messages](../reference/error-messages.md)
- [Data loading problems](data-loading.md)
