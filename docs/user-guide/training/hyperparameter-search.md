---
title: Search for better hyperparameters
description: Train the graph several times with different property values in Train > Hyperparameter Search and keep the values that score best.
type: how-to
area: training
audience: user
status: experimental
since: 1.0.8
owner: training
last_reviewed: 2026-11-25
related:
  - neural-architecture-search.md
  - losses-and-optimizers.md
  - ../graph-editor/compile-and-run.md
redirect_from:
  - ../../usage/hyperparameter_search.md
---

**Train > Hyperparameter Search** trains the graph on the canvas several
times with different values for the properties you pick, and keeps the
values that scored best. Each trial trains the way **Train** does: the same
prepared data and Data Split, a fresh model, its own scratch checkpoints
(removed after). The model in use and the Training Dashboard's runs stay as
they are.

## Before you start

- The graph compiles (**Compile**, F7) and its Data Input is loaded.
- Train, a cross-validation or a learning curve is not running: a search
  does not start while one of them runs, and Train waits while a search
  runs.

## Pick what to vary

1. In **Train > Hyperparameter Search**, select **Add from the graph**. The
   list comes from the compiled graph and shows, per node, the properties a
   trial can change (the table that follows).
2. Pick the properties. Each starts with a range around its value on the
   graph.
3. Optional: change a range in the **Range** column.

| Node | Properties |
| --- | --- |
| Layers | units, filters, out channels, hidden size, number of layers, embedding size, heads, dropout rate |
| The optimizer | learning rate, weight decay, beta1 / beta2, momentum, warm-up and minimum learning-rate ratios |
| The scheduler | its steps, rates and periods |
| DataLoader | batch size, epochs |

The last layer (it sets the outputs the loss expects, such as the 10
classes of MNIST) is listed but not searched.

| Values | Starting range | Example |
| --- | --- | --- |
| Log scale (rates) | A tenth to ten times the value | Learning rate 0.001: 0.0001 - 0.01 |
| Whole number (counts) | A quarter to twice, in quarter steps | Units 128: 32 - 256, step 32 |
| Choices (batch size) | Half, the same, double | 64: 32, 64, 128 |
| Decimal (fractions) | Around the value | Dropout 0.2: 0.1 - 0.4 |

## Choose how to search

| Method | How it picks values | Best for |
| --- | --- | --- |
| **Grid** | Every combination; a range is cut into 10 steps | One or two properties with few values |
| **Random** | Draws values within the ranges | A first look at many properties |
| **Bayesian** | A few random trials, then each next trial where the results so far say the score should improve most (a Gaussian process) | Good values in fewer trials |

Then set **Trials** (Random and Bayesian), **Stop after** (hours; the best so
far is kept) and **Score by** (validation loss, validation accuracy, or
validation MAE for numbers).

The DataLoader's own settings that are not varied apply to every trial: its
epochs, batch size and early stopping. The trials split the data as the
graph's Data Split does.

## Run the search

1. Select **Start search**. Train's checks and data preparation run first;
   the data is loaded and prepared once for all the trials. If a check stops
   it, the reason shows under the button and the compile report opens.
2. Watch the status line: the trial and epoch running, the time per trial
   and the time left. **Stop** drops the trial that is running and keeps the
   ones done.
3. When the search ends, the line says why: *Finished* (all the trials ran),
   *Stopped*, *Time limit reached*, or *Three trials in a row failed*; when
   it ended early, how many trials ran (*5 of 20 trials*).
4. Under **Best so far**, select **Apply best to the graph** to set those
   values on the canvas (Ctrl+Z takes them back). Train the graph to get the
   model.
5. Optional: **Export CSV** saves a row per trial.

The score per trial shows as bars (the best one light, a failed one short
and red), with a row per trial: its values, validation loss, accuracy (or
MAE), time and status. Hover a failed trial for its reason. A failed trial
does not stop the search; three failures in a row do.

## Set your defaults

Method, Trials, Stop after and Score by start from **Preferences > Search**
(out of the box: Random, 20 trials, 1 hour, validation loss). They are
saved with your Engine settings, apply to every project, and the panel takes
them again when you change them there (not while a search runs). **Reset**
puts the Engine defaults back.

## Example

Open `examples/cyxgraph/mnist_mlp.cyxgraph` and load `mnist_test.csv`
(10,000 rows; `python examples/cyxgraph/model_tools/get_mnist.py` downloads
it) in its Data Input. Vary **Adam learning rate** (0.0001 -
0.01), **Random**, 3 trials. Each trial takes a few seconds; in one run
the best (learning rate 0.0087) reached validation loss 0.309 and 94.4 %
accuracy, against 87.6 % at 0.0001.

## Related

- [Search for a better architecture](neural-architecture-search.md)
- [Choose a loss and an optimizer](losses-and-optimizers.md)
- [Compile and run a graph](../graph-editor/compile-and-run.md)
- [Training](index.md)
