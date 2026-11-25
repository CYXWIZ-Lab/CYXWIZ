---
title: Measure feature importance
description: Find out how much each input column matters to the trained model, with permutation, drop-column or weight-based importance.
type: how-to
area: evaluation
audience: user
status: stable
since: 1.0.7
owner: evaluation
last_reviewed: 2026-11-25
related:
  - test-a-trained-model.md
  - ../../main-window/model-window.md
redirect_from:
  - feature-importance.md
---

**Tools > Evaluation > Feature Importance** measures how much each input
column matters to the active model: the model you trained in this session,
or a checkpoint loaded with **Load checkpoint...** (the same model Test and
inference use). Any model works, a layer stack or a graph with branches
(Split, Add, Concatenate).

## Before you start

- Train the graph, or load a checkpoint with **Load checkpoint...** in the
  window. With no model the window says *Nothing to measure yet*.
- Load the data you want to measure on: a Data Input you applied.

## Steps

1. Open **Tools > Evaluation > Feature Importance**. The model card shows
   the active model and where it came from ("Trained this session - 9
   layers").
2. Under the data settings, pick:
   - **Dataset**: a loaded table, for example *Metric blobs - Data Input -
     480 rows - 9 cols*;
   - **Features**: the columns the model reads, all selected at first ("8
     selected");
   - **Target column**: the label, suggested from the training run
     ("class").
3. Pick the method (see the table that follows) and, for Permutation and
   Drop-Column, the **Scoring**: *Accuracy* for classification, *MSE* for
   regression. Permutation also has **Repeats** (10 by default; more repeats,
   less variance).
4. Click **Compute importance**. It runs in the background.
5. Read the result: the method and the baseline score, then a **Chart** tab
   (one bar per feature, largest on top, scaled 0 to 1) and a **Ranking**
   tab.
6. Optional: **Export** saves the ranking as CSV (rank, feature, importance
   and, for permutation, its standard deviation).

| Method | What it does | Models |
| --- | --- | --- |
| Permutation Importance | Shuffles one column at a time (**Repeats** times) and measures how much the score drops. | Any |
| Drop-Column Importance | Sets one column to zero at a time and measures the drop. | Any |
| Weight-Based | The mean absolute weight of each input in the first layer. | Layer stacks only |

For a graph with branches, Weight-Based stays off with the reason on
hover: its inputs feed several first layers, so there is no single first
layer to read.

## Example

The 17-node branching classifier
`examples/cyxgraph/branching/blobs_multibranch_classifier.cyxgraph`, trained 5
epochs on `examples/datasets/metric_blobs.csv`: Permutation Importance, 10
repeats, accuracy. Baseline 0.9771; ranking f3, f6, f1, f2, f5, f0, f4, f7.

## If something goes wrong

- **The result disappeared**: a result belongs to the model it was
  computed for. Training again or loading another checkpoint clears it.
- **Compute importance waits**: the model runs one job at a time. While
  Train, a test, **Model > Explain** or generation from this session's
  model is running, the window says which one and waits for you to run it
  again.

## Related

- [Test a trained model](test-a-trained-model.md)
- [The Model window](../../main-window/model-window.md): Explain shows which
  pixels made a prediction
- [Plot data](plotting.md): the **Feature importance** plot type draws
  importance tables from any source
