---
title: Cross-validation and learning curves
description: Train the graph once per fold to see how far one test can be trusted, and on shares of the rows to see whether more data helps.
type: how-to
area: evaluation
audience: user
status: experimental
since: 1.0.8
owner: evaluation
last_reviewed: 2026-11-25
related:
  - test-a-trained-model.md
  - test-results.md
  - ../training/checkpoints-and-early-stopping.md
redirect_from: []
---

One test split is one draw: a lucky or unlucky split can make a model look
better or worse than it is. **Cross-validation** trains the open graph once
per fold and scores each fold once. A **learning curve** trains it on
growing shares of the rows and shows whether more data would help. Both are
views of Data Studio's **Results** job.

## Before you start

- The graph compiles and trains with **Train** (F5).
- Train is not running: a cross-validation or learning-curve run waits
  while Train runs, and Train waits while one of them runs.

## Cross-validate the graph

1. Choose **Train > Evaluate > Cross-validate** (or **Tools > Evaluation >
   Cross-Validation**). Data Studio opens **Results > Cross-validation**.
2. Set **Folds** (typed, 2 to 20), **Keep class shares** (stratified folds
   for classes) and **Seed**.
3. Click **Run**.

Run goes through Train's own path: the graph is compiled and checked, data
that is not loaded is loaded, the memory question is asked when it applies,
and the graph's preparation steps run. Then, instead of one training, the
prepared table is trained once per fold, with fresh weights each time:

- the rows are the table less the graph's own **test rows** (the ones
  **Train > Test** tests on), so those stay unseen;
- each fold trains for the graph's epochs with its batch size, optimizer
  and device. Early stopping and the best-epoch restore are off, since they
  would pick the epoch by the fold being scored;
- each fold is scored by the test code: accuracy, macro F1 and loss
  (numbers: MAE, RMSE, R²).

It runs in the background (**Background Tasks** shows *Cross-validation:
fold 3 of 5*), with the folds listed as they finish; **Cancel** stops it.
It does not touch the trained model in use, its checkpoints or the Training
Dashboard.

### Read the answer

- The mean with its spread (*Accuracy 91.9% ± 3.7 over 5 folds*) and the
  range.
- A sentence on what to expect on new rows, with the latest test beside
  it: inside the folds' range, or a lucky or unlucky split.
- Bars per fold with the mean, and the fold table.
- A fold that stands out (*Fold 5's loss is high (2.663) though its
  accuracy is not low: a few mistakes made with great confidence*).

**Export CSV...** saves one row per fold.

The runner matches PyTorch training the same graph on the same folds:
every fold's accuracy and loss fall inside the spread of 8 seeds.

## Draw a learning curve

1. Choose **Train > Evaluate > Learning Curve** (or **Tools > Evaluation >
   Learning Curves**). Data Studio opens **Results > Learning curve**.
2. Set **Training rows**, the shares of each fold's training rows (typed:
   *10%, 25%, 50%, 75%, 100%*), **Folds** and **Seed**.
3. Pick the **Score**: accuracy, macro F1 or loss (numbers: MAE, RMSE, R²).
4. Click **Run**.

The graph is trained on each share and scored both on the rows it trained
on and on the fold. The plot shows training against validation with
one-standard-deviation bands; the table has the gap at each size. The
sentences say where validation levels off (or that it still improves, so
more rows would help) and what the gap at full size means: memorising, or
a model that needs more information rather than more rows.

Loss and accuracy per epoch of one run stay in the
[Training Dashboard](../../main-window/training-dashboard.md).

## Keep the settings in the graph

The **Cross-Validation** and **Learning Curves** nodes (Analytics) keep the
settings in the graph, on the data path the way Data Split is. **Add to the
graph** in the view puts one after the Data Input. Data passes through them
unchanged, so Train trains as usual. When the graph has one, **Run** (and
**Train > Evaluate**) uses its settings, and the view says so.

| Node | Settings (defaults) |
| --- | --- |
| Cross-Validation | `folds` (5; 2 to 20), `stratified` (true), `seed` (42), `results_table` (`cross_validation_results`) |
| Learning Curves | `folds` (5), `seed` (42), `train_sizes` (`10%, 25%, 50%, 75%, 100%`), `results_table` (`learning_curve_results`) |

A finished run is published as a table named by `results_table`: one row
per training (`fold`, `share`, `train_rows`, `val_rows`, the validation
scores, a learning curve's training scores, `epochs`, `seconds`). It is the
node's second output (**Results**), so a Plot or Data Output node can use
it, and Data Studio lists it like any table.

## Related

- [Test a trained model](test-a-trained-model.md)
- [Read test results](test-results.md)
- [Plot data](plotting.md)
