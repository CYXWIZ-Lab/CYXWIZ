---
title: Read test results
description: Data Studio's Results job and its Test view, section by section, and how to keep a test as a node.
type: reference
area: evaluation
audience: user
status: experimental
since: 1.0.8
owner: evaluation
last_reviewed: 2026-11-25
related:
  - test-a-trained-model.md
  - threshold-and-calibration.md
  - cross-validation-and-learning-curves.md
  - ../../main-window/model-window.md
redirect_from: []
---

A model's results live next to its data, in Data Studio's **Results** job.
After a test, Data Studio opens **Results > Test** by itself; **Train >
Test > View Test Results** opens it later. The **Tools > Evaluation** items
(Confusion Matrix, ROC Curve / AUC, Precision-Recall Curve) open the same
view.

## The Results job

Results has four views: **Test**, **Cross-validation**, **Learning curve**
and **Generations**. The job's line counts what there is to show (*test: 72
rows · 2 preview runs*, and any evaluation nodes in the graph).
Cross-validation and Learning curve are described in
[Cross-validation and learning curves](cross-validation-and-learning-curves.md).

## What the Test view shows

**Showing** picks the test:

| Choice | What it shows |
| --- | --- |
| **Latest test** | The last **Train > Test** run. Its rows are the `test_predictions` table. |
| **Kept: *name*** | A test kept with **Keep as node...**, stored in the project's `results/tests` folder. |
| **A table...** | Any table read as a test. Pick the table, **Classes** or **Numbers**, the **True** and **Predicted** columns and, for the curves, the **Scores**: one `score_<class>` column per class, or one column for two classes (then say whose score it is). Columns named like `label`, `prediction` or `score_<class>` are picked for you. Rows missing a value are left out and counted. |

A model scored in a script, or an imported CSV of predictions, gets every
section through **A table...**.

The header shows the headline figure (accuracy for classification, 1-NN
accuracy for metric learning, MAE for regression, token accuracy for
sequence models), how many rows were tested and how long it took, and the
other figures as chips (loss, macro and weighted F1, AUC; RMSE and R² for
numbers). A table read as a test has no loss or time, so they are not
shown.

## Sections

| Section | What it shows |
| --- | --- |
| Overview | Macro precision, recall and F1, weighted F1; the best and worst class and classes never predicted. A regression adds its **Bias**: the mean of predicted minus true (positive: predicts high). |
| Confusion matrix | Counts, or the share of each true class; the largest confusion. A cell's tooltip says *x of y (z%)*. |
| Classes | Precision, recall, F1, found, false alarms and support per class; filter and sort. |
| Errors | How many rows are wrong and the commonest mistake, the mistakes by pair (true -> predicted, with counts), the most confident ones first, with their images for an image model. |
| Curves | ROC and precision-recall (one line per class against the rest for three or more classes), AUC and AP. The precision-recall curve is drawn as steps, so its area is AP; a ring marks the threshold. |
| Threshold | The decision rule; see [Choose a threshold and calibrate](threshold-and-calibration.md). |
| Calibration | Whether the model's confidence can be taken at its word; see the same page. |
| Plots | Regression only: predicted vs actual, residuals vs predicted, residual histogram. |
| Predictions | Every row with its dataset row (an image test: its file, the full path on hover), or the mistakes, most confident first (regression: the largest errors first). |
| By column value | The error by the values (or ranges) of a column of the tested table. |
| Comparison | The latest test beside the one before (**Train > Test > Compare Test Results...**). Shown for the latest test only. |

Metric-learning tests have no Errors, Curves or Calibration sections.
Class names come from the data: a Text Tokenizer's label keeps the names of
its label column, so a sentiment model shows *Anxiety*, *Normal*,
*Depression* instead of class numbers.

The figures at a threshold are read off the class's ROC curve, which holds
the true and false positives at every distinct score, so moving the
threshold makes no new pass over the rows. They match scikit-learn.

### Errors

The summary reads, for example, "20 of the 1,007 test rows are wrong (98.01
% right). The commonest mistake is a Class 6 read as Class 5 (3 rows)".

- **Explain the first in the Model window**, and the **Explain** link under
  each mistake, open it in **Model > Explain** (see
  [The Model window](../../main-window/model-window.md#explain)).
- **Save the wrong rows as a table** keeps them as a dataset.

## The buttons under the view

| Button | Does |
| --- | --- |
| **Export CSV...**, **Export JSON...** | Save what is shown: the metrics, every class with its name, the confusion matrix, each prediction with its dataset row, a regression's rows, the tested table's name. JSON adds the curves and the quick-test flag. The save dialog starts in the project's `exports` folder. |
| **Keep as node...** | Keeps the latest test for later (next section). |
| **Clear** | Asks first, then forgets the latest test and the one before. Kept tests stay. |
| **Run test again** | Tests the current graph's trained model again. |

## The test_predictions table

`test_predictions` has a `row_id` column (each test row's row in the tested
table, counted from 0; for an image test, its row in the dataset's index
table) and, for an image test, a `file` column after it. Its metadata keeps
the tested table's name, the class names, the loss and the time, so the
results join back to the data and a copy read back later gives the whole
run.

## Keep a test as a node

Testing again replaces `test_predictions`; a kept test stays.

1. In **Results > Test**, with **Latest test** showing, click **Keep as
   node...**.
2. Type a name (the graph's name and *test* to start with) and click
   **Keep**.

The Engine then:

- saves the test's rows once as
  `results/tests/<name>_<date>_<time>.parquet` in the project (an existing
  file is never replaced), with its tested table, class names, loss, time
  and the name it was kept under;
- adds a **Data Input** reading that file and a **Test Evaluation** node
  after it, in free space on the canvas (Ctrl+Z removes them; the file
  stays). **Showing** then lists it under *Kept*.

### The Test Evaluation node

The **Test Evaluation** node (Analytics) reads any test's per-row table (a
kept test, or `test_predictions` itself) and gives the Test view's figures
as four tables:

| Output | Columns |
| --- | --- |
| Metrics | `metric`, `value`: accuracy, macro precision, recall and F1, weighted F1, AUC, loss, rows (regression: MAE, RMSE, bias, loss, rows) |
| Confusion | `true`, `predicted`, `count` (a classifier) |
| Per class | `class`, `rows`, `precision`, `recall`, `f1` (a classifier) |
| Worst rows | `dataset_row` (counted from 1), `true`, `predicted`, `confidence`: the most confident mistakes; regression: `error`, the largest errors first |

*Worst rows* keeps 20 rows (Properties: **worst_rows**). A table that is
not a test's is refused (*not a test's predictions*).

## Generations

The generation previews a language model's training writes (set on the
Data Loader's **Generation** tab). **Run** lists every run of the project
that wrote previews, newest first. Under it is the same Generations view as
the Training Dashboard's: the latest pass with one card per prompt, two
passes compared, or one prompt over time. When there is no test yet but
there are preview runs, Results opens on Generations. See
[Generation previews](../training/generation-previews.md).

## Related

- [Test a trained model](test-a-trained-model.md)
- [Choose a threshold and calibrate](threshold-and-calibration.md)
- [Cross-validation and learning curves](cross-validation-and-learning-curves.md)
- [The Model window](../../main-window/model-window.md)
