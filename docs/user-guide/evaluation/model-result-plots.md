---
title: Plots of model results
description: Draw a confusion matrix, ROC and precision-recall curves, calibration, residuals, learning curves and feature importance from any table of results.
type: reference
area: evaluation
audience: user
status: stable
since: 1.0.7
owner: evaluation
last_reviewed: 2026-11-25
related:
  - plotting.md
  - plot-node.md
  - test-results.md
redirect_from: []
---

The **Model results** plot types draw a trained model's results from plain
columns of any table: a predictions file, an evaluation node's output, a
training history. Each shows its figures in the plot's corner and under
**RESULTS** in the Data panel. For the tests the Engine runs itself, Data
Studio's [Results > Test](test-results.md) draws the same figures without
any setup; these types are for results from anywhere else.

## The types

| Type | Columns | What it draws |
| --- | --- | --- |
| Confusion matrix | **Actual** and **Predicted** label columns (text or numbers) | Each cell with the count and its share. **Colour and share by**: counts, share of actual (each row adds to 100%, the default), share of predicted, or share of all rows. Accuracy and rows under RESULTS. Hover: "Actual tested_positive · Predicted tested_negative", the rows, the share, right or wrong. |
| ROC curve | **Actual** and the **Score** of the positive class (a probability or any score; higher means positive) | The curve, AUC in the corner, the chance line. **Positive class**: auto picks 1, true, yes or a label with "positive", else the last label; type another to change it. Hover gives the threshold and both rates. |
| Precision-recall curve | The same columns | Precision by recall as a step, the average precision (AP, as scikit-learn computes it) and the positive share as a dashed baseline. |
| Calibration | **Actual** and a **Probability** (0 to 1) | The share of positives in each of the **Bins** by the mean predicted probability, the perfectly calibrated diagonal, rows per bin as bars along the bottom (right axis); Brier score and ECE. |
| Residuals | Number **Actual** and **Predicted** | Actual minus predicted by predicted, a line at 0; RMSE, MAE and R². Large tables are sampled like a scatter; the figures and exports use every row. |
| Learning curve | **X** (training rows or epoch) and one or more curves (train, validation) | A **Spread** column per curve draws a band (± that value). The last curve's best point is ringed: **Auto** takes the lowest for a name with loss or error, else the highest. |
| Feature importance | A **Feature** column and its **Importance** | Horizontal bars, largest on top; **Show the top** N (20); an optional **Spread** for error bars. Hover gives the value and the rank. |

Example: a diabetes classifier's predictions give a confusion matrix of
134, 16 / 36, 45 (accuracy 77.5%), a ROC curve with AUC 0.871, AP 0.767 and
a Brier score of 0.143.

## From evaluation nodes on the canvas

A [Plot node](plot-node.md) after an evaluation node opens on the fitting
plot:

```text
Data Input (actual, predicted, score) --> Confusion Matrix --> Plot
                                      \-> ROC Curve        --> Plot
```

| Node before the Plot | First plot |
| --- | --- |
| **Confusion Matrix** (`actual_col`, `predicted_col`) | A heatmap of *Actual* by *Predicted* with the counts in the cells (or the shares, when the node normalizes); hover says "Actual 0 · Predicted 1 · value 38". |
| **ROC Curve** (`actual_col`, `score_col`, `positive_label`) | The true positive rate by the false positive rate, the AUC in the title ("ROC curve · AUC 0.931") and the y = x chance line. |
| **Precision-Recall Curve** | Precision by recall, the average precision in the title. |

The same tables read from a file get the same first plot. The type and
columns can be changed like any plot.

## From Python

`cx.confusion_matrix(actual, predicted)`, `cx.roc_curve(actual, score)`,
`cx.pr_curve`, `cx.calibration`, `cx.residuals`, `cx.learning_curve(x,
*ys)` and `cx.importance(feature, value)` open the same plots from a
script; see [Plots from Python](plots-from-python.md).

## Related

- [Read test results](test-results.md)
- [Plot data](plotting.md)
- [Plot data on the canvas](plot-node.md)
