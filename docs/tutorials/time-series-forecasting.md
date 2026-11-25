---
title: Train a time-series forecaster
description: Train the Time-series forecasting starter on 144 months of airline passengers and learn windows, a chronological split and regression results.
type: tutorial
area: training
audience: user
status: experimental
since: 1.0.8
owner: docs
last_reviewed: 2026-11-25
related:
  - index.md
  - first-model-from-a-starter.md
  - ../user-guide/data-studio/time-series.md
  - ../user-guide/data-studio/forecast.md
---

In this lesson you train a model that predicts next month's number of
airline passengers from the twelve months before it. Unlike the other
lessons it predicts a number, not a class (a regression), and the order of
the rows matters: the model must never learn from the future it is tested
on.

**You need:** lesson 2 done, the **Time-series forecasting** starter
(`examples/cyxgraph/timeseries/airline_passengers_dense.cyxgraph`) and its
data `examples/datasets/airline_passengers.csv`: the monthly totals of
international airline passengers, January 1949 to December 1960, in
thousands (144 rows, columns `Month` and `Passengers`). About 15 minutes.

## 1. Open the starter

1. On an empty canvas, select **Open an example graph**, then **Time-series
   forecasting**. Press **F** to fit it in view.
2. Double-click the **Airline Passengers CSV** node (a Data Input). The rows
   are used as **Time series**, with `Passengers` as the label. The file is
   already in time order, and the rows keep the file's order.
3. Select **OK**.

## 2. Follow the series through the graph

| Node | What it does here |
| --- | --- |
| **Window (12 -> 1)** | A **Sliding Window** node. It cuts the series into overlapping windows: 12 months in, the next month out, moving one month at a time. 144 months give 132 windows. |
| **Chronological Split** | A **TS Split** node: the first 80 % of the windows train, the next 10 % validate, the last 10 % test. Nothing is shuffled across the split, so the test is the last stretch of the series, as a real forecast would be. |
| **DataLoader** | Batches of 16, 50 epochs. It shuffles the training windows among themselves, which is safe once the split is made. |
| **Dense**, **ReLU**, **Dropout**, **Dense(1)** | 32 hidden units, then one output: the forecast. |
| **MSELoss** | The loss is the mean of the squared errors, in squared thousands of passengers, so its first values are large. |
| **Adam** | Learning rate 0.001. |
| **Forecast** | The Output node, with 1 output. |

A random split would put months from 1960 in training and months from 1955
in the test: the model would be tested on the past with the future known.
Always split a series by time.

## 3. Train

1. Press **F7**, then **F5**.
2. The Training Dashboard shows the loss falling. A regression run has no
   accuracy; at the end the held-out test line gives the loss and the mean
   absolute error (MAE): how many thousand passengers the forecast is off,
   on average.

## 4. Read the forecast

1. Press **F8**. **Results > Test** opens with MAE as the headline, and
   RMSE and R² as chips.
2. Open **Plots**: predicted against actual, the residuals (actual minus
   predicted is the error) against the prediction, and their histogram.
   Points on the diagonal of the first plot are good forecasts.
3. Open **Overview** and read **Bias**, the mean of predicted minus true.
   This series keeps growing, and its last months are higher than any the
   model trained on. A model often predicts such months low; a negative
   bias says this one does.

## 5. Try a change

The series grows about fourfold over twelve years, which a small network
fits poorly without scaling. Try a longer memory:

1. Click the **Window (12 -> 1)** node and set **Input Width** to `24`.
2. Press **F5**, then **F8**, and compare with **Train > Test > Compare
   Test Results...**.

Longer windows leave fewer windows to train on (120 instead of 132), so
the result can go either way. That trade-off is the point of the exercise.

## Next steps

- Decompose a series in Data Studio: [Analyse a time series](../user-guide/data-studio/time-series.md)
- Classical forecasts without training: [Forecast a time series](../user-guide/data-studio/forecast.md)
- What the regression figures mean: [Read test results](../user-guide/evaluation/test-results.md)
- [Train a model from the Script Editor](python-in-the-script-editor.md)
