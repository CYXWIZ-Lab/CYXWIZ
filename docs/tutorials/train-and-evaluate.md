---
title: Train, evaluate and understand the results
description: Lesson 5. Test the digits model on rows it never saw, read the confusion matrix, the classes and the errors, and see which inputs drove a prediction.
type: tutorial
area: evaluation
audience: user
status: experimental
since: 1.0.8
owner: docs
last_reviewed: 2026-11-25
related:
  - index.md
  - first-model-from-a-starter.md
  - ../user-guide/evaluation/test-a-trained-model.md
  - ../user-guide/evaluation/test-results.md
  - ../main-window/model-window.md
---

In this lesson you go past the dashboard's one accuracy figure. You test the
digits model of lesson 2 on its held-out rows, find which digits it mixes up
and why, and look at the inputs that pushed one prediction.

**You need:** the `digits_mlp` graph from
[lesson 2](first-model-from-a-starter.md), or the **Multiclass image
classification** starter (`examples/cyxgraph/mnist_mlp.cyxgraph`), about
20 minutes.

## 1. Train in this session

The Engine tests the model it holds in memory, so train first.

1. Open the graph: on the Studio bar, open the graph's button, select
   **Load...** and pick `cyxgraph/digits_mlp.cyxgraph`. (Or open the
   starter from an empty canvas.)
2. Press **F5** and wait until the Training Dashboard reads *Completed*.
3. Read the dashboard's last lines: the active model (the best validation
   checkpoint) and **Held-out test**, the accuracy on the test rows.

## 2. Read the curves

On the dashboard's **Loss** chart, compare the training loss with the
validation loss.

- Both fall and stay close: the model learns patterns that hold for new
  rows.
- The validation loss stops falling or rises while the training loss keeps
  falling: the model starts to learn the training rows by heart
  (overfitting). The best validation checkpoint keeps the model from before
  that point.

The validation loss card says how far it is above the training loss.

## 3. Run the test

1. Choose **Train > Test > Run Test**, or press **F8** with the canvas
   focused.
2. The test runs in the background. When it ends, **Data Studio** opens
   **Results > Test** on it.

The header shows the accuracy, how many rows were tested (the test part of
the Data Split, about 180 rows) and the other figures as chips: the loss,
macro and weighted F1, and AUC.

## 4. Read the sections

Select each section in turn:

1. **Overview**: macro precision, recall and F1 (the plain average over the
   ten digits), weighted F1, and the best and the worst digit.
2. **Confusion matrix**: one row per true digit, one column per predicted
   digit. The diagonal holds the right answers; every other cell is a
   mix-up. Switch from counts to the share of each true class to compare
   digits fairly. The line under it names the largest confusion. Hover a
   cell: *x of y (z%)*.
3. **Classes**: precision (when the model says "3", how often it is a 3),
   recall (how many of the real 3s it finds), F1 and support (how many test
   rows) per digit. Sort by F1 to find the weakest digit.
4. **Errors**: how many rows are wrong, the commonest mistake, and the
   mistakes by pair (true -> predicted), the most confident first. A
   confident mistake is worth a look: the model was sure and wrong.
5. **Curves**: a ROC and a precision-recall curve per digit against the
   rest, with AUC and AP.

With about 18 test rows per digit, one row changes a digit's recall by
more than 5 points. Read the figures as rough.

## 5. See what drove a prediction

1. In **Errors**, select **Explain the first in the Model window**. The
   **Model** window opens on **Explain** with that test row picked.
2. Read the headline: the true digit, the predicted digit and how sure the
   model was.
3. This model reads a table row of 64 pixel values, so Explain draws one
   pair of bars per input column: **Integrated Gradients** (each pixel's
   share of the score) and **|saliency|** (how much the score moves when
   the pixel changes). Tall bars are the pixels that mattered most.
4. Change **Explain the score of** to the true class, to see which pixels
   spoke for the right answer.
5. Pick another row on the left, **All** or **Wrong only**.

Grad-CAM, and pictures instead of bars, need an image model with a
convolution; see [The Model window](../main-window/model-window.md#explain).

## 6. Compare two runs

1. In the **Data Loader**'s dialog (double-click it), set **Epochs** to `30`
   on the **Training** tab and select **OK**.
2. Press **F5**, then **F8**.
3. Choose **Train > Test > Compare Test Results...**. **Comparison** shows
   the new test beside the one before.

More epochs do not always help: the best validation checkpoint may come
early, and the rest of the run then changes nothing that is kept.

## Optional: look inside the model

**Tools > Model** opens the Model window's other tabs: **Summary** (each
layer's output shape and parameters; this model has about 17,000),
**Health** (whether each layer is still learning) and **Projector** (how
the test rows group in a layer's features).

## How far to trust one test

One test split is one draw, and 180 rows is few. **Train > Evaluate >
Cross-validate** trains once per fold and scores each fold; see
[Cross-validation and learning curves](../user-guide/evaluation/cross-validation-and-learning-curves.md).

## Next steps

- Lesson 6: [Save, export and use a model](save-export-and-use.md)
- Every section of the results: [Read test results](../user-guide/evaluation/test-results.md)
- Test a saved checkpoint: [Test a trained model](../user-guide/evaluation/test-a-trained-model.md)
- Pick a decision threshold: [Choose a threshold and calibrate](../user-guide/evaluation/threshold-and-calibration.md)
- Which columns matter overall: [Feature importance](../user-guide/evaluation/feature-importance.md)
