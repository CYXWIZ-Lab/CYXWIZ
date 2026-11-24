---
title: Train on unbalanced classes
description: Weight the loss per class or balance the training batches when some classes have far more rows than others.
type: how-to
area: training
audience: user
status: stable
since: 1.0.7
owner: training
last_reviewed: 2026-11-25
related:
  - losses-and-optimizers.md
  - train-and-watch.md
  - ../evaluation/test-results.md
---

Use this when one class has far more rows than another, so a model can
score well by mostly predicting the large class. The Engine offers two
remedies: weight the loss, or balance the batches the Data Loader draws.
Use one of them, not both.

To see how unbalanced your classes are, look at the label column in Data
Studio's [Profile](../data-studio/profile.md). To see whether the model
neglects a class, read the per-class figures of a test in
[Test results](../evaluation/test-results.md).

## Weight the loss

On a **CrossEntropy / Token CE** node, set **Class Weight**:

| Value | Does |
| --- | --- |
| `none` | Every row counts the same (default). |
| `balanced` | The Engine counts the classes in the Train part when training starts and weights each class by rows / (classes x rows of the class), as scikit-learn's `balanced` does. The log names the counts and the weights. |
| `manual` | Uses **Class Weights**: one number per class, in class order, separated by commas or spaces, for example `1, 4.5, 2`. |

Compile refuses `manual` without weights ("CrossEntropy class_weight=manual
requires class_weights."), a weight that is not a non-negative number, and a
list whose length differs from the number of classes. `balanced` stops the
run, with the reason, when a class has no rows in the Train part, and for
time-series windows, which have no class counts.

Class weights exist only on CrossEntropy. On another loss they are ignored,
and Compile warns. For two classes with **BCE with Logits**, use **Pos
Weight**; for rare classes in general, **Focal Loss**.

## Balance the batches

The Data Loader can draw the training batches so the classes come up
equally often:

1. Double-click the **Data Loader** and open its **Balance** tab.
2. Check **Balance training classes**.
3. Pick a **Mode**:
   - **Oversample minority**: repeats rows of the small classes up to the
     target;
   - **Undersample majority**: keeps only the target number of rows of each
     large class;
   - **Weighted sampler**: draws each row's class at random with equal
     chances, keeping the epoch's length.
4. Pick **Target per class**: `max`, `median`, `min` or a number.
   Optional: **Balance seed** fixes the draw.

Balancing changes the training batches only; validation and test keep
their real class mix, so their figures stay honest. It applies to table
data (in memory or read from Parquet on disk) with a class label; it does
not apply to images, audio or token sequences, and not to a regression.

## Related

- [Losses and optimizers](losses-and-optimizers.md)
- [Train a model and watch it](train-and-watch.md)
- [Test results](../evaluation/test-results.md)
