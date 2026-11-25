---
title: Choose a threshold and calibrate
description: Pick the decision threshold for a class, set a confidence floor, calibrate the model's probabilities and keep the decision with the model.
type: how-to
area: evaluation
audience: user
status: experimental
since: 1.0.8
owner: evaluation
last_reviewed: 2026-11-25
related:
  - test-results.md
  - test-a-trained-model.md
redirect_from: []
---

A classifier predicts a class when its score is high enough. The
**Threshold** and **Calibration** sections of Data Studio's **Results >
Test** let you choose that rule from a test, check whether the model's
probabilities can be trusted, and keep both with the trained model, so the
next test, an export and the local inference server use them.

## Before you start

- A finished test of a classifier (**Train > Test > Run Test**, F8); see
  [Test a trained model](test-a-trained-model.md).
- Threshold needs the class scores (the Curves section is shown).
  Calibration needs probabilities for two or more classes. Neither is shown
  for metric learning.

## Choose a threshold

1. In **Results > Test**, open the **Threshold** section.
2. Pick the class: **Positive class** for a two-class (yes / no) model, or
   **Class against the rest** for a model with more classes.
3. Set the threshold in one of these ways:
   - type it in **Threshold** (or use - / +);
   - click **Best F1**;
   - type a share in **Recall at least** and click **Use**: the threshold
     that finds that share of the class;
   - click **Cost of each error...**, type the cost of a false alarm and of
     a miss in your own units, and click **Use the lowest cost**.
4. Read the figures at that threshold: precision, recall, specificity, F1,
   balanced accuracy, MCC, accuracy and the 2 x 2 counts.

Moving the threshold makes no new pass over the rows: the figures are read
off the class's ROC curve. They match scikit-learn.

## Set a confidence floor

A model with several classes also gets a **Confidence floor**: below it a
row is left for a person instead of being answered. The curve shows the
share of rows answered, and how many of those are right, at every floor.

## Calibrate the probabilities

1. Open the **Calibration** section. Bars show the share right in each
   probability band against the line of perfect calibration, with the
   calibration error (ECE).
2. Pick the method: **Temperature** (temperature scaling, for several
   classes; it changes no prediction), **Platt** (a logistic fit, yes / no)
   or **Isotonic** (isotonic regression, yes / no).
3. Switch **Before** / **After** to compare. The fit is made on the even
   test rows and shown on the odd ones, so it is not judged on the rows it
   was fitted to.

The threshold sweep, the recall target, the costs, ECE, temperature
scaling, Platt and isotonic regression are checked against scikit-learn
and SciPy.

## Keep the decision with the model

1. Click **Keep *0.500* with the model** (the threshold shown) under
   Threshold, and **Keep the temperature with the model** (or **Keep
   Platt's sigmoid with the model**, **Keep the isotonic fit with the
   model**) under Calibration.
2. The next test applies the kept decision and says so ("This test used
   the model's kept temperature 0.804").

What carries the decision:

- the checkpoint's `metadata.json` and an exported model (`decision`);
- the local inference server: `/v1/predict` returns each row's calibrated
  `probabilities` and its `predicted` class by the kept threshold, and
  `/v1/model` shows the decision.

**Forget** drops a kept decision.

## If something goes wrong

**A calibration cannot be fitted**: a test that already used a kept
calibration cannot fit another on top of it. Click **Forget**, test again,
then calibrate.

## Related

- [Read test results](test-results.md)
- [Test a trained model](test-a-trained-model.md)
- [Export a model](../deployment/export-a-model.md)
