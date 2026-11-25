---
title: Test a trained model
description: Run the trained model on the test part of the data, test a saved checkpoint, and find the results.
type: how-to
area: evaluation
audience: user
status: stable
since: 1.0.7
owner: evaluation
last_reviewed: 2026-11-25
related:
  - test-results.md
  - threshold-and-calibration.md
  - cross-validation-and-learning-curves.md
  - ../training/checkpoints-and-early-stopping.md
redirect_from:
  - ../../usage/testing.md
---

After training, test the model on rows it never trained on: the test part
of the graph's Data Split. The test runs from the **Train > Test** menu, and
its results open in Data Studio's **Results > Test**.

## Before you start

- The graph has a Data Split with a test part (or a dataset wired to its
  **Test** input; see [Split images and audio](../data-studio/split-images-and-audio.md)).
- The model is trained in this session, or you have a saved checkpoint of
  the graph.

## Run a test

1. Choose **Train > Test > Run Test** (**F8**). The whole test part runs.
   Optional: **Train > Test > Run Quick Test** runs only the first 10
   batches. Its results are marked *quick test* and are never taken for a
   full run.
2. The test runs in the background; **Background Tasks** shows it. When it
   ends, Data Studio opens **Results > Test** on it.
3. To open the results again later, choose **Train > Test > View Test
   Results**.

What the results show, section by section, is in
[Read test results](test-results.md).

## Which rows are tested

The test part is the one the model trained without. The Data Split the
model trained with (its ratios, **Stratified**, seed, and the images a
Quality node left out) is kept with the model and in every checkpoint it
saves. When the graph's Data Split changed since, the log says so and the
test still reads the part the model never saw. A checkpoint from an older
Engine that has no split recorded is tested with the graph's current split.

Image and audio graphs test their Data Split's test part through the same
image steps as training. Each image's file is kept with its result.

A graph with branches (Split, Add, Concatenate) is tested like any other:
the model it trained in this session is the one tested.

## Test a saved checkpoint

1. Choose **Train > Test > Load Checkpoint for Testing...**. A folder picker
   opens in the project's `checkpoints` folder, listing each training run
   (`train-<id>`).
2. Pick a run (its best checkpoint is used) or a checkpoint folder.
3. The Engine builds the graph's own model and loads the checkpoint into
   it. The graph on the canvas must compile with its Data Inputs applied;
   otherwise the load stops and lists what to fix.
4. Run the test as in [Run a test](#run-a-test).

Training with *save best checkpoint* keeps and restores its best epoch;
see [Checkpoints, early stopping and resume](../training/checkpoints-and-early-stopping.md).

## Save and compare results

- **Train > Test > Export Test Report...** saves the latest test as JSON
  (the same file as **Export JSON...** in Results > Test).
- **Train > Test > Compare Test Results...** opens Results > Test on
  **Comparison**: the latest test beside the one before.

## How far one test can be trusted

One test split is one draw. **Train > Evaluate > Cross-validate** trains
the graph once per fold and scores each fold once; **Train > Evaluate >
Learning Curve** shows whether more rows would help. See
[Cross-validation and learning curves](cross-validation-and-learning-curves.md).

Changed in 1.0.8: the results moved from the Test Results window (and the
Confusion Matrix, ROC / AUC and PR Curve windows) into Data Studio's
**Results > Test**.

## If something goes wrong

**The test failed**: Results > Test shows the reason above the last
successful results; a failed run is never compared against. Background
Tasks shows the failed task, and `engine_log.txt` has the details.

## Related

- [Read test results](test-results.md)
- [Choose a threshold and calibrate](threshold-and-calibration.md)
- [Feature importance](feature-importance.md)
- [Export a model](../deployment/export-a-model.md)
