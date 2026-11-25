---
title: Train your first model from a starter graph
description: Lesson 2. Open the digits starter graph, learn what each node does, compile it, train it and read the Training Dashboard.
type: tutorial
area: training
audience: user
status: experimental
since: 1.0.8
owner: docs
last_reviewed: 2026-11-25
related:
  - index.md
  - tour-the-main-window.md
  - train-and-evaluate.md
  - ../user-guide/training/train-and-watch.md
  - ../main-window/training-dashboard.md
---

In this lesson you train a model that reads handwritten digits. You open a
working graph, follow the data through it node by node, check it with
**Compile**, train it with **Train**, and read what the Training Dashboard
tells you.

**You need:** the `cyxwiz-tutorials` project from
[lesson 1](tour-the-main-window.md) (any project works), the **Multiclass
image classification** starter (`examples/cyxgraph/mnist_mlp.cyxgraph`)
and its data `examples/datasets/digits_8x8.csv`, about 15 minutes.

## 1. Open the starter

If the starter is still on the canvas from lesson 1, skip to part 2.

1. On an empty canvas, select **Open an example graph**, then
   **Multiclass image classification**. (From the start page: **Task
   starter graphs** > **Multiclass image classification** > **Open**.)
2. Press **F** to fit the graph in view.

You see twelve nodes in a row, linked left to right, and one link that
runs from the **DataLoader** down to the loss.

## 2. Follow the data through the graph

Click each node in turn and read it in **Properties**. The data flows from
left to right:

| Node | What it does here |
| --- | --- |
| **Data Input** | Reads `digits_8x8.csv`: 1,797 rows, 64 pixel columns (grey levels 0 to 16) and the label column `class` (0 to 9). Double-click it to see its dialog: the file, **Table**, and the **Label column**. |
| **Train/Val/Test Split** | Splits the rows 80 / 10 / 10 into training, validation and test parts, keeping each digit's share the same in every part (**Stratified**), with seed 42. |
| **DataLoader** | Serves the training rows in shuffled batches of 128, for 10 epochs. Its second output carries the labels to the loss. |
| **Normalize** | Divides every pixel by 16 (mean 0, std 16), so the values run from 0 to 1. |
| **Dense**, **ReLU**, **Dense**, **ReLU** | Two hidden layers of 128 and 64 units. ReLU keeps positive values and sets the rest to 0. |
| **Dense** (10 units) | Gives one score per digit. |
| **CrossEntropy** | Measures how wrong the scores are against the true digit: the loss. |
| **Adam** | Changes the weights after each batch to lower the loss, with learning rate 0.001. |
| **Output** | Marks the end of the model: 10 classes. |

Hover a pin (the small circles on a node's sides) to see what it carries:
its name, its type and its links.

## 3. Compile

1. Press **F7**, or select **Compile** on the Studio bar (it may be in
   **More** when the canvas is narrow).
2. The compile report opens. Its verdict should be **Ready to train**.
   The tab **Configuration** shows what will train: the layers,
   the input per sample (64 values), the parameters, the split and the
   batches.
3. If the report lists *Data is not loaded - open the node and click
   Apply* on the Data Input, select **Open its dialog**, then **Apply**,
   and compile again.
4. Select **Close**.

The Studio bar also checks the graph as you edit: the chip beside
**Compile** reads **Ready** while the graph can train.

## 4. Train

1. Press **F5**, or select **Train** on the Studio bar.
2. The **Training Dashboard** opens at the bottom right. Its state reads
   *Preparing*, then *Training*. On the Studio bar, **Train** becomes
   **Epoch 1 / 10** with **Pause** and **Stop**, and the links on the
   canvas pulse.
3. Wait for the state to read *Completed*. On a CPU the ten epochs take a
   few seconds.

## 5. Read the dashboard

From top to bottom:

1. **The header** says how the run ended ("Finished: every epoch ran."),
   where it ran (the CPU or your GPU), and the data: about 1,430 training,
   180 validation and 180 test rows in batches of 128.
2. **The active model** line names the model the Engine keeps for testing
   and export: the best validation checkpoint, with its epoch.
3. **Held-out test** gives the accuracy on the test rows, which the model
   never saw while training. Expect above 90 %; your figure differs a
   little from run to run.
4. **The cards** show the training and validation loss and accuracy of the
   last epoch. When the validation loss sits far above the training loss,
   the model learns the training rows by heart (overfits).
5. **The charts** show the loss falling and the accuracy rising over the
   epochs. Hold **Ctrl** and turn the mouse wheel over a chart to zoom it.

The run is added to **Runs**, the table of this session's runs.

## 6. Save the graph

1. Click the canvas, then press **Ctrl+S** (or select **Save** on the
   Studio bar).
2. The graph has no file yet, so the Engine asks for one. Save it in the
   project's `cyxgraph` folder as `digits_mlp`.

The Studio's tab now reads **CyxWiz Studio · digits_mlp**. The trained
model lives in this session; lesson 6 shows how to keep it.

## Optional: change one thing and train again

1. Click the first **Dense** node and set **Units** to `32` in
   **Properties**.
2. Press **F5**. Compare the new run with the first in **Runs**.

A smaller layer trains faster and usually scores a little lower.

## Next steps

- Lesson 3: [Bring in and prepare your own data](prepare-your-own-data.md)
- Everything about a run: [Train a model and watch it](../user-guide/training/train-and-watch.md)
- Every line of the dashboard: [Training Dashboard](../main-window/training-dashboard.md)
- What Compile and Train do: [Compile and run a graph](../user-guide/graph-editor/compile-and-run.md)
