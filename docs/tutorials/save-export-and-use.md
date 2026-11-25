---
title: Save, export and use a model
description: Lesson 6. Save the graph, export the trained model as a CyxWiz package, write it as PyTorch code, and test it again in a later session.
type: tutorial
area: deployment
audience: user
status: experimental
since: 1.0.8
owner: docs
last_reviewed: 2026-11-25
related:
  - index.md
  - train-and-evaluate.md
  - ../user-guide/deployment/export-a-model.md
  - ../main-window/toolbar.md
  - ../user-guide/training/checkpoints-and-early-stopping.md
---

A trained model lives in the Engine's memory until you keep it. In this
lesson you keep the digits model three ways: as a graph file, as an
exported model file, and as Python code for PyTorch. Then you close the
Engine, open it again and test the model you kept.

**You need:** the `digits_mlp` graph from
[lesson 2](first-model-from-a-starter.md), trained in this session (press
**F5** if you have restarted the Engine since), about 15 minutes.

## 1. Save the graph and the project

1. Click the canvas and press **Ctrl+S**. The graph's file
   `cyxgraph/digits_mlp.cyxgraph` holds the nodes, their links and their
   properties. It does not hold the trained weights.
2. Choose **File > Save**. It saves the graph, every open script, the
   window layout and the project's settings.

## 2. Export the trained model

1. Choose **File > Export Model > CyxWiz package (.cyxmodel)** (**Ctrl+E**).
   The **Export model** dialog opens.
2. **File** reads `exports/digits_mlp.cyxmodel` in the project folder.
3. Under **Include**, keep **Graph** and **Training history** on.
4. Optional: type an **Author** and a **Description**.
5. Select **Export**. The result lists the file, its size, the parameters
   and layers, and the time. Select **Show in folder** to see the file, then
   **Close**.

A CyxWiz package holds the weights with the graph and the training history.
It is how any trained model is kept; there is no separate Save Model.
**Weights only (.safetensors)** in the same menu writes the weights alone,
for other frameworks to read.

## 3. Look inside the package

1. Choose **File > Open Model File...** (**Ctrl+I**).
2. Select **Browse...**; it starts in the project's `exports` folder. Pick
   `digits_mlp.cyxmodel`.
3. Read what it holds: the format, **Parameters** (17,226 for this network),
   **Trained** (the epochs, the final accuracy and loss), **Contains** and
   the layers with their shapes.
4. Close the dialog. (**Open its graph** would load the package's graph
   onto the canvas, without its weights.)

## 4. Write the model as PyTorch code

1. On the Studio bar, set the mode to **Code Gen** (when the canvas is
   narrow, the mode is in **More**).
2. Set the framework, beside the mode, to **PyTorch**.
3. Open **Code** and select **Generate into Script Editor**.
4. The Script Editor opens a tab `digits_mlp_pytorch.py`. A line under the
   Studio bar says how many lines were written, and where.

Read the code: a `GeneratedModel` class (`torch.nn.Module`) with the
graph's layers in order, then the loss and the optimizer from the graph's
loss and Adam nodes. Loading the data and the training loop are left to
you, as a commented example at the end. To keep the file, select **Code >
Save as .py...**.

The mode changes only what **Code** writes; **Train** trains on this
computer in every mode. Set the mode back to **Local Training** if you
like.

## 5. Use the model in a later session

Your training run also kept its best checkpoint in the project's
`checkpoints` folder (the Data Loader's **Save the best validation
checkpoint**, on by default). That checkpoint is what you load to use the
model again without training.

1. Choose **File > Exit** and start the Engine again. Open the
   `cyxwiz-tutorials` project from **Recent Projects**.
2. Load `cyxgraph/digits_mlp.cyxgraph` if it is not on the canvas.
3. Choose **Train > Test > Load Checkpoint for Testing...**. A folder picker
   opens in `checkpoints`, with one folder per training run (`train-<id>`).
4. Pick the newest run. The Engine builds the graph's model and loads the
   best checkpoint into it; the Training Dashboard reads *Model loaded*.
5. Press **F8**. **Results > Test** shows the same figures as in lesson 5.

The loaded model is the active model again: **Export Model**, the **Model**
window and **Run Test** all use it.

## Clean up

Optional: delete `exports/digits_mlp.cyxmodel` and the folders in
`checkpoints` from the Asset Browser or your file manager. The graph stays
in `cyxgraph`.

## You finished the learning path

You can now open a graph, follow its data, prepare your own data, build and
compile a graph, train and test it, explain a prediction, and keep the
result. Go on with the lessons by kind of data, or with Python:

- [Classify text](text-classification.md)
- [Classify audio](audio-classification.md)
- [Train a time-series forecaster](time-series-forecasting.md)
- [Train a model from the Script Editor](python-in-the-script-editor.md)

## Next steps

- Every export format and option: [Exporting a model](../user-guide/deployment/export-a-model.md)
- Best checkpoints, early stopping and resuming:
  [Checkpoints, early stopping and resume](../user-guide/training/checkpoints-and-early-stopping.md)
- The Code menu and the modes: [The Studio bar](../main-window/toolbar.md#the-mode-the-framework-and-code)
