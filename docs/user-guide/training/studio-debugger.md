---
title: Check a graph with Local Debug
description: Run one sample of your data through the model in the Studio Debugger before training, and read what the run found.
type: how-to
area: training
audience: user
status: stable
since: 1.0.7
owner: training
last_reviewed: 2026-11-25
related:
  - layer-capture.md
  - inspect-layers.md
  - where-the-model-looks.md
  - ../graph-editor/compile-and-run.md
  - ../../main-window/toolbar.md
redirect_from: []
---

The Studio Debugger runs your graph on one item of your data before you
spend time on full training (**Local Debug**, F6), and shows what every
step found. A passed Local Debug means the data reaches the model, the
shapes fit, the loss is computed and gradients flow back to every weight.

## Before you start

- A graph on the CyxWiz Studio canvas with a Data Input, the model's
  layers, a loss and an optimizer. See
  [Compile and run a graph](../graph-editor/compile-and-run.md).
- Optional: the graph compiles cleanly (**Compile**, F7). Local Debug
  compiles it first anyway and stops on errors.

## Run Local Debug

1. Open the debugger: **Train > Local Debug**, **Local Debug** on the
   [Studio bar](../../main-window/toolbar.md), F6, or **View > Panels >
   Studio Debugger**.
2. In the debugger's toolbar, pick the run mode in the drop-down next to
   **Studio Debugger**. **Local Debug** is the one most graphs need; the
   other modes are in the table that follows.
3. Optional: pick the item to run with **Sample** (- / +). See
   [Which sample runs](#which-sample-runs).
4. Optional: turn on **Layer capture** to keep what every layer held. See
   [Capture what every layer holds](layer-capture.md).
5. Select **Run**. The steps show in a card as they run, each marked when
   it ends, with its time. **Stop** ends the run after the current step;
   steps already finished keep their results, and the run is saved as
   *Stopped* so you can read it.
6. Read the result in the debugger's sections (next section). When Local
   Debug passed on the graph as it is now, select **Train** (F5) under the
   status line to start training.

| Mode | What it does |
| --- | --- |
| **Full Workflow** | Compile, Preflight, Smoke Run and Local Debug in turn, and collects runtime evidence. Every required step must pass; a Smoke Run the graph does not support is skipped, not failed. |
| **Preflight** | Compiles the graph and checks the dataset, preprocessing, shapes, loss and training readiness, without running the model. |
| **Local Debug** | Compiles and checks the graph, then runs the picked sample through forward, loss, backward, gradient and optimizer checks. Timings are wall-clock times on the host, not device kernel times. |
| **Smoke Run** | Runs preprocessing and a short real-data training sample. Text graphs only; for other graphs the mode reads *(not for this graph)* and its tooltip says why. |
| **Runtime Trace** | Reads the latest training trace (status, warnings, transfers, synchronisations, fallbacks) and crash evidence. It does not run the graph. |

## Which sample runs

Local Debug runs one item of your data through the model, forward and
backward. **Sample 7** is item 7 of the dataset (row, image or clip), in
the dataset's own numbering, whichever part of the split it falls in. The
item is prepared the way validation sees it: the data steps before the Data
Split, Resize / Crop and Normalize, the label form, but no random
augmentation (flips, crops, audio noise), so the same sample gives the same
numbers every run. The stepper is locked while a run is active.

The Overview's **Sample** line names what ran and its part of the split,
for example *row 7 of blobs, test* or *image 5 of cat_dog, train*. Images,
tables in memory or on disk, sparse features and audio work this way. Token
tagging sequences and data loaded through the older dataset path run a
synthetic sample instead; the Sample line then says *a synthetic sample
(seed 1337)* and a warning gives the reason. Data that a reopened project
has not loaded yet loads first, as for Train.

## Read the run

The rail on the left lists the debugger's sections; a dot beside a section
takes the colour of what it found. Under them, **Steps this run** lists the
run's steps with their state (hover a step for its detail).

| Section | Views | What you read there |
| --- | --- | --- |
| **Overview** | **Summary**, **Runs**, **Compare** | The outcome, the sample and the device; the saved runs; one run against another. See [Training runs and their snapshots](layer-capture.md#training-runs-and-their-snapshots). |
| **Data** | One view | The batch as the model receives it, after the data steps. |
| **Model** | **Layers**, **Construction**, **Shape comparison**, **Graph path** | What every layer held ([Inspect the layers](inspect-layers.md)), how the model was built from the graph, expected against observed shapes, and the path through the graph. |
| **Training** | **Gradients**, **Loss & metrics** | Gradient health per layer; how the loss and the metrics were computed. |
| **Runtime** | **Execution**, **Timeline**, **Data preparation**, **Backend**, **Tensors**, **Memory**, **Crash** | Where the time and memory went, which back end ran each step, and the last crash's evidence. |
| **Diagnostics** | One view | The Studio events of the run. |

The right column has three tabs: **Inspector** (the selected trace, node or
item), **Issues (n)** (what the run found) and **Fixes (n)** (what to
change, per node).

## The debugger and the canvas

Under the debugger's status line:

- **Canvas** goes back to the graph. The debugger hides; F6 or **View >
  Panels > Studio Debugger** brings it back with its run.
- **Compile report** (F7) opens the compile report.
- **Train** (F5) appears once Local Debug passed on the graph as it is now,
  with the result, for example *Local Debug: Passed at 16:52, graph
  unchanged*.

**Saved runs** (top right) reopens an earlier run from the project's run
store. **Full size** (the expand button beside it) gives the debugger the
whole width: it hides the side panels (Asset Browser, Nodes, Patterns,
Properties, Info) while the debugger is shown. Select it again, switch to
another tab or close the debugger, and the panels come back.

## If something goes wrong

- **Run is disabled, "The graph editor is not available"**: no graph is
  open on the canvas. Open or create one.
- **Run is disabled in Smoke Run**: the graph is not a text graph. Pick
  **Local Debug** or **Full Workflow**.
- **The Sample line says "a synthetic sample"**: the data cannot hand over
  one item by number (token tagging, the older dataset path). The run still
  checks shapes and gradients, on generated values.

## Related

- [Capture what every layer holds](layer-capture.md)
- [Inspect the layers](inspect-layers.md)
- [See where the model looks](where-the-model-looks.md)
- [Compile and run a graph](../graph-editor/compile-and-run.md)
- [Training](index.md)
