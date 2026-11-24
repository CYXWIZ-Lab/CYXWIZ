---
title: Compile and run a graph
description: Check a graph with Compile, read the compile report, pick a run mode, and train, debug or run the data steps.
type: how-to
area: graph-editor
audience: user
status: stable
since: 1.0.7
owner: graph-editor
last_reviewed: 2026-11-24
related:
  - fix-missing-values.md
  - connect-nodes.md
  - node-properties.md
  - ../../main-window/toolbar.md
  - ../training/studio-debugger.md
---

Use this when your graph is wired and you want to know whether it can
train, then run it. The buttons are on the Studio bar above the canvas; see
[The Studio bar](../../main-window/toolbar.md) for every button and state.

## Compile the graph

The Studio bar checks the graph as you edit, about half a second after
each change. The chip beside **Compile** reads **Checking...**, **Ready**,
**N errors**, **Failed** (the graph could not be checked) or **Empty**. In
Data Pipeline mode it reads **Not checked** until you press Compile.

To see every finding:

1. Click **Compile** on the Studio bar, press **F7**, or choose **Train >
   Compile Graph**.
2. The compile report opens, titled **Compile: &lt;graph&gt;**.

Click the chip instead for **Open the compile report** and **Show the first
finding on the canvas**.

## Read the compile report

The report starts with the verdict: **Ready to train**, **Ready to train,
with N warnings** or **Not ready to train: N errors**. Chips count the
errors, warnings and notes, and the layers that run on the GPU.

**Findings (N)** lists errors, then warnings, then notes:

| Level | Means |
| --- | --- |
| **Error** | Train will not start until you fix it. |
| **Warning** | Train runs; read it first. |
| **Note** | Information. |

Each finding names its node. A finding with its own code shows it, for
example `CW-C-0201` for data that is not loaded. Click **Show on canvas** to
jump to the node. Where the fix is known, a button does it:

| Button | Does |
| --- | --- |
| **Open its dialog** | Opens the Data Input whose data is not loaded, so you can click **Apply**. |
| **Add a Text Tokenizer** | Adds the tokenizer a text input needs. |
| **Add a Resize** | Adds the Resize an image graph needs. |
| **Add Mel Spectrogram**, **Add Spectrogram**, **Add MFCC** | Adds the feature node an audio graph needs. |
| **Open Data Studio** | Opens Data Studio to check the data before a long run. |

Two more tabs:

- **Where each layer runs**: one sentence ("All 6 layers run on the GPU"),
  counts for GPU, CPU, mixed, at risk, unsupported and unknown, and a row
  per layer in plain words. Layers that need attention come first. Each
  row's **Details** gives the reason code and the backends.
- **Configuration**: the layers, input per sample, outputs, learnable
  parameters, then the data (dataset, where it is held, the split), the
  training loop (batches, epochs, metrics, workers, seed) and learning
  (loss, optimizer, learning rate, labels). **Memory of a training step
  (estimate)** breaks down the memory one step needs: weights, gradients,
  optimizer state, kept state and one batch's activations. When the
  device's memory is known the last row gives the share, and above 100% it
  suggests a smaller batch or input. The estimate never blocks training.

The footer shows the last **Local Debug** on this graph with **Open the
Studio Debugger**, then **Copy report** (the whole report as text, for a bug
report), **Run Local Debug (F6)**, **Close** and, when the graph is ready,
**Train**.

The same window explains a stop, titled **Cannot start training**,
**Cannot run Local Debug** or **Cannot load the checkpoint**, with the
reason ("Training did not start: ...") and "Nothing was trained and nothing
changed in the graph." Missing values in the data are explained in this
window too; see [Fix missing values before training](fix-missing-values.md).

Hover a link next to a broken node on the canvas to read **&lt;node&gt;: will
not compile** and the message.

### Code export checks the graph separately

**Code** on the Studio bar runs its own, older check with its own wording
("Graph must have at least one Output node."). A graph the report calls
**Ready to train** can still be refused by **Code**. Fix what Code says
before generating code.

## Pick a run mode

The Studio bar has three modes. A new graph starts in **Code Gen**.

| Mode | What it adds |
| --- | --- |
| **Code Gen** | A framework picker (PyTorch, TensorFlow, Keras, PyCyxWiz) for **Code**, which writes Python into the Script Editor or a `.py` file. |
| **Data Pipeline** | **Execute Pipeline**: runs the data steps (Data Input and the steps before the Data Split) and shows each step's result, with **Cancel**. Training nodes are left to Train. |
| **Local Training** | Nothing extra: Train runs the whole graph on this computer. |

**Train** trains the graph on this computer in all three modes. You do not
need to switch to **Local Training** to train.

## Run the graph

| Action | How | What happens |
| --- | --- | --- |
| **Local Debug** | **F6**, the Studio bar, or **Train > Local Debug** | One sample goes forward and backward through the model in the background; the Studio Debugger opens with the result. See [Studio Debugger](../training/studio-debugger.md). |
| **Train** | **F5**, the Studio bar, **Train > Start Training**, or **Train** in the report | Compiles, prepares the data, then trains. The Training Dashboard opens. |
| **Execute Pipeline** | Studio bar, Data Pipeline mode | Runs the data steps only. |
| **Run Test** | **F8**, or **Train > Test > Run Test** | Tests the trained model on the test split; results open in Data Studio. |

While training, the Studio bar shows **Epoch N / M**, **Pause** (it pauses
after the current step; **Resume** goes on) and **Stop**. The same are in
the **Train** menu and on the Training Dashboard.

When **Train** is greyed, hover it for the reason, for example "Train waits
while a Data Input is loading its data. It can start when that ends." or
"The graph is not ready to train. It needs a dataset input, model layers
and a loss." The reason "Training is not available yet (no dataset
loaded?)" usually means a Data Input has not loaded its data: open it and
click **Apply**, or press **F7** to see the findings.

## What Train does first

Before the first batch, **Train** does these:

1. **Compiles** the graph. With errors it opens **Cannot start training**
   and stops.
2. **Loads the Data Inputs** that feed the Data Loader. A reopened project
   keeps each Data Input's settings but not its loaded data, so Train loads
   it as **Apply** would (the Training Dashboard says "Loading ... before
   training..."). If one cannot be loaded, **Cannot start training** says
   why. Local Debug loads it the same way.
3. **Runs the data steps** before the Data Split, as **Execute Pipeline**
   would, and trains on their result. After an **Execute Pipeline** run,
   Train reads that result and does not run the steps again, as long as
   nothing up to the Split's input changed. Editing a model layer keeps the
   result; editing a data step or the Data Input prepares the data again.
   To see what changed, type `show logs grep "marked dirty"` in **Console >
   Commands**. A Data Input linked straight to the Split runs nothing.
4. **Checks for missing values** in the table the Split reads; see
   [Fix missing values before training](fix-missing-values.md).
5. **Checks memory.** When preparing the data may not fit, **Memory risk
   before training** shows the estimate, the free memory and the safe
   budget. **Cancel** is the default; **Continue anyway** goes on.

Rules that apply while the data steps run:

- A Count or TF-IDF vectorizer set to sparse output gives sparse features,
  and training batches them as sparse. Graphs with sparse steps always
  prepare again.
- A step that makes the Engine grow past the safe memory budget is stopped
  ("the memory guard stopped '...'"); use fewer rows or columns (Sample
  Rows, Select Columns) and run again.
- The label column must be in the steps' result. When a step makes or
  renames the label, pick it in the Data Split's **Label Column** in
  Properties. After **Execute Pipeline** that list shows the columns that
  reach the Split; "(the Data Input's label)" keeps the Data Input's.
- A Data Input on the Split's **Validation** or **Test** pin goes through
  the Training branch's steps, which learn from the Train rows only. Link
  such an input straight to the Data Split, with no step between.

While a graph trains, tests or runs a pipeline or simulation, the graph and
project are locked: loading or clearing a graph says **Graph replacement
locked**, and **File > New Project**, **Open Project** and **Close Project**
say why they cannot run. Stop the run or wait for it.

If the Engine's settings file turns on `require_debug_before_train` (in its
`debug` section), **Train** stays greyed until Local Debug passes on the
graph as it is: "This project requires a passing Local Debug (F6) on this
graph".

## Changes

- Changed in 1.0.8: **Fix and train** in the report, and **Memory of a
  training step (estimate)**.

## Related

- [Fix missing values before training](fix-missing-values.md)
- [The Studio bar](../../main-window/toolbar.md)
- [Studio Debugger](../training/studio-debugger.md)
