# The CyxWiz Studio bar

The bar at the top of the CyxWiz Studio canvas, one row:

> the graph · Save · Edit · View · the mode · the framework · Code · More ...
> Compile › Local Debug › Train

The window's tab names the graph: **CyxWiz Studio · mnist_mlp**, with a dot
after the name while there are unsaved changes (also in its own window's
title when the Studio is moved out of the main window).

## The graph

- **The graph's name** (the file name, or *Untitled graph*). A dot after the
  name means the graph changed since it was saved. Hover for the full path.
  Click it for the graph menu:
  - **Save** (Ctrl+S): writes the graph's own file. A graph that has no file
    yet asks for a name. On an empty canvas nothing is written and Save says
    so: *Nothing to save yet*, "The canvas is empty. Add a node, then save."
  - **Save As...** (Ctrl+Shift+S): writes a new file and the open graph
    moves to it.
  - **Load...**: opens another graph in place of this one. The loaded graph
    starts its own undo history: Ctrl+Z does not bring back the graph that
    was on the canvas before.
  - **Export a copy of the graph...**: writes a copy where you choose; the
    open graph stays on its own file.
  - **Custom Node Editor**.
  - **Clear the canvas...**: asks first; Ctrl+Z brings the nodes back. The
    file changes only when you save.
  - While a run keeps the graph busy (training, a test, a pipeline, a Data
    Input loading its data), Load and Clear stay in the menu, greyed,
    with the reason under them ("Not while training is active ...").
- **Save** is also on the bar.

A starter opened from the start page, a pattern template and an imported
model open as *Untitled graph*: Save asks for a name, so the example files
next to the Engine are never overwritten.

A graph opened from the start page, the Asset Browser or the command line
(`cyxwiz-engine graph.cyxgraph`) brings the Studio's tab to the front, even
when the Script Editor tab was in front.

Ctrl+S and Ctrl+Shift+S save the graph while the canvas is focused (in the
Script Editor they save the script); see Preferences > Shortcuts.

## Edit and View

- **Edit**: **Select all** (Ctrl+A), **Duplicate** (Ctrl+D) and **Delete
  selected** (Delete). Without a selection Duplicate and Delete are greyed
  with "Select nodes first" under them.
- **Undo** (Ctrl+Z) and **Redo** (Ctrl+Y or Ctrl+Shift+Z) on the canvas also
  take back moving nodes: one step per drag, whether it moved one node, a
  selection or a frame with its nodes.
- **View · 100%** (the zoom on the button): **Zoom** with **-** and **+**
  (greyed at 25% and 200%, with the reason on hover; the view keeps its
  centre), **Zoom to 100%**, **Fit all nodes** (F), **Minimap** (M) and
  **Find in the graph** (Ctrl+F).

## The mode, the framework and Code

**Code Gen**, **Data Pipeline** and **Local Training** side by side, one
click each (hover for what each does).

- In **Code Gen** mode the framework is beside it (PyTorch, TensorFlow,
  Keras, PyCyxWiz). In the other modes it is not on the bar, and the Code
  menu names the framework it uses ("Generate PyTorch Code into Script
  Editor"), so code is never written for a framework you cannot see.
- **Code**:
  - **Generate into Script Editor** opens a Script Editor tab named after
    the graph and the framework (`mnist_mlp_pytorch.py`); a second Generate
    replaces that tab's text.
  - **Save as .py...** asks where and writes the file.
  - When the graph cannot be turned into code, both are greyed with the
    reason under them ("The canvas is empty. Add a node first.", "Graph
    must have at least one Output node.").
  - A line under the bar says what was written where and when: "Generated
    118 lines of PyTorch code into Script Editor tab mnist_mlp_pytorch.py ·
    14:32" (**Hide** removes it). A generated file's header carries the time
    it was generated.
- **Data Pipeline** mode has **Execute Pipeline** next to Code; while it runs,
  its progress, the step it is on (hover) and **Cancel**.
- A graph with simulation nodes shows **Run Sim** here instead: during a run
  **Pause** (or **Resume**), **Step**, **Stop Sim** and the sim time (see
  [Graph simulation](graph_simulation.md)).

## Compile, Local Debug, Train (right)

- **Compile** (F7) and **Local Debug** (F6) each show their last result.
  Click the name to run the step (Compile opens its report, Local Debug the
  Studio Debugger); click the result for its menu (the result in a line,
  **Open the compile report** / **Show the first finding on the canvas**,
  or **Run Local Debug** / **Open the Studio Debugger**). A second Local
  Debug cannot start while one runs.
  - Compile, checked as you edit (400 ms after a change): **Checking...**,
    **Ready** (with the warnings in its menu), **N errors**, **Empty**. In
    Data Pipeline mode the graph is not checked as you edit (a data graph
    has no model): **Not checked**, and Compile checks it when you press it.
  - Local Debug: **Not run**, **Passed 19:12**, **Failed** (its menu says
    where, for example "Compile gate failed"), **Graph changed** (it ran on
    the graph before your last edit; Apply on a Data Input counts),
    **Running...**.
- **Train** (F5). Its tooltip says when Local Debug is not a pass on this
  graph: "Local Debug passed at 18:40; the graph changed since. Train runs
  anyway; F6 checks the new graph first." With `require_debug_before_train`
  on (the `debug` section of the Engine's settings file), Train is disabled until Local Debug passes on
  the graph as it is: "This project requires a passing Local Debug (F6) on
  this graph". While something keeps the graph busy before a run (a Data
  Input loading its data, a test, a pipeline), Train is greyed with the
  reason: "Train waits while a Data Input is loading its data. It can start
  when that ends."
- While training, Train's place shows the run: **Epoch 3 / 10**, **Pause**
  (or **Resume**: "Paused at epoch 3 / 10") and **Stop**. A run waiting for
  the GPU says **Waiting for the GPU** (hover for which run holds it). An RL
  graph (Environment, Policy Head, Value Head, PPO) trains with **Train**:
  while PPO runs, the update and the steps take Train's place, with
  **Pause** / **Resume** and **Stop** (see [Training a policy](rl_training.md)).
- A reopened project keeps each Data Input's settings, not its loaded data.
  **Train** loads the Data Inputs that feed the Data Loader first, as their
  **Apply** would (the Training Dashboard says "Loading Sentiment CSV before
  training..."), then trains. If one cannot be loaded, *Cannot Start
  Training* says why ("Could not load the data before training: ... file not
  found: ..."). This includes an HDF5 file, read from the dataset saved in
  the node (a template opened without Apply trains at once).
- On a training graph, **Execute Pipeline** runs the data steps (Data Input
  and the preparation steps before the Data Split) and leaves the Split,
  Loader, model, loss and optimizer to training.
- **Train** after **Execute Pipeline** reads the table the run prepared; the
  data steps do not run again. The Console says "training reads
  'ds_operator_StandardScaler_5', prepared by the canvas run". This holds
  while nothing up to the Split's Training Dataset input changed: editing
  the model (a Dense layer's units) keeps it, editing a data step (Standard
  Scaler's With Mean) or the Data Input prepares the data again. Graphs
  whose steps write sparse output always prepare again. `show logs grep
  "marked dirty"` in Console > Commands names the changed setting
  ("Standard Scaler marked dirty (parameters changed: with_mean)").
- Without a current canvas result, **Train** runs the data steps before the
  Data Split first, as **Execute Pipeline** would (the canvas shows each
  step run), and trains on their result: "Train: running the data steps
  before the Data Split first", then "Train: the data steps ended; training
  reads 'ds_operator_StandardScaler_5'". If a step fails, *Cannot Start
  Training* says which and why. A second **Train** with nothing changed
  reads the same table without running the steps; in a later session it is
  read from disk when it was kept (Loaded data > Disk caches).
- The steps can be scalers, encoders, tokenizers and the other preparation
  nodes, and also SQL Query, Text Clean, Fill Missing, Select Columns, Filter
  Rows, Data Convert, Remove Duplicates, Sort, Rename, Join, Math Formula,
  String Manipulation, Sample Rows, Binning and Polynomial Features. A Data
  Input wired straight to the Split runs nothing. A Count / TF-IDF
  vectorizer set to Sparse output makes sparse features in the run
  ("Registered sparse feature dataset 'ds_operator_TFIDFVectorizer_5':
  rows=3000 features=10"), and training batches them as sparse; they are
  kept on disk for the next session like a table. Image paths are still
  prepared by training itself.
- A step whose run makes the Engine grow past the safe memory budget (the
  data budget, at most 90% of free memory) stops: "the memory guard stopped
  'Scale': the Engine grew by ... while it ran, the limit is 13.9 GB. Use
  fewer rows or columns (Sample Rows, Select Columns) and run it again."
- The label column must be in the steps' result. When a SQL step drops or
  renames it, training stops before it starts: "The label column 'class' is
  not in the data the steps before the Data Split made (columns: f0, ...,
  f7, species)". When a step makes or renames the label, pick it in the
  Data Split's **Label Column** (Properties): after **Execute Pipeline** it
  lists the columns reaching the Split, and "(the Data Input's label)"
  keeps the Data Input's. Before a run it is a text field. The compile
  report then warns "Label column 'species' is not in 'blobs'; the steps
  before the Data Split must make it (checked when they run)", and training
  uses it: "TabularLoader: Starting Arrow training: dataset=ds_sql_5, ...
  label=species".
- A supplied Validation or Test input (a Data Input on the Split's
  Validation / Test pin) goes through the Training branch's steps: they learn
  from the Train rows and transform its rows too (see
  [big_data.md](big_data.md)). A step between it and the Split is refused
  ("Test input goes through the Training branch's steps (they learn from the
  Train rows and transform its rows too); connect its Data Input to the Data
  Split directly").
- **Compile** opens the compile report (below); so does **Train** when it
  cannot start.
- While a graph trains (from the moment its data is being prepared), tests,
  or runs a pipeline or simulation, the graph and the project stay as they
  are: loading or clearing a graph shows *Graph replacement locked*, and
  File > New Project, Open Project, Open Recent and Close Project show
  *Cannot ... the project* with the reason. Stop the run, or wait for it,
  then try again.

## An empty canvas

An empty canvas says **Start a graph** and how to add a node, with three
buttons: **Search nodes** (the box at the top right of the canvas takes the
keyboard), **Open the Nodes panel** (drag a node from it onto the canvas)
and **Open an example graph** (the five task starters of the start page;
it opens as *Untitled graph*, so Save asks for a name).

## Find in the graph

Ctrl+F (or View > Find in the graph) opens a bar in the canvas's top-left
corner. It finds nodes by name, by type and by any setting's name or value
("dense", "relu", "0.001").

- Every match gets a ring on the canvas, the current one stronger, and the
  view centres on it at the zoom it is at.
- **Enter** or **F3** goes to the next match, **Shift+Enter** or
  **Shift+F3** to the previous one, and **Esc** closes the bar, while you
  type.
- "2 of 3" says where you are; "No match" when nothing matches.

## Questions and notices

- **Clear the canvas?** names how many nodes and links go; Ctrl+Z brings
  them back.
- Connecting a 4-D output (Conv2D, a pooling layer) to a layer that takes
  a flat input asks **<layer> needs a flat input** and offers **Insert
  Flatten**. Ctrl+Z removes the Flatten again.
- **Nothing to save yet** when Save or Save As is pressed on an empty
  canvas. File > Save still saves the project.
- **Graph busy** when a load or clear is refused while a run uses the
  graph.

## A narrow canvas

Nothing is cut off. When the row is too narrow, it folds in this order, and
comes back as it widens:

1. The mode becomes a menu with the mode's whole name (**Code Gen**),
   **View** drops the zoom from its button, and the simulation buttons keep
   their icons only.
2. Then, one at a time, into **More** at the end of the left side: Local
   Debug, Compile, View, Edit, Code (with Execute Pipeline), the framework,
   Save. **More** shows each as a section with its own items.
3. Then a run's line shortens to its state (*Epoch 3 / 10*, *Paused 3 /
   10*, *Waiting*, *Update 196*); hover it for the whole line.
4. Then the mode goes into **More**, and last the graph's button keeps its
   icon and the unsaved dot (the window's tab names the graph; hover the
   button for its name and path). The Studio in the default layout is this
   narrow: the graph's button, **More** and **Train**.

Train (or a run's line with Pause / Resume and Stop) and a running
pipeline's progress always stay on the bar.

While Compile is in **More**, a dot on **More**'s corner keeps its state in
sight: green when the graph compiles, red when it does not, grey before the
first compile. Hover **More** for Compile's line ("Compile: Not ready to
train: 2 errors").

## The canvas footer

Along the canvas bottom: the node and link counts, the selection ("2
selected"), and what the canvas is doing: "Graph replacement locked: a Data
Input is loading its data" while a load or clear would be refused, else the
mode ("Data Pipeline mode · training nodes run in Train"). In a narrow
canvas the right side is cut with "..." rather than running into the counts.

## The compile report

Compile (F7) shows what the graph compiles to, in a window titled
"Compile: <graph>". The same window explains a stop: "Cannot start training",
"Cannot run Local Debug", "Cannot load the checkpoint".

- **The verdict** comes first: "Ready to train", "Ready to train, with 1
  warning", "Not ready to train: 1 error", or the reason something stopped
  ("Training did not start: the data could not be loaded"). Chips count the
  errors, warnings and notes, and how many layers run on the GPU.
- **Findings**: errors first, then warnings, then notes. Each names its node,
  with **Show on canvas**. Where the fix is known it is offered: **Open its
  dialog** for data that is not loaded or failed to load, **Open Data
  Studio** for the note that suggests checking the data (Explore, Validate)
  before a long training. A finding's code shows only when it has its own
  (for example CW-C-0201, data not loaded).
- **Local Debug**, above the buttons: the last Local Debug run on the graph
  as it is ("Local Debug: Passed at 18:40; the graph changed since", "Local
  Debug: Compile gate failed."), with **Open the Studio Debugger**; and
  **Run Local Debug (F6)** next to Close / Train. Shown in the Compile and
  the Cannot-start-training reports.
- **Where each layer runs**: one sentence ("All 6 layers run on the GPU"),
  counts for GPU, CPU, mixed, at risk, unsupported and unknown, and a row per
  layer in plain words. Layers that need attention come first, with what to
  do. Each row's **Details** keeps the reason code, the expected and fallback
  backends and what was observed.
- **Configuration**: layers, input per sample, outputs and the learnable
  parameters (the layers' AS COMPILED counts), then Data (dataset, where it
  is: in memory, on disk or loaded lazily, and the split), the training loop
  (batches, epochs, metrics, workers, seed, gradient accumulation) and
  learning (loss, optimizer and learning rate, labels).
- **Copy report** puts all of it on the clipboard as text for a bug report.
  When the graph is ready, **Train** sits next to **Close**.

When the data a step prepares before training may not fit in memory, *Memory
risk before training* shows the estimate, the free memory and the safe
budget. **Cancel** is the default; **Continue anyway** is the red text
button. Cancelling tells the Training Dashboard that training did not start.

Other failures without a panel of their own (Run Test refusals, loading a
checkpoint, an empty report) show their own title, the message, **Copy** and
**Close**.


## The Studio Debugger and the canvas

Under the debugger's status line: **Canvas** (back to the graph; the
debugger hides, F6 or View > Studio Debugger brings it back with its run),
**Compile report** (F7), and, once Local Debug passed on the graph as it is
now, **Train** (F5), with the result: "Local Debug: Passed at 16:52, graph
unchanged".
