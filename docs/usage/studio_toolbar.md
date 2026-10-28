# The CyxWiz Studio bar

The bar at the top of the CyxWiz Studio canvas, one row in groups.

## The graph (left)

- **The graph's name** (the file name, or *Untitled graph*). A dot after the
  name means the graph changed since it was saved. Hover for the full path.
  Click it for the graph menu:
  - **Save** (Ctrl+S): writes the graph's own file. A graph that has no file
    yet asks for a name.
  - **Save As...** (Ctrl+Shift+S): writes a new file; the open file stays as
    it was.
  - **Load...**, **Export...**, **Custom Node Editor**.
  - **Clear the canvas...**: asks first; Ctrl+Z brings the nodes back. The
    file changes only when you save.
- **Save** is also on the bar.

## A narrow canvas

Nothing hides behind a "more" menu; the bar folds by width, in this order,
and comes back as it widens: the labels go first (Code / Pipeline /
Training, the mode's actions as icons, Compile as a check mark and Local
Debug as "Debug" with short results such as "Changed"); then Edit and View
become one-level menus (**Edit**, **View · 100%**); then the mode becomes a
menu (**Pipeline**); last, Compile and Local Debug become coloured dots
(their result on hover, their menu on click) and the graph shows its file
name without the extension. The graph, Save, the Compile and Debug results,
Train and a running pipeline's progress always stay.

## The canvas footer

Along the canvas bottom: the node and link counts, the selection ("2
selected"), and what the canvas is doing: "Graph replacement locked:
training is running" while a load or clear would be refused, else the mode
("Data Pipeline mode · training nodes run in Train").

A starter opened from the start page, a pattern template and an imported
model open as *Untitled graph*: Save asks for a name, so the example files
next to the Engine are never overwritten.

Ctrl+S and Ctrl+Shift+S save the graph while the canvas is focused (in the
Script Editor they save the script); see Preferences > Shortcuts.

## Edit and view (left)

Icons, with their names on hover: **Select all** (Ctrl+A), **Duplicate**
(Ctrl+D) and **Delete** (Delete), dimmed until nodes are selected (hover
says "select nodes first"); zoom out, the zoom level (click it for the zoom
menu: zoom in / out, zoom to 100%, fit all nodes, minimap), zoom in;
**Fit** frames all nodes (F); **Minimap** (M, highlighted while shown).

## The mode (middle)

**Code Gen**, **Data Pipeline** and **Local Training** side by side, one
click each (hover for what each does). Next to it the mode's action: Code
Gen, the framework (PyTorch, TensorFlow, Keras, PyCyxWiz) and **Generate**;
Data Pipeline, **Execute Pipeline**, then its progress, the step it is on
(hover) and **Cancel** while it runs; then **Export**. A graph with
simulation or RL nodes shows **Run Sim** / **Train RL** / **Export ONNX**
here instead, with the sim time, its speed or the RL episode and reward.

## Compile, Local Debug, Train (right)

- **Compile** (F7) and **Local Debug** (F6) each show their last result.
  Click the name to run the step (Compile opens its report, Local Debug the
  Studio Debugger); click the result for its menu (the result in a line,
  **Open the compile report** / **Show the first finding on the canvas**,
  or **Run Local Debug** / **Open the Studio Debugger**).
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
  this graph". While training, Train becomes the epoch and **Stop**.
- A reopened project keeps each Data Input's settings, not its loaded data.
  **Train** loads the Data Inputs that feed the Data Loader first, as their
  **Apply** would (the Training Dashboard says "Loading Sentiment CSV before
  training..."), then trains. If one cannot be loaded, *Cannot Start
  Training* says why ("Could not load the data before training: ... file not
  found: ..."). HDF5 sources are still applied in their dialog.
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
- Graphs with simulation or RL nodes add **Run Sim** / **Stop Sim** (with the
  simulated time and speed), **Train RL** / **Stop RL** and **Export ONNX**.

## A narrow canvas

When the canvas is too narrow for everything, the edit, code and view tools
(then Local Debug and the simulation / RL tools) move into the **...** menu.
The graph, Save, zoom, Compile and Train always stay on the bar.

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

