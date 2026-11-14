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
  **Find in the graph** (Ctrl+F). The minimap fades while a node is under
  it (unless the pointer is on it), so the node stays readable.

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
  the node (a template opened without Apply trains at once). Local Debug
  and the other Studio Debugger runs (all but Runtime Trace) load it the
  same way first: the debugger says "Loading Metric blobs first; the run
  starts when it is loaded", and a Train afterwards finds it loaded.
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
- **Configuration**: layers, input per sample, outputs, the learnable
  parameters (the layers' AS COMPILED counts; BatchNorm's running statistics
  are kept state, not parameters) and about how much memory one training step
  needs, then Data (dataset, where it is: in memory, on disk or loaded lazily,
  and the split), the training loop (batches, epochs, metrics, workers, seed,
  gradient accumulation) and learning (loss, optimizer and learning rate,
  labels).
- **Memory of a training step (estimate)** breaks that number down: the
  weights, their gradients, the optimizer's state (Adam keeps two tensors per
  parameter, SGD without momentum none), kept state, and the activations of
  one batch (each layer's output and what it keeps for the backward pass:
  attention weights, recurrent gates, a convolution's unfolded input). It is
  counted from the model the graph builds, so every layer kind, extension
  nodes included, is in it. The device runtime and the training data come on
  top. When the device's memory size is known, the last row gives the share;
  above 100% the tile says so and suggests a smaller batch or input. The
  estimate never blocks training.
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

**Full size** (the expand button beside Saved runs) gives the debugger the
whole width. It hides the side panels (Asset Browser, Nodes, Patterns,
Properties, Info, Wallet) while the debugger is shown. Press it again,
switch to another tab or close the debugger, and the panels it hid come
back.

### Which sample Local Debug runs

Local Debug runs one item of your data through the model, forward and
backward. Pick it with **Sample** (- / +) next to Run: Sample 7 is item 7 of
the dataset (row, image or clip), in the dataset's own numbering, whichever
part of the split it falls in. The item is prepared the way validation sees
it: the data steps before the Data Split, Resize / Crop and Normalize, the
label form, but no random augmentation (flips, crops, audio noise), so the
same sample gives the same numbers every run.

The Overview's **Sample** line names what ran and its part of the split,
for example "row 7 of blobs, test" or "image 5 of cat_dog, train". Images,
tables in memory or on disk, sparse features and audio work this way. Token
tagging sequences and data loaded through the old dataset path still run a
synthetic sample; the Sample line then says "a synthetic sample (seed 1337)"
and a warning gives the reason. Data a reopened project has not loaded yet
loads first, as for Train.

### Layer capture

Local Debug and training can keep what every layer held: its input and
output, its weights (with BatchNorm's running statistics as kept state) and
the gradients of the weights. The layer views of the Studio Debugger read
them. Capture is off until you turn it on, with **Layer capture** in the
Studio Debugger's toolbar (for this session) or in **Preferences > Studio
Debugger** (your defaults):

- **When**: **Off** (Local Debug and training run exactly as without it),
  **Local Debug runs**, or **Training, every N epochs** (N = 5 to start).
  Training also captures after the last epoch, an early stop's included,
  and Local Debug runs are captured too.
- **What, for every layer**: activations, parameters, gradients (all ticked to
  start).
- **On**: **The picked sample** (the debugger's Sample number) or **A batch
  of 8**, the picked item and the seven after it (the picked item alone when
  the data ends before them).
- **Keep the captures of the last N runs** (10 to start). Older runs keep
  their report; their captures are deleted.
- **Feature-map colour**: **Heat** (black through red to yellow) or
  **Grayscale**, for every activation layer; each view can switch it.
- **Feature-map order**: **Strongest first** (largest mean), **Highest peak
  first** (largest value) or **Channel order**.
- **Where it looks: method**: **Occlusion** (the default) or **Saliency**.
- **Saliency shows**: **Gradient size** (the default) or **Gradient ×
  input**.

Preferences applies and saves at once, for every project (the
`studio_debugger` section of the Engine's settings file).

**Layer capture** in the toolbar names what the next runs keep ("Layer
capture: every 5 epochs") and opens the same When / What / On choices,
starting from your defaults. Changes there apply to this session's next Local
Debug and training runs (a run in progress keeps what it started with); the
Heat / Grayscale switch in Model > Layers belongs to the same settings. The
panel also gives the size of one snapshot from the compiled model on the
canvas: the input and every layer's output for the captured samples, the
parameters with their kept state, and the gradients, plus the total for a
training run of the graph's epochs. **Save as my default** writes the
session's settings, the colour included, to Preferences; **Use my defaults**
drops the session's changes.

A training snapshot runs the sample once, forward and backward, with no
weight update and with the model in evaluation mode (BatchNorm on its running
statistics, no dropout). Training therefore comes out exactly the same with
capture on or off. The snapshot's gradients are those of that evaluation
pass, not of a training batch.

A capture stays on the compute device while the run is open, up to 256 MB per
step (tensors past that are left out and counted). It is also saved with the
run in the debug run store, under `debug_runs/studio/<run>/layers/`: Local
Debug as `local_debug`, training snapshots as `epoch_0005` and so on. On the
ResNet example one step takes 25 MB for all 36 layers. Data that cannot hand
an item over by number (token tagging sequences, sparse features in
training) has no training snapshots; the log says why.

A training run with snapshots also saves `training.json` beside them, rewritten
at each snapshot: every epoch so far (training loss and accuracy, or the MAE
for a regression, and the validation loss and metric when validation ran),
the graph's name, and the snapshot items' names, labels and classes.

### Training runs and their snapshots

**Overview > Runs** lists the saved runs, newest first: Local Debug runs and
training runs that kept layer snapshots. A training run's line names its
graph, its epochs, its best validation (and at which epoch) and its snapshot
epochs, for example "20 epochs · best validation 92.97 % (epoch 18) · 4 layer
snapshots (epochs 5, 10, 15, 20)". A Local Debug run's line says whether it
passed and its issues; **Details** gives its traces, events, graph, device,
transfers and where it is saved. Click a run to show it.

**Overview > Compare** for a shown training run is its snapshot table: each
snapshot's epoch, training loss, validation loss, validation accuracy (or
MAE) and size on disk, with **Open in Model > Layers**. Snapshots saved before
the epochs were kept say "not recorded". For a Local Debug run, Compare is
the comparison with the latest run, as before.

### Model > Layers

**Model > Layers** (the Model section's first view) shows the shown run's
layer capture. A live Local Debug run shows its own capture; a run picked in
**Saved runs** reads its capture from the run store. Without a capture the view
says how to get one.

- The line above the list names the picked item and the model's answer
  ("dog.11289.jpg · label class 1 · predicted class 0, 63.3 % · 36 layers
  captured · image 0 of cat_dog, train"), then, for a batch of 8, an **Item**
  stepper over the batch. The prediction follows the loss and the last layer:
  one logit through a sigmoid (BCE with logits), logits through a softmax
  (cross-entropy), probabilities from a Sigmoid or Softmax layer, or the value
  for regression. Class names come from the data (as typed on the Data Input);
  names that are only numbers read as "class 1". When the run has more than
  one capture (training snapshots), **Snapshot** steps through them (‹ epoch
  10 ›, or pick one from the list), with "n of m snapshots" and that epoch's
  validation; the picked layer, tab and channel stay as you step. **Compare
  with** picks another snapshot of the run and draws the shown view for both,
  side by side: feature maps with the same channels in the same order,
  filters in the same order, weights and gradients with the same rows and
  kernel slice (each on its own scale), values and units. Where it looks stays
  one snapshot at a time.
- The **layer list** names every captured layer, its kind and its output for
  one sample ("64 × 64 × 32"), in sections named by the graph's canvas groups
  (as they were when the run started). The kind adds what the tensors show:
  the kernel ("Conv2D 3 × 3"), a stride from the input and output sizes
  (", stride 2") and a Dense layer's parameters ("Dense, 257 parameters").
  To get headings, group the nodes on the canvas and name the groups:
  - select the nodes and press **Ctrl+G** (or right-click the canvas >
    **Create Group**);
  - double-click the group's name band to rename it ("Stem", "Stage 1",
    "Head").
  The next run shows the names. Layers in no group are listed without a
  heading. Details: [Groups, frames and names on the canvas](canvas_groups_frames.md).
- The **views** follow the layer's tensors, not its kind, so every layer gets
  the views its data supports:
  - **Feature maps** (an output of height × width × channels): 16 channels for
    the sample in the chosen **Order**: strongest first (largest mean),
    highest peak first (largest value) or channel order; **All N channels**
    shows every channel (up to 256). **Colour** switches Heat and Grayscale;
    the scale runs from 0 to the 99.5th percentile, so one bright pixel does
    not darken the rest. The count of dead channels (zero everywhere) is
    under the maps. Click a map to enlarge it in the Inspector.
  - **Units** (a flat output): each unit's value for the sample; hover a bar
    for its number.
  - **Values**: a histogram of every output value (log counts), with min, max,
    mean, the share of zeros and any NaN or Inf.
  - **Weights** and **Gradients**: the layer's main parameter as a heatmap in
    the blue (negative) to red (positive) scale. Conv weights [kh, kw, cin,
    cout] show one row per output channel over kernel × input channels; Dense
    weights [out, in] one row per output. **Slice** shows one kernel position
    instead: output channels by input channels at, say, (1, 2) of the 3 × 3
    kernel. Hover a cell for its index in the tensor and its value; − and +
    page through the rows of a large matrix.
    The histogram and numbers are under it.
  - **Filters** (the first convolution on an image, 1 or 3 input channels):
    each output channel's kernel drawn as a small picture, RGB for a colour
    image, grey for one channel. Each filter is scaled to its own range over
    all its colours, so its colour balance shows; the number under it is its
    weight norm (hover for the filter's number). This is the pattern the
    channel responds to: where the image looks like it, that channel's
    feature map is bright. A Local Debug run uses a fresh model, so its
    filters are the starting weights (noise); trained filters (colour edges
    and blobs) show in a capture from a training run (Layer capture:
    Training, every N epochs). Only the first layer sees the image's
    colours, so deeper convolutions have no Filters view.
  - **Where it looks** (the layer the model's input enters): which parts of
    the input the model relies on. **Method** picks how: **Occlusion** or
    **Saliency**.

    **Occlusion** covers each piece of the input in turn and runs the model
    again; the view shows how the picked output's score moved:
    - an image: a 12 × 12 px patch every 4 px (196 positions on 64 × 64),
      filled with the image's mean per channel; the change is drawn over the
      picture;
    - tokens (a text graph): each token replaced by the pad id, padding left
      out; one chip per token, by its id in the vocabulary;
    - table features: each feature replaced by its mean over up to 512 of the
      graph's prepared rows, spread over the data (the data steps as Local
      Debug runs them); a bar per feature.
    **Explain** picks the output: the predicted class, the true class, or
    any other class from the list. **Compute** runs the forward passes on the
    device, 64 at a time, off the UI thread; **Stop** ends it. Blue means
    covering that piece lowers the score, so the model relies on it; red
    means it raises it. The headline names the piece that lowers the score
    most, with the score before and after. A box, a mask region and the next
    token are explained once those heads exist.

    Where it looks runs the run's model again: it is rebuilt from the
    capture's weights (keep **Parameters** in Layer capture) and the
    canvas's graph, which must be the run's (a saved run of a graph you
    changed since says so; run Local Debug again). A Local Debug run's
    model has its starting weights, so its map shows where an untrained model
    looks; a capture from a training run (Layer capture: Training, every N
    epochs) shows the trained model.

    **Saliency** is the gradient of the picked score with respect to the
    input: how much the score moves for a tiny change of each pixel, token
    or feature. It takes one forward and one backward pass on the device,
    so it is instant. **Of** says what is differentiated: the class's logit
    (the score before the softmax, which does not saturate near 100 %); for
    one sigmoid output, the logit, or minus it for the other class.
    **Show** picks what is drawn:
    - **Gradient size** (the default): where a small change moves the score
      most, in the Heat scale. An image: per pixel, the largest of its
      channels' gradients. Tokens: the length of the gradient at each
      token's embedding (token ids have no gradient, so it is taken at the
      Embedding's output); padding is left out. Table features: the
      gradient's size.
    - **Gradient × input**: signed, in occlusion's blue and red. Does this
      element push the score up or pull it down, given its value? An image:
      summed over the channels. Tokens: the gradient · the embedding.
      Features: the gradient × the value.
    The headline names the most sensitive element (or the one that pushes
    the score most); hover the map or a chip for its value. Occlusion and
    saliency answer different questions (removing a whole piece versus a
    tiny change), so they often agree but need not. **Save as my default**
    keeps the method and Show; both are also in **Preferences > Studio
    Debugger**. Saliency is checked against PyTorch
    (`cyxwiz-engine/tests/computation_truth/reference/generate_saliency_fixtures.py`).
  - **Top images** (every layer with channels or units; **Top samples** when
    the model's input is not pictures): the items from the data that excite a
    channel most. **Compute** takes 320 items spread over the data (every
    k-th, as validation prepares them, so a folder sorted by class gives both
    classes), runs them through the run's model in batches on the device
    (**Stop** ends it) and shows, per channel, the 9 strongest. **Channels**
    picks the rows:
    - **The Dense head's strongest votes**, when the model's last layer is a
      Dense reading these channels (after a Global Avg Pool): for each class
      the two channels with the largest weight towards it, with the weight
      and the class it votes for;
    - **The clicked channel**: the one last clicked in Feature maps or Units;
    - **The 16 strongest for this sample** (conv layers).
    On a conv layer **Ranked by** picks a channel's strength per image: its
    **Maximum** over the map (it fires somewhere) or its **Mean** (it fires
    all over). The settings sit in one bar with **Compute** (or **Stop**) on
    the right. While it runs, a card shows its steps, each ticked as it ends:
    **Load the data** (a reopened project's Data Inputs load first, as for
    Train; it then waits for the graph to compile and continues by itself),
    **Prepare 320 images**, and **Run this snapshot's model** with a bar and
    the count, plus the time so far. The result is one card per channel: how
    it was picked, a summary ("4 × class 0, 5 × class 1 · 15.9 to 11.7") and
    the 9 pictures, each captioned with its rank and label and its value
    below. Click a picture to see it large in the Inspector with its file,
    label, rank and value, and **Make it the picked sample** (run Local Debug
    to see it). Results are not saved. Patches (the part of the image the
    channel sees) come later.
  - **Contributions** (a Dense layer): how its output was computed for the
    picked sample, output = the sum of w × x + b, from the captured input and
    weights of that step (the same numbers the forward used). On the model's
    head **Explain** picks the output (the predicted class first, then the
    others); on a hidden Dense **Unit** picks the unit (this sample's
    strongest first, any unit from the list). The 8 largest terms come first,
    with w, x and w × x (blue pulls the output down, red pushes it up); the
    line under them gives the sum, the bias and the value, then the other
    logits and the prediction (a head) or the activation after it (a hidden
    layer), and what the other inputs add up to. Click a row: the Inspector
    says where that input comes from (for a Flatten of a conv stage, the
    layer, row, column and channel).
  - **Embedding** (a flat output with more than one feature, such as a
    Flatten, a Dense or a pooled vector): where the items of the data land
    in this layer's features. **Compute** takes 320 images (or samples)
    spread over the data, runs them through the snapshot's model and
    projects the layer's output to 2-D with the **Method** you pick: **PCA**
    (in a moment; it keeps the large distances, and the caption says how
    much of the variance the two components hold), **t-SNE** or **UMAP** (a
    few seconds; they keep neighbours together, and the gaps between their
    clusters mean little). **Colour by** the **True class** or the
    **Predicted class**; with the predicted class the items the model gets
    wrong are ringed in amber. The white ring is this sample (when it is
    one of the 320, that point is ringed rather than drawn twice). The steps show
    in a card while it runs, as for Top images (the data loads first on a
    reopened project). Hover a point for its item, label and prediction;
    click it to see it in the Inspector, with **Make it the picked sample**.
    The legend sits beside the plot, or below it when the column is narrow.
    Separate clouds per class mean the layer already tells the classes
    apart; mixed ones mean the layers after it still have work to do.
    Results are not saved.
  - **Unit weights** (a Dense layer that reads the model's image flattened:
    only Flatten, Dropout or Reshape before it, and as many inputs as the
    image has pixels and colours): each unit's weights drawn as the image
    they read. A grey image is drawn red where a bright pixel raises the
    unit and blue where it lowers it; a colour image as the colours that
    raise it (bright) or lower it (dark); each unit on its own scale.
    **Order** picks the largest weight norm first or unit order; the first
    32 units show, and **All N units** shows every one. Hover a unit for its
    weight norm and range; click it for the Inspector: the unit large, its
    norm, bias and value for this sample, and **Top images of unit N**,
    which opens Top images on that unit. A data table (a CSV of pixel
    columns) is not an image to the model, so its first Dense has no Unit
    weights.
- Views that run the model again (Where it looks, Top images, Embedding)
  rebuild it from the canvas's graph. With no graph open, or while it
  compiles or has errors, the view says which and continues by itself once
  the graph compiles cleanly.
- The **Inspector** shows the sample as the first layer received it (after
  the data steps), its name and label, the prediction with how it was read
  (from the evaluation pass, with dropout off and BatchNorm on its running
  statistics, the same pass Where it looks scores; the run's own training
  step only when the model cannot be rebuilt, and then it says so)
  ("logit -0.5472 → sigmoid 0.3665"; the raw output when the loss gives no
  rule), and the clicked feature map enlarged with its mean, maximum and rank
  (with Filters: how to read a filter).

The colour and the order are part of the session's settings (Layer capture in
the toolbar): **Save as my default** keeps them, and Preferences > Studio
Debugger > Layer views sets them too.

The views' own choices work the same way: Where it looks (Method, Show), Top
images (Channels, Ranked by), Embedding (Method, Colour by) and Unit weights
(Order) each carry a **Save as my default** link beside their controls, and
Preferences > Studio Debugger > Layer views lists them all. A change in a view
lasts for the session until it is saved. A layer that does not offer the saved
Top images channels (no Dense head after it, or not a feature-map layer)
starts on the first choice it has.

Everything is computed on the compute device; only what a view draws (the
shown maps, a page of heatmap rows, a histogram) is copied to the host.
