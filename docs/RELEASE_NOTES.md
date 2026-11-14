# CyxWiz Studio release notes

Notable user-facing changes, newest first. Engineering detail lives in the
private Data Studio tickets (`tofix*/track*`).

## Unreleased

### Changed

- **A data file opens in the Table Viewer.** Double-click a CSV (or TSV,
  Parquet, JSON Lines, Feather, ...) in the Asset Browser to look at its
  first rows in the Table Viewer, read in a task; nothing is loaded into
  the graph. A file a Data Input loaded opens as that dataset. Before, a
  double-click added a Data Input node; that is now **Create Data Input**
  in the file's menu and preview. The menu's **Open** for data files is
  **Open as text**.
- **The Nodes panel fits its column.** Cards take as many columns as the
  panel holds and names use two lines ("Depthwise Conv2D", "Instance
  Norm") instead of being cut after ten letters; a narrow panel no longer
  cuts a third column at its edge. The "Real" chip over every icon is
  gone: a small dot marks only the nodes that run in part or are blocked.
  The panel's search lists nodes named by the words first, and nodes that
  only mention them under **Related**.
- **New nodes and notes go to a free spot.** A node added from the Nodes
  panel or the canvas search, a Data Input from the Asset Browser and a new
  note no longer land on top of the nodes at the centre of the view; the
  view moves to show them when they are outside it (the right-click menu
  still adds where you clicked). The minimap fades while a node is under
  it.
- **Notes can be edited, deleted and undone.** Right-click a note for
  Edit, Minimise and Delete; the Delete key removes a selected note; the
  note dialog has a title, text, a colour and Save / Cancel / Delete
  (its buttons fell below the window, so a note could not be deleted);
  adding, moving, minimising, editing and deleting a note are each one
  Undo step.
- **The Console shows a terminal's output.** In the default layout the
  Console is taller (the Viewport is a tab beside it instead of a strip
  below), the Python REPL's empty input is two lines, and the REPL and
  Commands inputs have no outline.
- **The Info panel follows Undo.** It kept showing a node that Undo or
  Delete had removed from the canvas.
- **The Theme Editor shows every colour the Engine draws, and every row
  changes the screen.** Engine colours: the accent, the 24 node families,
  the data types (pins, links, plot series) and the status words, with a
  warning when one is hard to read on the window. Surfaces: window, text,
  fields, buttons, bars, tabs, tables. Canvas: the grid (the old Node Editor
  tab had 19 rows of which 14 changed nothing). Colours are typed: a hex
  field and R, G, B with - / + steps (no drag fields); the Shape tab uses
  typed numbers instead of sliders.
- **Colours follow the chosen theme.** Train, OK, Publish and every
  primary button, selection, check marks, links, the selected tab's line
  and the canvas box selector use the theme's one accent: purple in
  CyxWiz Dark and Cream, blue in CyxWiz Light, orange in Unreal Engine,
  each preset its own (before,
  the purple was fixed on every theme while Unreal marked choices in
  orange). Text on the accent picks black or white to stay readable.
  Secondary buttons (Save, Edit, Cancel ...) are the theme's Button fill,
  a step off the window, in every theme. A saved custom theme keeps the
  accent, the node colours, the data-type colours and the status colours.
- **Safetensors import reads each tensor's type.** Importing a
  `.safetensors` model took every tensor for float32. Float16 and float64
  tensors are now converted, integer tensors are skipped with a warning, and a
  header whose sizes do not fit the file is refused with the reason.
- **Frozen layers are kept in checkpoints and exports.** A frozen layer (today
  an Embedding with **Freeze Loaded Weights**) was left out of the model's
  parameter list, so checkpoints, exported models and "Start from" lost its
  weights. It is now saved and loaded with the rest; only the optimizer skips
  it. A checkpoint written before by a model with a frozen layer lacks those
  weights and no longer loads: train or save it again. In `pycyxwiz`,
  `SequentialModel.get_parameters()` returns frozen layers too;
  `get_trainable_parameters()` and `get_frozen_parameter_names()` are new.
- **"Start from" works on graphs with branches.** Starting a run from a
  checkpoint, and leaving normalisation, bias and embedding weights out of
  weight decay, needed a plain layer stack; both now work on residual and
  other branched graphs.
- **Saved defaults for the layer views' choices.** Top images (Channels,
  Ranked by), Embedding (Method, Colour by) and Unit weights (Order) each
  have a **Save as my default** link beside their controls, and
  Preferences > Studio Debugger > Layer views sets them too. A change in a
  view lasts for the session until it is saved.
- **The Studio Debugger reads image MLPs right.** A model that flattens the
  image first (no convolution before its Dense layers) takes images in
  another layout than a CNN; the debugger now keeps one layout for both. On
  such a model the Inspector shows the sample's picture, the views say Top
  images, and Where it looks (occlusion and saliency), Top images and
  Embedding run the model on the right pixels. Captures saved before are
  read the same way.
- **Saliency through BatchNorm1d.** An input gradient through a model with
  BatchNorm1d in eval mode (saliency) stopped with "backward requires a
  matching training-mode forward pass"; it now uses the running statistics,
  as PyTorch does.
- **PCA, t-SNE and UMAP are fast on wide data.** With more features than
  samples (a layer's 8,192 values for a few hundred images) PCA works from
  the samples' Gram matrix; it took minutes before and now takes well under
  a second, with the same result. t-SNE and UMAP start from it too.
- **Neural Architecture Search counts like the Model window.**
  - Each version is compiled the way the canvas compiles; its parameters
    come from the built model and its FLOPs from each compiled layer, so
    NAS and Tools > Model > Summary give the same numbers. Before, an MLP
    of 109,386 parameters was counted as 158,570.
  - A version that does not compile is not scored, and the reason is shown.
  - "Start from a suggestion" is gone: its generated models had no data to
    be compiled from. Guide: `docs/usage/neural_architecture_search.md`.
- **Neural Architecture Search changes the model, not a copy of it.**
  - **Apply to CyxWiz Studio** (and **Apply Best Architecture**) now
    replace the canvas graph: unchanged nodes keep their place and group,
    new ones are placed between their neighbours, and Ctrl+Z undoes it.
    Before, a second copy of the whole graph was pasted on the canvas with
    the same node ids, which broke the canvas and the compile.
  - Add / Remove Layer keep the graph linked; Remove Layer and Change
    Units no longer touch the head (the layer that sets the class count).
  - **Swap Layer** now works (it did nothing).
  - Added and swapped layers fit their place (no Conv2D in a Dense model).
  - The score counts model layers only, names every layer type, and counts
    Add / Concatenate joins as skip connections.
  - Patterns added to the canvas get fresh node ids.
  - Guide: `docs/usage/neural_architecture_search.md`.
- **Canvas: groups can be named, frames copied, nodes renamed.**
  - Double-click a group's name band (or right-click it > **Rename...**) to
    name the group. Its right-click menu also has **Ungroup**.
  - A node's **Rename...** now renames it (it did nothing before).
  - A frame's **Copy** and **Cut** take the frame and the nodes inside it,
    and **Ctrl+V** pastes them in the same layout. Pasted nodes now keep
    their layout relative to each other.
  - A node inside a frame keeps its own clicks and right-click menu.
  - **Generate** says why when the graph cannot be turned into code,
    instead of only writing it to the log.
  - Guide: `docs/usage/canvas_groups_frames.md`.
- **Studio Debugger Local Debug runs the sample you pick**, item N of your
  data prepared the way validation sees it (fixed steps and Normalize, no
  random augmentation), for images, tables in memory or on disk, sparse
  features and audio; the Overview's Sample line names it and its part of the
  split ("row 7 of blobs, test"). Other data kinds still run a synthetic
  sample and say so. Local Debug also runs branched graphs (skip connections,
  Add, Concatenate) by their graph plan, as training does. On a reopened
  project, Local Debug (and the other Studio Debugger runs) loads the graph's
  data first, the way Train does, instead of stopping with "Data is not
  loaded".
- **Device names read as names everywhere.** ArrayFire reports CUDA and OpenCL
  devices as "NVIDIA_GeForce_GTX_1050_Ti"; the backend now gives them with
  spaces, so the Training Dashboard, the Studio Debugger, logs and Compile show
  "NVIDIA GeForce GTX 1050 Ti". Plugin SDK: `cyxwiz::DeviceDisplayName`.
- **Compile (F7) estimates a training step's memory for every layer kind.**
  The old note counted only Dense layers on image graphs (3 MB for the 1.2 M
  parameter ResNet example, and an error above a guessed 3 GB). The estimate
  now comes from the built model: weights, gradients, the optimizer's state,
  kept state and the batch's activations, including attention weights,
  recurrent gates and convolution columns; on the ResNet example it is
  1.35 GB against 1.41 GB measured. It shows as a tile and a breakdown, with
  the device's share when known, and never blocks training. The parameter
  count leaves BatchNorm's running statistics out. Plugin SDK:
  `Module::CachedElementsPerSample` and `BackwardWorkingElementsPerSample`
  let a module report what it keeps; `Optimizer::StateTensorsPerParameter`.
- **BatchNorm's running mean and variance are state, not parameters missing
  a gradient.** Local Debug no longer reports them (24 on the ResNet example)
  as "parameters missing gradient", and Gradient Health no longer says needs
  attention for them. Plugin SDK: `Module::GetStateNames()` names the
  `GetParameters` keys a module keeps but does not train; the gradient check
  and training skip them. Python modules declare such a tensor with
  `add_state(name, value)` and update it with `set_state`.
- **Audio augmentation applies to training clips only.** The AudioAugmentation
  node's noise, time stretch and pitch shift used to alter validation and
  test clips too, so their metrics changed from run to run.
- **The CyxWiz Studio bar is one row that never hides Train** (guide in
  `docs/usage/studio_toolbar.md`). What does not fit folds into **More**
  instead of being cut off; Compile and Local Debug keep their names.
  - **Code** (Generate into Script Editor, Save as .py) replaces the gear
    and Export icons. It says why when the graph cannot be turned into
    code, and a line under the bar says what was written where and when.
    Generated files carry the time they were generated (they carried the
    time the Engine was built).
  - The framework shows only in Code Gen mode; in the other modes the Code
    menu names the one it uses.
  - **Pause** / **Resume** and **Stop** are on the bar while a model or a
    PPO policy trains. Train is greyed, with the reason, while a Data Input
    loads its data.
  - The Studio's tab names the graph, with a dot while it has unsaved
    changes.
  - An empty canvas says how to start, with Search nodes, the Nodes panel
    and the example graphs.
  - **Find** (Ctrl+F) sits in the canvas corner, marks every match and
    keeps Enter, F3 and Esc working while you type.
  - Save on an empty canvas says why nothing was saved; the graph menu has
    **Export a copy of the graph**.
- **Studio fixes.**
  - Delete no longer removes the selected nodes while you type in another
    panel.
  - Undo removes a Flatten that Insert Flatten added.
  - Duplicate keeps what you copied.
  - Zoom stops at 25% and 200% and keeps the view centred; + and - zoom.
  - Local Debug cannot start twice.
  - Clear and Load wait while the graph is busy, and say why.
  - Nodes, notes and frames dropped on the canvas land under the pointer.
    Notes in saved graphs show a little higher than before.
  - A graph opened from the start page, the Asset Browser or the command
    line comes to the front instead of staying behind the Script Editor.
  - Ctrl+Z right after Load no longer brings back the previous canvas
    under the loaded graph's name (Save would have written it over the
    loaded file).
  - Moving nodes is an undo step (one per drag); after an undo the links
    keep their compile colours instead of turning grey.
  - When Compile folds into **More**, a dot in Compile's colour on **More**
    keeps its state on the bar.
  - Undoing back to the saved layout clears the unsaved dot (positions are
    saved to a hundredth of a grid unit).
- **The menu bar's right end** (guide in `docs/usage/menu_bar.md`): your
  account sits in the window's top-right corner as initials, name and plan
  (it was a bare icon with the email as a tooltip), its menu says who you are
  and adds **Servers...**; the project name opens Show in Explorer, Copy Path
  and Close Project. New items: File > Show Project Folder, View > Full
  Screen (F11), Help > Documentation (F1), API Reference, What's New, Known
  Issues and Open Logs Folder (the first two were planned).
- **Studio Debugger > Runtime > Timeline zooms and clusters** (guide in
  `docs/usage/monitoring.md`): drag across the lanes to zoom, **Whole run**
  to go back; short events that would overlap merge into a bar with their
  count (hover lists them, click zooms in); a long event others happened
  during is a band under them; only failed events are red and warnings
  amber (a `device_resident` loss step was drawn red); every event the
  trace keeps is drawn, not the last 42 a thread; a legend.
- **The font atlas uses a quarter of the GPU memory** (32 MB instead of
  128 MB): glyphs are no longer oversampled (they are pixel-snapped, so it
  bought nothing), and the five extra icon packs keep only the node icons
  the canvas draws.
- **A seed repeats on every operating system and device** (guide in
  `docs/usage/seeds_and_reproducibility.md`). Splits, folds, batch order,
  balancing, augmentation, samples and weight initialisation draw the same
  numbers on Windows, Linux and macOS and on CPU, CUDA and OpenCL; until now
  they repeated only on one platform. **The same seed now gives different
  splits, batches and initial weights than in earlier releases, once**, so a
  rerun old project's metrics can move a little.
  - Audio augmentation repeats from the DataLoader seed, and Add Noise no
    longer adds the same noise to every clip (`add_noise(seed=...)`).
  - Plugin SDK: new header `cyxwiz/portable_random.h` (shuffle and draws
    that are the same with every compiler).
- **Tools > Monitoring is one Monitor window** (guide in
  `docs/usage/monitoring.md`, example `examples/python/system_memory.py`)
  with tabs System, Memory, Loaded data and Frames. It replaces Memory
  Visualization and System Monitor.
  - Every figure is measured, on Windows, Linux and macOS. The old System
    Monitor showed made-up numbers on Linux and macOS, and the old memory
    charts were never filled in.
  - The compute device card says what the Engine holds on the device and,
    for NVIDIA cards, what the device has free for every program. Other
    drivers read "Not reported by this driver".
  - The Engine's memory is recorded all the time (process, device, tables,
    tensors), so the Memory chart has a history when you open it. Export
    asks where to save.
  - **Clear Caches...** opens at any time and lists every cache with its
    size (it did nothing before the first training run). **Free Unused
    Memory** (was Garbage Collection) says what it freed.
  - The Engine no longer re-lists the compute devices several times a second
    while these windows are open.
- **Tools > Diagnostics switches are kept** between sessions and are also in
  Preferences > General.
  - **Log Frame-rate Changes** (was Log Idle Mode Transitions) logs every
    change of pace: idle, another window in front, minimised.
  - **Log Every Python Command** (was Verbose Python Logging, which changed
    nothing) logs each command with its duration.
  - **Frame Time Overlay** shows the whole frame and the Engine's own
    interface-building time as two figures (they shared one label).
- **Studio Debugger > Runtime**: the Timeline is drawn on a time axis (new
  runs), the memory bars are named for what they measure (host tensors,
  with a peak per run), and the Crash view no longer reads a file every
  frame.
  - Plugin SDK: new header `cyxwiz/device_memory.h`
    (`FreeUnusedDeviceMemory` answers what it gave back;
    `ReleaseUnusedDeviceMemory` is unchanged). Python: `pycyxwiz.system`.
- **Tools > Model is one window** (guide in `docs/usage/model_tools.md`,
  example `examples/cyxgraph/model_tools`) with tabs Summary, Learning
  rate, Explain, Health and Projector; its menu items open it on a tab.
  - Summary counts parameters and multiply-adds the way PyTorch does
    (checked on five models), with the memory of a batch's activations.
  - Learning rate runs a real range test on a copy of the model and
    suggests a rate (steepest fall or lowest / 10); **Use** writes it into
    the optimizer node.
  - Explain shows Grad-CAM, saliency and Integrated Gradients for any kept
    test row, as Captum computes them, on the model's own device.
  - Health is recorded during every training: weight and gradient size by
    epoch, a weight histogram and dead ReLU units; the Training Dashboard
    warns and links to it.
  - Projector maps a layer's features of the test rows (PCA, t-SNE, UMAP).
  - The Model Summary, Architecture Diagram, Learning Rate Finder, Grad-CAM,
    NAS and DNN Inference windows are gone (the diagram is the canvas; NAS
    had no search; DNN Inference ran fixed demo networks). The Grad-CAM and
    Saliency Map nodes are no longer in the node list.
- **Results > Test has Errors, Threshold and Calibration**: the mistakes by
  pair with their images, thresholds by F1, recall or cost, a confidence
  floor for several classes, and temperature / Platt / isotonic calibration
  (checked against scikit-learn). A threshold and a calibration can be kept
  with the model: the next test applies it, checkpoints and exports carry
  it, and the local inference server returns calibrated probabilities.
- A graph that reads a CSV of pixel columns into a CNN through a leading
  Reshape (784 -> 28,28,1) trains, and compiles before its data is loaded.
- **pycyxwiz.model**: the Model window from Python on a `Sequential`:
  `summary` (the same multiply-add counter), `lr_range_test` /
  `LearningRateRange`, `saliency`, `smooth_grad`, `integrated_gradients`,
  and the calibration fits (`reliability`, `fit_temperature`, `fit_platt`,
  `fit_isotonic`); numpy optional. Example `examples/python/model_tools.py`.
- **Plugin SDK**: new headers `cyxwiz/calibration.h`,
  `cyxwiz/learning_rate_range.h` and `cyxwiz/model_cost.h`
  (`CountMultiplyAdds`, `ModuleCostInput`); the convolution, dense,
  recurrent and attention modules gain a non-virtual `CostInput`.
  Layout changes: `SequentialModel` gains a member (`SetForwardObserver`),
  and `GradCAMResult` is rewritten (the `overlay` field is gone; maps are
  the gradient at the layer's output, as pytorch-grad-cam). A plugin that
  subclasses or holds a `SequentialModel` by value, or reads
  `GradCAMResult`, must be rebuilt against these headers.

- **Tools > Optimization and Calculus is one window** (guide in
  `docs/usage/optimization_calculus.md`). Its six menu items open it on
  Optimizers, Curvature, Linear, Quadratic, Derivative or Integral. It takes
  typed formulas with exact derivatives, follows the training optimizers on
  a formula's contour map, solves linear and quadratic programs of any size
  with HiGHS (shadow prices, infeasible and unbounded in words), and gives
  adaptive integrals with an error estimate. The six separate windows are
  gone.
- The Template nodes GD Visualizer, Convexity Analyzer, LP Solver, QP
  Solver, Differentiation and Integration are no longer in the node list. They had no executor; the window does that
  work, and the Portfolio Weights node brings a program into a graph.
- **Plugin SDK (`cyxwiz/optimization.h`) is rewritten**: `LinearProgram`,
  `QuadraticProgram`, `ProgramResult`, `Optimization::SolveLinearProgram`,
  `SolveQuadraticProgram`, `PortfolioProgram` and `FollowOptimizer`.
  Removed: `GradientDescent`, `MomentumGD`, `Adam`, `RMSprop`,
  `AnalyzeConvexity`, `ComputeHessian`, `SolveLP`, `SolveLPWithBounds`,
  `SolveQP`, `SolveUnconstrainedQP`, the numerical derivative and integral
  functions, the test functions (Rosenbrock, Sphere, ...), `VectorNorm` and
  their result structs. Their replacements are in `cyxwiz/calculus.h` (`Calculus::
  Integrate`, `Differentiate`, `ErrorByStep`, `CurvatureAt`,
  `CurvatureOverRange`) and `cyxwiz/expression.h` (`Expression`).

- **Tools > Text opens Data Studio's Explore > Text** (Tokenization: Tokens,
  Word Frequency: Words, TF-IDF Analysis: Terms, Word Embeddings: Embeddings,
  Sentiment Analysis: Sentiment; Data > Tokenization: Tokens). The five
  separate text windows are gone; Language Model Generation stays a window.
- **The Sentiment Analysis node is VADER** (vaderSentiment 3.3.2, checked
  rule for rule): it adds `<text>_compound`, `<text>_positive`,
  `<text>_neutral`, `<text>_negative` and `<text>_sentiment` ("positive" from
  0.05, "negative" at -0.05 and below, else "neutral") beside the input
  columns and keeps the label and the split. The Simple and AFINN methods are
  gone: a saved graph's `method` and `label_col` settings are ignored, and a
  graph that read the old output columns needs the new names.
- **The TF-IDF and Count Vectorizer nodes split words by one rule** shared
  with Explore > Text: letters and digits of any script, lower-cased, two or
  more long, an apostrophe inside a word kept ("don't"); the English stop
  words are scikit-learn's `ENGLISH_STOP_WORDS` and their contractions. The
  feature columns can differ from before: refit a saved vectorizer state
  rather than reusing it.
- **Text Clean keeps the letters, digits and marks of every script** when it
  removes special characters (it kept ASCII only).
- **The t-SNE node's `learning_rate` defaults to 0 = auto** (scikit-learn's
  rows / 48, at least 50); a typed value still wins. Small data now
  separates in 3D as it does in scikit-learn.
- The Template nodes Word Frequency and the second Tokenizer are no longer in
  the node list (Explore > Text > Words and the Tokenizer node do that work).
- **Plugin SDK (`cyxwiz/text_processing.h`)**: `Stem` is the Porter (1980)
  algorithm (NLTK's ORIGINAL_ALGORITHM); `GetStopwords` returns
  scikit-learn's list and the contractions; `ComputeTFIDF` / `ComputeTF` use
  raw counts on `Words`. Removed: `SentimentResult` and the lexicon sentiment
  functions (use `Vader`), the sample text generators, and
  `CreateOneHotEmbeddings` / `CreateRandomEmbeddings`. Added: `Words`,
  `NGrams`, `Terms`, `Vader` and `VaderScores`.

- **Training Dashboard points are the mean since the previous point**, not
  the epoch's running mean, so the curves no longer jump at each epoch start;
  the training cards' "best" names a whole epoch, never one batch.
- **Data Studio ML readiness follows the flow back from each Data Split
  input**: through a Quality Analyzer to its Data Input, or to a step's run
  result (SQL, Text Clean, ...), and says which one has not been loaded or run.
- **Canvas groups follow the zoom**, and a group's name no longer sits on its
  first node's title.
- **Plugin SDK: `cyxwiz::image::RandAugmentOp` is now `AugmentOp`** in
  `cyxwiz/image_augmentation.h`, with a new `Invert` value: AutoAugment,
  TrivialAugment and AugMix pick from the same list as RandAugment. A plugin
  that names `RandAugmentOp` needs the new name; `kRandAugmentOps` (14)
  still counts RandAugment's own ops, `kAugmentOps` (15) the whole list.
- **Image Rotate fills the corners outside the image with 0 through
  torchvision's fill mask** (RandomRotation's default), which changes
  bilinear borders slightly.
- **Recurrent layers (RNN, LSTM, GRU) now reset their state at the start of
  every forward call.** Each batch starts from a zero hidden (and cell)
  state, matching PyTorch and TensorFlow defaults and making CPU and GPU
  training numerically identical. Previously the CPU path carried the
  final state of the previous batch into the next one; training numerics
  may shift slightly if a workflow relied on that. Streaming inference or
  deliberate truncated BPTT can still seed a call explicitly through the
  layer's hidden/cell state setters, which apply to the next forward call
  only.
- Bidirectional LSTM (and RNN) training no longer fail closed: both run as
  split forward/reverse branches, like GRU, on the CPU reference or the
  native neural provider.

### Added

- **CyxWiz Cream theme.** A warm light theme beside CyxWiz Light: a cream
  window, ivory fields and menus, warm brown text, the CyxWiz purple accent
  (View > Theme or Preferences > Appearance; the Theme Editor changes any
  of it).
- **Squeeze-Excitation node.** Scales each channel of a feature map by a
  learned gate (mean over the map, Dense - ReLU - Dense - Sigmoid), as
  torchvision's SqueezeExcitation. Settings: the hidden width (channels /
  reduction or an exact number), the inner activation (ReLU, Hardswish,
  Swish), the gate (Sigmoid, Hardsigmoid) and bias. Properties lists the
  six steps inside the node with their outputs and weights ("Inside this
  layer").
- **Nodes menu commands work from the menu bar.** Group Selected, Duplicate,
  Delete Selected and Freeze in the Nodes menu were always greyed when the
  menu was opened, because the open menu took the focus from the canvas; the
  canvas now keeps it while a menu is open.
- **Frozen layers.** Every layer node with weights (Dense, the convolutions,
  the normalisations, Squeeze-Excitation, recurrent and attention layers) has
  a **Frozen** setting: the layer keeps its weights while the rest trains.
  BatchNorm adds **When frozen**: keep its running statistics (default) or
  keep updating them. Both are checked against PyTorch with `requires_grad`
  off. The As compiled card shows "Weights: N frozen".
- **Freezing from the canvas.** A frozen node is dimmed with a snowflake
  mark, and the line under the canvas shows "Trainable N · Frozen M".
  **Nodes > Freeze** (also in the canvas right-click menu under Actions)
  freezes or unfreezes the selection, every layer loaded from the weights
  file, all but the head, or all but the last N layers. **Preferences >
  Training** sets what a new BatchNorm does with its statistics when frozen.
- **MobileNetV2 and MobileNetV3 checked against PyTorch.** Built from canvas
  nodes and started from torchvision's published ImageNet files, both give
  torchvision's outputs, as ResNet-18 does. Squeeze-Excitation takes
  torchvision's `fc1` / `fc2` weights by name, and weight files in PyTorch's
  format from before 1.6 (torchvision's MobileNetV2 file) are read.
- **ResNet-18 fine-tuning example.** `cats_dogs_resnet18_finetune.cyxgraph`
  is torchvision's ResNet-18 node for node, started from the published
  ImageNet weights with the backbone frozen (86.29 % test accuracy on cats vs
  dogs after three epochs; 92.60 % after three more with every layer
  training). Checked against PyTorch: with those weights the
  graph's outputs equal torchvision's.
- **Start training from a published weights file.** Data Loader >
  Checkpoints > **Start from** is now one choice: random weights, a
  checkpoint of an earlier run, or **a weights file** (a PyTorch state-dict
  `.pth` / `.pt`, or `.safetensors`). Layers are matched by node name and
  PyTorch's layouts are converted; a layer the file has no fitting weights
  for starts fresh. The dialog shows what the file gives the graph before
  training, with a match report per layer. PyTorch is not needed and the
  file is never run. See docs/usage/pretrained_weights.md.
- **ReLU6 activation node.** `min(max(x, 0), 6)` (torch.nn.ReLU6), the
  activation of MobileNetV1 and MobileNetV2; also `pycyxwiz.relu6`, `ReLU6`
  and `ReLU6Module`.
- **Bias switch on the convolution nodes.** Conv2D, Depthwise Conv2D, Conv1D,
  Conv3D and ConvTranspose2D have a **Bias** setting (on by default). Off is
  what a convolution followed by BatchNorm uses in ResNet and MobileNet.
- **Padding on MaxPool2D and AvgPool2D.** A **Padding** setting (0 by
  default, at most half the pool size), as `torch.nn.MaxPool2d(padding=...)`.
- **Hardsigmoid activation node.** `relu6(x + 3) / 6` (torch.nn.Hardsigmoid),
  a sigmoid made of straight pieces; also `pycyxwiz.hardsigmoid`,
  `Hardsigmoid` and `HardsigmoidModule`.
- **Hardswish activation node.** `x * relu6(x + 3) / 6` (torch.nn.Hardswish),
  the activation of MobileNetV3; also `pycyxwiz.hardswish` and
  `HardswishModule`.
- **One-cycle learning-rate schedule.** `lr_schedule = one_cycle` on any
  optimizer node: torch's OneCycleLR over the run's optimizer updates, with
  `learning_rate` as the peak and `warmup_ratio` as the rising share.
- **SGD: Nesterov momentum and weight decay.** The SGD node gains
  `nesterov`, `weight_decay` and `weight_decay_exclude` (none /
  norms_and_biases), with torch.optim.SGD's arithmetic. Before, a
  `weight_decay` on SGD was ignored with a warning.
- **Image Gabor node.** A bank of striped kernels over every image, done
  once when the images are prepared: one channel per orientation and
  wavelength (4 x 2 by default), as the energy of both phases or one phase.
  With Keep image the model reads the photo and the bank (`[H, W, 3 + 8]`);
  the channels are named and shown as a sheet. Example:
  `examples/cyxgraph/cats_dogs_gabor_bank.cyxgraph`.
- **Images with more than three channels.** Image Edges can keep the image
  and add the edge map as a 4th channel (`keep_image`); the model then
  reads RGB + edges. Steps after it run on each channel; colour steps and
  ImageNet statistics say why they need RGB; Normalize's per-channel rows
  are named (R, G, B, edges). Such an image is shown as a sheet: the colour
  picture and one named grey tile per further channel, in Data Studio >
  Prepare (thumbnail with `+1`, hover for the sheet), Explore > Images and
  the Studio Debugger's Inspector, Top images, Embedding and Where it looks.
- **Image Edges: one direction, the angle, LoG and DoG.** Sobel and Scharr
  take `direction` (magnitude, x, y or angle); two new methods are `log`
  (Laplacian of Gaussian, with `sigma`) and `dog` (Difference of Gaussians,
  `sigma` and `sigma2`). Image sets prepared before stay valid.
- **Image Filter node.** One fixed kernel over every image, done once when
  the images are prepared: a classic one (identity, box, sharpen, emboss,
  outline, Prewitt x / y, Roberts x / y, Kirsch and Robinson in eight
  directions) or the numbers you type (`0 -1 0; -1 5 -1; 0 -1 0`), with
  Normalize, a border choice and three ways to store answers outside 0 to
  255 (clip, absolute, around mid-grey). On the canvas and in Data Studio >
  Prepare. Example: `examples/cyxgraph/cats_dogs_image_filter.cyxgraph`.
- **Neural Architecture Search trains its best versions.** Tools > Model >
  Neural Architecture Search reads the input and output sizes from the
  graph (no typed Input Configuration). It shows the quick score as a card
  with its parts. Each change (add, remove, swap a layer, units, activation)
  is a step with its score; Apply puts the version on the canvas (Ctrl+Z
  undoes) and "Back to the graph's version" drops the steps. The search
  keeps the 10 best different versions; **Train the best N** trains them
  the way Train does (data prepared once, the model in use untouched) and
  gives each one's validation loss and accuracy, with Apply per row. Search
  defaults are in Preferences > Search. Guide:
  `docs/usage/neural_architecture_search.md`.
- **Hyperparameter Search trains the graph.** Train > Hyperparameter
  Search picks what to vary from the compiled graph (layer sizes, the
  optimizer's rates, the DataLoader's batch size and epochs; not the output
  layer) with a starting range around each value, and searches by Grid,
  Random or Bayesian. Each trial trains the way Train does, on data
  prepared once; the Trials view shows progress (Stop keeps the trials
  done, and the line says whether a search finished, was stopped, reached
  its time limit or had three trials in a row fail), the best trial (Apply best
  to the graph, Ctrl+Z undoes; Export CSV), the score per trial and every
  trial's values, scores and failures. Before, the panel took free-text
  parameter names and never trained. Preferences > Search sets how a new
  search starts (method, trials, time limit, score). Guide:
  `docs/usage/hyperparameter_search.md`.
- **Favourites and Recent in the Nodes panel.** Star a card (or right-click
  it) to keep the node under Favourites at the top of the panel; Recent
  lists the last nodes you added, newest first, as chips to click or drag.
  The filter gains *Favourites*. Both are saved with your Engine settings
  (every project); **Preferences > Nodes panel** turns Recent off, sets how
  many it shows and clears it. Guide: `docs/usage/node_palette.md`.
- **Layer capture for the Studio Debugger** (Preferences > Studio Debugger,
  off to start). Local Debug and training can keep every layer's input and
  output, weights and gradients. Training takes a snapshot every N epochs and
  after the last one, on the debugger's picked sample or a batch of 8, and
  training comes out exactly the same with it on or off. Captures are saved
  with their run, and only the last 10 runs keep theirs; Model > Layers shows
  them (below). Your defaults for when, what, which input and the
  feature-map colour (Heat or Grayscale) are kept in Preferences; Layer
  capture in the debugger's toolbar changes them for the session, shows one
  snapshot's size for the model on the canvas and saves them as your
  defaults. Guide: `docs/usage/studio_toolbar.md`, "Layer capture".
- **Studio Debugger Full size**: a toolbar button hides the side panels
  while the debugger is shown and brings them back afterwards.
- **Studio Debugger > Model > Layers**: every captured layer with the views
  its tensors support. Feature maps (strongest channels first, Heat or
  Grayscale), each unit's value, a histogram of the values, and weights and
  gradients as a heatmap whose cells give their index and value. The
  Inspector shows the sample as the model received it and the clicked map
  enlarged. Saved runs show their captures too, and a batch of 8 steps
  through its items. The view names the picked item, its label and the
  model's prediction ("predicted cat, 99.0 %", read the way the loss reads
  the output), groups the layers by the graph's canvas groups, gives each
  layer's kernel, stride and parameter count, orders feature maps by mean,
  peak or number (a Preferences default too) and slices conv weights by
  kernel position. The first convolution's **Filters** view draws each
  learned filter as a small RGB (or grey) picture with its weight norm.
  **Where it looks** shows which parts of the input the model relies on:
  each image patch, token or table feature is covered in turn and the
  model runs again on the GPU; the change in the picked class's score is
  drawn over the input. The prediction shown now comes from the evaluation
  pass (dropout off), the same one Where it looks scores. **Saliency** is
  the second method: the gradient of the picked class's logit with respect
  to the input, as its size or gradient × input, from one forward and one
  backward pass (for text, at each token's embedding); checked against
  PyTorch. The method and what saliency shows are Preferences defaults.
  **Training snapshots** can be browsed: Overview > Runs lists training runs
  that kept layer snapshots beside Local Debug runs; Overview > Compare shows
  a training run's snapshots with each epoch's loss and validation accuracy;
  Model > Layers steps through them keeping the layer and channel, and
  **Compare with** draws the same view for two epochs side by side.
  **Top images** shows, for a layer's channels, the 9 images (of 320 spread
  over the data) that excite each most: the Dense head's strongest votes per
  class, the clicked channel, or this sample's strongest channels; ranked by
  a channel's maximum or mean over its map.
  **Contributions** shows how a Dense layer's output was computed for the
  sample: its largest w × x terms, the sum, bias and value, and where an
  input comes from.
  **Embedding** projects 320 items' values at a flat layer to 2-D (PCA,
  t-SNE or UMAP), coloured by true or predicted class with the mistakes
  ringed; click a point to inspect its item.
  **Unit weights** draws each unit of a Dense layer that reads the image
  flattened as the picture of what it looks for, the strongest units first;
  click one for it large with its bias, value and top images.
  Guide: `docs/usage/studio_toolbar.md`, "Model > Layers".
- **The Portfolio Weights node** (Analytics): the least risky mix of assets
  from a table of returns, with an optional target return and long only,
  fitted on the Train rows; one row per asset with its weight, mean, sd and
  share of the risk. Quadratic > From a returns table in the window reads
  the same problem and adds this node (example:
  `examples/cyxgraph/portfolio/`).
- **Python: `pycyxwiz.calculus`** (`Expression`, `integrate`, `derivative`,
  `step_errors`, `curvature`, `curvature_over_range`, `gauss_legendre`) and
  **`pycyxwiz.optimize`** (`linprog`, `qp`, `portfolio`, `follow`), checked
  against scipy and torch (example: `examples/python/optimization_calculus.py`).

- SDK: `cyxwiz::TimeSeriesStop` (time_series.h): while one lives, the time
  series fits of the thread (exponential smoothing, ARIMA, STL, the ADF
  test) ask its stop query between passes and end with "Cancelled.".
- **Data Studio Explore > Text**: one text column, grouped by a category or
  the label, in five looks: Words, Terms (TF-IDF: what sets each group apart,
  a row's terms and the rows most like it, the text quality), Sentiment
  (VADER per row and group), Tokens (the Tokenizer node's tokenizer: lengths,
  the rows a limit cuts, the vocabulary's coverage) and Embeddings (word2vec /
  fastText skip-gram or CBOW, GloVe, from a sample by default; nearest words).
  Each look adds its node to the graph. Guide in `docs/usage/text.md`.
- **The word map**: word vectors in 3D (t-SNE, UMAP or PCA, k-means clusters
  named by their words), the word looked up and its neighbours ringed and
  joined, labels clear of each other, in a window of its own; also the **Map**
  tab of the Word Embeddings node's dialog.
- **Python: `pycyxwiz.text`** with `words`, `terms`, `english_stop_words`,
  `stem` and `vader`, the same rules as the tab and the nodes (example:
  `examples/python/text_words_and_sentiment.py`).
- **Example: cats vs dogs ResNet from scratch**
  (`examples/cyxgraph/cats_dogs_resnet_scratch.cyxgraph`, guide in
  `docs/usage/image_classifier_resnet.md`): 92.6 % test accuracy on one
  GTX 1050 Ti.
- **Python: `BCEWithLogitsLoss(reduction, pos_weight, label_smoothing)`** and
  a `label_smoothing` property.
- **BCE with Logits has `label_smoothing`**: training targets become
  target x (1 - e) + e / 2, the two-class form of CrossEntropy's label
  smoothing; checked against PyTorch. Plugin SDK: `BCEWithLogitsLoss` takes it
  as a third constructor argument (`GetLabelSmoothing` / `SetLabelSmoothing`).
- `KNOWN_ISSUES.md` ships in every release package, next to `LICENSE`. It lists known problems
  in the GPU runtimes and what to do about them. The first entry is an ArrayFire oneAPI bug in
  indexed min / max along rows (arrayfire/arrayfire#3732), which CyxWiz already works around.
- Environment switch `CYXWIZ_DISABLE_NEURAL_PROVIDERS=1` forces every
  recurrent layer onto the portable path for the process, as an escape
  hatch when diagnosing a GPU provider.
- Native neural providers for recurrent training on NVIDIA (CUDA) and
  OpenCL GPUs, selected per run by the device you choose. Stacked and
  bidirectional RNN, LSTM and GRU train on the GPU; the placement audit
  shows which path each layer took.
- On OpenCL devices, recurrent layers with fewer than 16 hidden units
  stay on the portable path because the OpenCL provider measured slower
  than the CPU there; the placement audit says so
  (`opencl_provider_below_retention_floor`). NVIDIA CUDA has no such floor.
- The OpenCL provider is verified on NVIDIA and on Intel integrated
  graphics (UHD 630). When a machine has several OpenCL GPUs, the run's
  selected device decides which one serves it.

### Fixed

- The node icon pack you choose is kept: View > Node Icon Pack and the new
  Node icons row of Preferences > Appearance set it, the next launch starts
  with it, and Reset appearance goes back to Font Awesome.
- The node icon packs (View > Node Icon Pack: Tabler, Remix, Lucide, Iconoir,
  Phosphor) drew the right icons only by chance: most nodes showed "?",
  nothing, or a Font Awesome icon. Each pack now draws the icons it names,
  from a font of its own.
- Font Awesome icons that showed the wrong picture or nothing: the circle of
  nodes (an envelope before), circle pause (an exit arrow), share, compare,
  ranking, left-right arrows, the poll chart, the rockslide (a money
  transfer), the Cosine LR node (empty) and the Scatter item of a table
  column's Plot menu (a ventilator mask, now dots).
- The Calculator bound a leading minus tighter than `^`, so `-2^2` gave 4 and
  `exp(-x^2)` gave exp(x^2). It now reads formulas through the same parser
  as the Optimization and Calculus window: `-2^2` is -4.
- Training with NAdam no longer stops with "NAdam has invalid continuation
  state" after about a thousand steps: the product of its momentum terms
  reaches 0 in float32 there, which is a valid state (PyTorch carries on
  the same way).
- Language Model Generation: a Generate from IDs run started with Details
  open now finishes when Details is closed (the panel stayed "Generating");
  Regenerate, Generate again and Continue further are off while another run
  or a speed measurement runs, or training runs on the session's model, with
  the reason on hover; Regenerate keeps the reply when the prompt cannot be
  built, and Send keeps a chat message that could not start.
- Data Studio Signal > Spectrogram labels its frequencies beside the cells,
  not on top of them.
- The TF-IDF and Count Vectorizer nodes and Explore > Text's Terms count a
  large column in a fraction of the memory: each term is kept once, and a
  row holds its term ids and counts, not a table of strings (300,000 rows of
  words and word pairs, min_df 1: 1.8 GB instead of 6.3 GB, and 29 s instead
  of 102 s, with the same figures).
- Linear algebra on a table of many rows no longer runs the GPU out of
  memory: the SVD (and rank, condition, least squares) of a tall matrix
  comes from its small Gram matrix instead of the full SVD, whose U is rows
  x rows (160,000 rows: 200 GB). Tools > Linear Algebra's From a table and
  the new steps use it.
- The PCA node learns from the Train rows only (it learned from every row,
  validation and test rows included, and dropped the split column when it
  output pc_* alone): its means, deviations and components come from the
  Train rows, every row is projected, the split is carried, and it can save
  and reuse its fitted values like the scalers.
- Data Studio Series and Forecast: Stop ends the fit running in the
  background (it ran on to its end, up to minutes on a long series).
- Training Dashboard: the Active model card no longer shows the internal
  provenance key (`restored_best_checkpoint`, wrapped mid-word) under the
  State line that says it in words.
