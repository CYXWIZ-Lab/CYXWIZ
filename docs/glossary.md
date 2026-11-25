---
title: Glossary
description: The terms used in the CyxWiz Engine, its documentation and its ecosystem, from Accuracy to Workloads, each with a link to the page that explains it.
type: reference
area: general
audience: user
status: stable
since: 1.0.7
owner: general
last_reviewed: 2026-11-25
related:
  - faq.md
  - concepts/index.md
  - main-window/index.md
  - reference/error-messages.md
---

Terms are in alphabetical order. A term in **bold** inside a definition has
its own entry. UI names are written as the Engine shows them.

## A

**Accuracy**: the share of predictions that match the label, shown on the
Training Dashboard for classification. See
[Train a model and watch it](user-guide/training/train-and-watch.md).

**Activation**: a node that applies a function element by element after a
layer, such as ReLU, Sigmoid or Softmax.

**Adam, AdamW**: **optimizers** that adapt the step size per weight. AdamW
applies weight decay separately from the gradient. See
[Losses and optimizers](user-guide/training/losses-and-optimizers.md).

**AGPL**: the GNU Affero General Public License, version 3 or later, the
licence of the Engine and the Server Node. See the
[Licensing FAQ](legal/licensing-faq.md).

**Alpha**: the current release stage. Only the latest release gets fixes.
See [Support and versioning](about/support-and-versioning.md).

**Annotation Editor**: a panel for drawing segmentation and box labels on
images.

**Apache License 2.0**: the licence of the CyxWiz **Plugin SDK**, so
plugins may use any licence.

**API token**: a short-lived token the Engine gets after you **sign in**,
sent with each call to the CyxWiz server side.

**Apply**: the button in a node dialog that takes its settings; in the Data
Input dialog it loads the data. See
[The Data Input dialog](main-window/data-input-dialog.md).

**ArrayFire**: the open-source array library the Engine computes with, on
the CPU and on GPUs through CUDA, OpenCL and oneAPI. See
[Compute back ends](concepts/compute-back-ends.md).

**Arrow**: Apache Arrow, the columnar in-memory table format the Engine
uses for tabular data.

**Arrow IPC, Feather**: file formats that store Arrow tables as they are in
memory.

**Asset Browser**: the panel that lists the project's files.

**Audit**: the checks a Data Input runs on a loaded dataset. With errors,
**Apply** is refused. See [Data loading problems](troubleshooting/data-loading.md).

**Augmentation**: random changes to training images or audio (flips, crops,
noise) drawn per batch, so the model sees varied samples.

## B

**Back end**: the runtime that executes the computation: ArrayFire's CPU,
CUDA, OpenCL or oneAPI runtime. See
[Compute back ends](concepts/compute-back-ends.md).

**Background Tasks**: the panel listing long jobs (loads, pipeline runs,
debug runs) with their progress. Its menu name is **Tasks**.

**Band**: one of the four stages a training graph's data passes through:
data preparation, partitioning, iteration and the model. See
[The data pipeline](concepts/data-pipeline.md).

**Batch**: the group of samples the model sees in one training step.

**Batch size**: the number of samples in a batch, set on the **Data
Loader**.

**Batcher**: the part of the Engine that cuts a dataset into batches for
training, in memory or from disk.

**BatchNorm**: a layer that normalises its inputs with batch statistics in
training and with running statistics in validation and test.

**Best checkpoint**: the checkpoint with the best validation score so far.
See [Checkpoints, early stopping and resume](user-guide/training/checkpoints-and-early-stopping.md).

**Build fingerprint**: what a plugin records about the SDK and compiler it
was built with; the Engine compares it before loading the plugin.

**Bundle**: see **Python node bundle**.

## C

**Calibration**: how well predicted probabilities match observed
frequencies. See
[Threshold and calibration](user-guide/evaluation/threshold-and-calibration.md).

**Canvas**: the area of **CyxWiz Studio** where you place and connect nodes.
See [The canvas](user-guide/graph-editor/canvas.md).

**Central server**: the CyxWiz service that keeps the list of server nodes,
handles **reservations** and sign-in. It is run by CyxWiz; its source is not
public. See [The CyxWiz ecosystem](concepts/ecosystem.md).

**Checkpoint**: a saved copy of the model's weights and training state
during a run, used to resume or to keep the best model.

**Class weights**: weights that make rare classes count more in the loss.
See [Train on unbalanced classes](user-guide/training/class-imbalance.md).

**Cloud Browser**: the panel that lists your CyxCloud datasets and files.

**Code Gen**: the run mode that writes Python code for the graph with the
**Code** button (PyTorch, TensorFlow, Keras or PyCyxWiz).

**Code export**: turning a graph into Python code. See
[The Studio bar](main-window/toolbar.md).

**Column role**: what a Data Input column is for: a feature, the label, an
id, or left out of the features.

**Command Palette**: Ctrl+P; lists every menu command with its shortcut.

**Commercial licence**: the licence CyxWiz offers for products that do not
want the AGPL's obligations. See the [Licensing FAQ](legal/licensing-faq.md).

**Compile**: checking a graph before it runs (F7). It turns the graph into a
training configuration and lists findings. See
[How graphs run](concepts/how-graphs-run.md).

**Compile report**: the window that lists the compile findings, where each
layer runs, and the configuration. See
[Compile and run a graph](user-guide/graph-editor/compile-and-run.md).

**Compute route**: a device together with the back end that drives it, such
as "CUDA on GeForce RTX ..." or "CPU". Training uses only verified routes.

**Confusion matrix**: a table of predicted against true classes, shown in
the test results.

**Console**: the bottom panel that shows the Engine's log messages and runs
**Commands**. See [Console and logs](usage/console_logs.md).

**Conv2D**: a 2-D convolution layer for images.

**Count Vectorizer**: a node that turns text into word counts per row.

**CPU fallback**: running on the CPU route when no GPU route is verified or
selected. The Engine uses ArrayFire's CPU back end for it, not separate CPU
code.

**Cross-entropy**: the loss for classification: it compares predicted class
probabilities with the true class.

**Cross-validation**: training and scoring a model on several folds of the
data to see how its score varies. See
[Cross-validation and learning curves](user-guide/evaluation/cross-validation-and-learning-curves.md).

**CUDA**: NVIDIA's GPU computing platform; one of the GPU back ends.

**CW code**: a diagnostic code such as `CW-C-0201` at the start of a
message. See [Error messages](reference/error-messages.md).

**.cyxgraph**: the file format of a graph: its nodes, links, settings and
layout, as JSON.

**CyxCloud**: CyxWiz's storage service for datasets, used by remote training
and Data Inputs that name a cloud file. See
[Datasets that live in CyxCloud](server-and-cloud/cyxcloud/datasets.md).

**.cyxmodel**: the CyxWiz model package: weights, the graph when included,
and metadata. See [Model formats](concepts/model-formats.md).

**.cyxnode**: a **Python node bundle**: a folder with a manifest and Python
code that adds nodes. See [Python nodes](extending/python-nodes.md).

**cyxwiz-backend**: the shared library with tensors, layers, models, losses,
optimizers and algorithms. The Engine and the Server Node both use it.

**CyxWiz Studio**: the graph editor panel: the canvas and the **Studio bar**.
See [Graph editor](user-guide/graph-editor/index.md).

## D

**Dashboard node**: a node that lays out several plots and figures as one
view. See [Visualization nodes](reference/nodes/categories/visualization.md).

**Data budget**: how much memory the Engine may use for tables. A larger
table stays on disk. Set in **Preferences > General**. See
[Data bigger than memory](concepts/data-bigger-than-memory.md).

**Data Convert**: a node that converts a table between file formats.

**Data Input**: the node that loads a file or folder into the graph. See
[Bring data into Data Studio](user-guide/data-studio/bring-in-data.md).

**Data leakage**: information from the validation or test rows reaching
training, so scores look better than on new data.

**Data lifecycle**: when the Engine loads and releases data: on Apply,
project close, node delete, graph clear, undo and unload. See
[Data lifecycle](concepts/data-lifecycle.md).

**Data Loader**: the node that sets epochs, batch size, shuffling and
checkpoints, and feeds batches to the model. See
[Data Split and Data Loader dialogs](main-window/data-split-and-loader-dialogs.md).

**Data Output**: a node that writes a table to a file.

**Data Pipeline**: the run mode with **Execute Pipeline**, which runs the
data steps and shows each step's result.

**Data Split**: the node that assigns each row to Train, Validation or Test.

**Data Studio**: the panel for the open graph's data: explore, prepare,
validate and read results. See [Data Studio](user-guide/data-studio/index.md).

**Dataset**: the data a Data Input loaded, held under a name in the
**registry**.

**Dataset Input**: a node that reads a dataset Data Studio already holds,
without owning it.

**Dataset Manager**: the panel for your CyxCloud datasets. Its menu name is
**Cloud Manager**.

**Debug messages**: detailed log lines, recorded unless **Record Debug
messages** is off in **Preferences > General**.

**Dense**: a fully connected layer: every input feeds every output unit.

**Device**: a CPU or GPU the Engine can compute on. Listed in
**Preferences > Devices**.

**Device-resident**: data kept in the GPU's memory between steps, without
copies to the host.

**Disk-backed**: a table kept on disk as a Parquet file and read in pages,
because it is larger than the data budget allows in memory.

**Dropout**: a layer that zeroes a random share of values in training only.

## E

**Early stopping**: ending training when the validation score has not
improved for a set number of epochs.

**Embedding**: a layer that maps tokens or ids to learned vectors; also the
vectors a model produces for its inputs.

**engine_config.json**: the file the Engine saves Preferences in. See
[Preferences](main-window/preferences.md).

**engine_log.txt**: the Engine's log file, beside the Engine program. See
[Crash dumps and logs](troubleshooting/crash-dumps-and-logs.md).

**Epoch**: one pass over the Train rows.

**Error, Warning, Note**: the three levels of a compile finding. An error
blocks Train; a warning does not; a note informs.

**Execute Pipeline**: runs the data steps of a graph without training.

**Extension node**: a node that a plugin or a Python node bundle adds. See
[Your own nodes](usage/extension_nodes.md).

## F

**F1 score**: the harmonic mean of precision and recall.

**Feature**: an input column or value the model learns from.

**Feature importance**: how much each feature changes the model's score.
See [Feature importance](user-guide/evaluation/feature-importance.md).

**Fine-tuning**: training a model that starts from pretrained weights.

**Finding**: one entry of the compile report.

**float32, float64**: 32-bit and 64-bit floating-point numbers. Training
uses float32; some analysis uses float64 where the device has it. See
[Numerical accuracy](concepts/numerical-accuracy.md).

**Fold**: one part of the data in cross-validation.

**Frame**: a labelled box on the canvas that holds nodes. See
[Groups, frames and names](user-guide/graph-editor/groups-frames-and-names.md).

**Freeze layers**: keeping some layers' weights fixed during training. See
[Freeze layers](user-guide/training/freeze-layers.md).

## G

**Gradient**: how much the loss changes with each weight; the optimizer
uses it to update the weights.

**Gradient clipping**: limiting the size of gradients. See
[Gradient clipping](usage/gradient_clipping.md).

**Graph**: the nodes and links that describe the data steps, the model and
how it trains.

**Group**: nodes collapsed into one node on the canvas.

**GPU pack**: an optional download that adds one GPU back end (CUDA, OpenCL
or oneAPI) beside the Engine. See
[Add a GPU pack and verify it](install/gpu-packs.md).

**gRPC**: the remote-call framework the Engine, the Server Node and the
CyxWiz services talk over.

**GRU, LSTM**: recurrent layers for sequences.

**Guided steps**: building a recipe step by filling a form instead of
writing SQL. See [Guided steps](user-guide/data-studio/guided-steps.md).

## H

**HDF5**: a file format for large numeric arrays and tables.

**Hyperparameter**: a setting chosen before training, such as the learning
rate or the batch size.

**Hyperparameter search**: training many variants to find good
hyperparameters. See
[Hyperparameter search](user-guide/training/hyperparameter-search.md).

## I

**Image recipe**: the recipe steps for image datasets. See
[Prepare images with the image recipe](user-guide/data-studio/image-recipe.md).

**Info panel**: the panel that describes the selected node. Its menu name
is **Node Info**.

**Installer**: **CyxWiz Installer**, which installs, updates and removes the
Engine and its GPU packs.

## K

**k-fold**: cross-validation with k folds; each fold is the test part once.

**Known issues**: confirmed problems not fixed yet. See
[Known issues](known-issues.md).

## L

**Label**: the value the model learns to predict.

**Label smoothing**: a cross-entropy option that softens one-hot targets.

**Layer**: a node that transforms tensors and usually has weights, such as
Dense or Conv2D.

**Layer capture**: keeping what every layer held during Local Debug and
training. See [Layer capture](user-guide/training/layer-capture.md).

**Learning curve**: a model's score against the amount of training data.

**Learning rate**: the size of each optimizer step.

**Learning-rate scheduler**: a node that changes the learning rate during
training. See [Learning-rate schedulers](usage/lr_schedulers.md).

**Link**: a connection from an output pin to an input pin.

**Local Debug**: F6; runs one sample forward and backward and opens the
Studio Debugger with what each layer held. See
[The Studio Debugger](user-guide/training/studio-debugger.md).

**Local Training**: the run mode for training on this computer.

**Loaded data**: the datasets the Engine holds in memory. See
[Loaded data](concepts/loaded-data.md).

**Loss**: the number training lowers: how far predictions are from the
labels. Also the node that computes it.

## M

**Manifest**: a plugin's `plugin.json` or a bundle's description file: id,
version, API version, library and permissions.

**Memory guard**: the check that stops a data step when the Engine grows
past a safe limit while it runs.

**Metric**: a score of the model, such as accuracy, F1 or mean squared
error.

**Metric learning**: training a model so similar samples get close
embeddings. See [Metric learning](user-guide/training/metric-learning.md).

**MFCC, Mel Spectrogram, Spectrogram**: nodes that turn audio into features
a model can read.

**Model seed**: the seed for weight initialisation and dropout, set on the
Data Loader. See [Seeds and reproducibility](concepts/seeds-and-reproducibility.md).

**Model window**: **Tools > Model**, a window with a trained model's
Summary, Learning rate, Explain, Health and Projector tabs. See [The Model window](main-window/model-window.md).

**MSE**: mean squared error, the usual loss for regression.

**MultiHeadAttention**: an attention layer with several heads, used in
transformers.

**MuJoCo**: the physics engine of the simulation plugin, used for
reinforcement learning environments. See
[MuJoCo viewport](usage/mujoco_viewport.md).

## N

**Neural architecture search**: searching over model shapes automatically.
See [Neural architecture search](user-guide/training/neural-architecture-search.md).

**Node**: one box on the canvas: a data step, a layer, a loss, an
optimizer, a plot.

**Node Browser**: the **Nodes** panel, which lists every node by category.
See [Nodes panel](main-window/nodes-panel.md).

**Node contract**: what a node promises: its pins, settings, shapes and
behaviour in each phase. See [The node contract](reference/graph-spec/node-contract.md).

**Node dialog**: the window a node opens on double-click, for nodes that
need a preview or many settings. See [Node dialogs](main-window/node-dialogs.md).

**Normalize**: a step that rescales values to a common range.

## O

**oneAPI**: Intel's computing platform; one of the GPU back ends, for Intel
GPUs.

**ONNX**: an open model file format. The Engine exports layer stacks to it
in builds with ONNX support.

**OpenCL**: an open GPU computing standard; one of the GPU back ends, for
NVIDIA and Intel GPUs.

**Optimizer**: the node that updates the weights from the gradients: SGD,
Adam, AdamW, RMSprop, Adagrad or NAdam.

**Output node**: the node that marks the model's output.

**Overfitting**: a model that fits the Train rows much better than new
data; the validation loss rises while the training loss falls.

## P

**P2P Training**: training on a server node over a direct connection from
the Engine; also the panel that shows its progress.

**Parity**: agreement of the Engine's results with a reference
implementation. See [Numerical accuracy](concepts/numerical-accuracy.md).

**Parquet**: a columnar file format. The Engine keeps disk-backed tables as
Parquet files.

**Pattern**: a ready-made graph or part of a graph that you insert from the
**Pattern Browser**. See [Patterns](usage/patterns.md).

**Permission**: something a plugin asks to do, such as reading files or
making network connections, which you allow or deny.

**Phase**: training, validation or test. Some nodes act differently per
phase (Dropout, BatchNorm, augmentation).

**Pin**: an input or output point on a node. A link joins two pins of
matching types. See [Connect nodes](user-guide/graph-editor/connect-nodes.md).

**Plot node**: a node that draws a chart from its input. See
[The Plot node](user-guide/evaluation/plot-node.md).

**Plugin**: a library that adds nodes, data formats, panels or analyses to
the Engine. See [Your own nodes](usage/extension_nodes.md).

**Plugin Manager**: the panel to install, approve and remove plugins and
bundles (**Apps > Plugin Manager**).

**Plugin SDK**: the headers, libraries and CMake files to build plugins.
See [SDK contents](sdk/sdk-contents.md).

**PPO**: Proximal Policy Optimization, the reinforcement learning algorithm
the Engine trains policies with. See [RL training](usage/rl_training.md).

**Preferences**: the Engine's settings dialog (Ctrl+,). See
[Preferences](main-window/preferences.md).

**Preparation Recipe**: the ordered data steps Data Studio applies to a
dataset before training. See [Prepare data with a recipe](user-guide/data-studio/prepare.md).

**Pretrained weights**: weights from an earlier training, for example a
PyTorch or Safetensors file. See
[Start from pretrained weights](user-guide/training/start-from-pretrained-weights.md).

**Project**: a folder with your graphs, scripts, data references and a
Python environment, opened with **File > Open Project**.

**Properties**: the panel with the selected node's settings. See
[Properties](main-window/properties.md).

**Protocol**: the gRPC messages and services between the Engine, the Server
Node and the CyxWiz services (`cyxwiz-protocol`). See
[The protocol](internals/protocol.md).

**Provider**: CyxWiz's own GPU code for layers ArrayFire cannot run well on
a device, such as `cyxwiz.nvidia-cublas-cell`. See
[What runs where on a GPU](concepts/gpu-providers.md).

**pycyxwiz**: the Python module that exposes the backend to scripts and
Python nodes. See [Python API reference](extending/python-api-reference.md).

**Python node bundle**: a `.cyxnode` folder of Python nodes; its code runs
only after you approve it.

**PyTorch oracle**: PyTorch used as the reference that the Engine's tests
compare results with. See [Tests](internals/testing.md).

## Q

**Quality Analyzer**: a tool that finds blurry, dark and duplicate images.
See [Quality Analyzer](user-guide/data-studio/quality-analyzer.md).

**Query Console**: a panel for SQL queries on loaded tables.

## R

**Recall, precision**: of the rows of a class, the share found (recall); of
the rows predicted as a class, the share right (precision).

**Record Debug messages**: the switch in **Preferences > General > Logs**.

**Registry**: where the Engine holds loaded datasets by name. Training,
Data Studio and the views all read from it.

**Regularization**: methods that keep weights small or sparse to reduce
overfitting. See [Regularization](usage/regularization.md).

**Reservation**: a block of time on a server node held for you. See
[Server Connection](server-and-cloud/server-address.md).

**Resize**: a node that gives every image the same size, needed for
batches.

**Results**: Data Studio's job with test results, cross-validation and
explanations. See [Read test results](user-guide/evaluation/test-results.md).

**Resume**: continuing a training run from a checkpoint.

**Row Filter**: a node that keeps rows matching a condition. See
[Row Filter dialog](main-window/row-filter-dialog.md).

**Run mode**: Code Gen, Data Pipeline or Local Training, picked on the
Studio bar.

**Run Test**: F8; tests the trained model on the Test split.

## S

**Safetensors**: a weight file format; the Engine exports and imports it.

**Sample Rows, Select Columns**: recipe steps that keep fewer rows or
columns.

**Script Editor**: the panel for Python scripts and notebooks.

**SDK**: see **Plugin SDK**.

**Seed**: a number that fixes random draws, so a run repeats. See
[Seeds and reproducibility](concepts/seeds-and-reproducibility.md).

**Server Connection**: the panel where you connect, sign in, reserve a node
and train on it.

**Server Node**: the CyxWiz program that runs training jobs for an Engine on
another computer. See [The CyxWiz ecosystem](concepts/ecosystem.md).

**SGD**: stochastic gradient descent, the basic optimizer.

**Sign in**: logging in with your CyxWiz account, from the account chip.

**Signature**: a plugin's cryptographic signature; a library that does not
match it is refused.

**Smoke Run**: a short run on real data that the Studio Debugger makes to
check a text graph end to end.

**Softmax**: an activation that turns scores into class probabilities.

**Sparse features**: features stored as non-zero entries only, from a Count
or TF-IDF vectorizer.

**SQL Workbench**: a dialog for building recipe steps with SQL. See
[SQL Workbench](user-guide/data-studio/sql-workbench.md).

**Standard Scaler**: a step that centres a column on 0 with a standard
deviation of 1.

**Start page**: the **Get started** page the Engine opens with. See
[The start page](main-window/start-page.md).

**Status bar**: the bar at the bottom of the main window. See
[Status bar](main-window/status-bar.md).

**Stratified split**: a split that keeps each class's share in every part.

**Studio bar**: the toolbar above the canvas with Compile, Train, the run
mode and Code. See [The Studio bar](main-window/toolbar.md).

**Studio Debugger**: the panel that shows Local Debug runs, layer views and
runtime traces. See [The Studio Debugger](user-guide/training/studio-debugger.md).

## T

**Table Viewer**: a panel that shows any table, with sort, filter and plot.
See [Table Viewer](main-window/table-viewer.md).

**Task starter graph**: a ready graph with a small dataset on the start
page.

**Tensor**: an n-dimensional array of numbers, what flows between layers.

**Test split**: rows kept out of training and validation, for the final
score.

**Text Tokenizer**: a node that splits text into tokens and numbers them.
See [Text node dialogs](main-window/text-node-dialogs.md).

**TF-IDF Vectorizer**: a node that turns text into word weights per row.

**Threshold**: the probability above which a binary model predicts the
positive class.

**Time Series Split**: a split that keeps rows in time order.

**Tool sidebar**: the strip of panel icons at the window's edge.

**Train**: F5; compiles the graph, prepares the data and trains.

**Train split**: the rows the model learns from.

**Training Dashboard**: the panel with the running training's curves and
controls. See [Training Dashboard](main-window/training-dashboard.md).

**Transformer**: a model built from attention and feed-forward blocks.

## U

**Unload from memory**: the Data Input action that releases its dataset.

## V

**Validation split**: rows scored during training to watch for overfitting
and pick the best checkpoint.

**Variable Explorer**: the panel that lists Python variables.

**Verify**: the check that runs a route's operations on your hardware and
records its speed. Training uses only verified routes. See
[Add a GPU pack and verify it](install/gpu-packs.md).

**Viewport**: the 3-D panel for simulations. See
[MuJoCo viewport](usage/mujoco_viewport.md).

**Vocabulary**: the list of tokens a text model knows, built by the Text
Vocabulary node.

## W

**Warm start**: starting a training from an earlier checkpoint's weights.

**Weight**: a learned number in a layer.

**Weight decay**: a penalty that shrinks weights at each step.

**Where each layer runs**: the compile report tab that says whether each
layer runs on the GPU or the CPU.

**Where the model looks**: the Studio Debugger's Where it looks, Top images
and Embedding views of what a trained model relies on. See [Where the model looks](user-guide/training/where-the-model-looks.md).

**Workloads**: the installer tab with install profiles such as **CPU only**
and **Recommended**.
