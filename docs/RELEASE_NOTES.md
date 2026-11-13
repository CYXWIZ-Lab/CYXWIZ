# CyxWiz Studio release notes

Notable user-facing changes, newest first. Engineering detail lives in the
private Data Studio tickets (`tofix*/track*`).

## Unreleased

### Changed

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
- **Model Summary and NAS count parameters from the real shapes.**
  - Model Summary (and Architecture Diagram) start from the input shape the
    compile reads from the data. Before, any graph without a Dataset Input
    node was analysed as 28 x 28 x 1 images.
  - A Conv2D or pooling layer with padding `same` keeps its height and
    width; before, `same` was read as no padding, so every later shape and
    Dense size was wrong.
  - Neural Architecture Search sizes each layer by what reaches it. Before,
    every layer was counted as if it saw the input: 158,570 parameters for
    an MLP that has 109,386.
  - Guide: `docs/usage/model_summary.md`.
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
  - Patterns and suggestions added to the canvas get fresh node ids.
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
