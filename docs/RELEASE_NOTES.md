# CyxWiz Studio release notes

Notable user-facing changes, newest first. Engineering detail lives in the
private Data Studio tickets (`tofix*/track*`).

## Unreleased

### Changed

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
