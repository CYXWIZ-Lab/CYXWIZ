---
title: The Studio Debugger
description: Run Local Debug on one sample, capture what every layer holds, and read the layer views of the Studio Debugger.
type: reference
area: training
audience: user
status: stable
since: 1.0.7
owner: training
last_reviewed: 2026-11-24
related:
  - ../graph-editor/compile-and-run.md
  - ../../main-window/toolbar.md
redirect_from: []
---

The Studio Debugger runs your graph on one item of your data before you
spend time on full training (**Local Debug**, F6), and shows what every
layer held. Open it with **Train > Local Debug**, **Local Debug** on the
[Studio bar](../../main-window/toolbar.md), or **View > Panels > Studio
Debugger**.

## The debugger and the canvas

Under the debugger's status line: **Canvas** (back to the graph; the
debugger hides, F6 or View > Studio Debugger brings it back with its run),
**Compile report** (F7), and, once Local Debug passed on the graph as it is
now, **Train** (F5), with the result: "Local Debug: Passed at 16:52, graph
unchanged".

**Full size** (the expand button beside Saved runs) gives the debugger the
whole width. It hides the side panels (Asset Browser, Nodes, Patterns,
Properties, Info) while the debugger is shown. Press it again,
switch to another tab or close the debugger, and the panels it hid come
back.

## Which sample Local Debug runs

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

## Layer capture

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

## Training runs and their snapshots

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

## Model > Layers

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
  heading. Details: [Groups, frames and names on the canvas](../graph-editor/groups-frames-and-names.md).
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
    most, with the score before and after.

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
    to see it). Results are not saved.
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
