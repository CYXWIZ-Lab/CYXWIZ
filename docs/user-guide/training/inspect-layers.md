---
title: Inspect the layers
description: Read a run's layer capture in the Studio Debugger's Model > Layers view, from feature maps and weights to a Dense layer's sum.
type: reference
area: training
audience: user
status: experimental
since: 1.0.8
owner: training
last_reviewed: 2026-11-25
related:
  - layer-capture.md
  - where-the-model-looks.md
  - studio-debugger.md
  - ../graph-editor/groups-frames-and-names.md
---

**Model > Layers**, the first view of the Studio Debugger's **Model**
section, shows the shown run's layer capture. A live Local Debug run shows
its own capture; a run picked in **Saved runs** reads its capture from the
run store. Without a capture the view says how to get one (see
[Capture what every layer holds](layer-capture.md)).

## The line above the list

It names the picked item and the model's answer, for example
*dog.11289.jpg · label class 1 · predicted class 0, 63.3 % · 36 layers
captured · image 0 of cat_dog, train*.

- **Item** steps through the batch when the capture is a batch of 8.
- The prediction follows the loss and the last layer: one logit through a
  sigmoid (BCE with logits), logits through a softmax (cross-entropy),
  probabilities from a Sigmoid or Softmax layer, or the value for
  regression. Class names come from the data (as typed on the Data Input);
  names that are only numbers read as *class 1*.
- **Snapshot** steps through a run's captures when it has more than one
  (training snapshots): *‹ epoch 10 ›*, or pick one from the list, with
  *n of m snapshots* and that epoch's validation. The picked layer, tab and
  channel stay as you step.
- **Compare with** picks another snapshot of the run and draws the shown
  view for both, side by side: feature maps with the same channels in the
  same order, filters in the same order, weights and gradients with the
  same rows and kernel slice (each on its own scale), values and units.
  Where it looks stays one snapshot at a time.

## The layer list

Every captured layer, its kind and its output for one sample (*64 × 64 ×
32*). The kind adds what the tensors show: the kernel (*Conv2D 3 × 3*), a
stride from the input and output sizes (*, stride 2*) and a Dense layer's
parameters (*Dense, 257 parameters*).

The list is in sections named by the graph's canvas groups, as they were
when the run started. To get headings, group the nodes and name the groups:

1. On the canvas, select the nodes and press **Ctrl+G** (or right-click the
   canvas > **Create Group**).
2. Double-click the group's name band to rename it (*Stem*, *Stage 1*,
   *Head*).
3. Run again. Layers in no group are listed without a heading.

Details: [Groups, frames and names on the canvas](../graph-editor/groups-frames-and-names.md).

## The views

The views follow the layer's tensors, not its kind, so every layer gets the
views its data supports.

| View | Shown for | What it shows |
| --- | --- | --- |
| **Feature maps** | An output of height × width × channels | 16 channels of the sample in the chosen **Order**: strongest first (largest mean), highest peak first (largest value) or channel order. **All N channels** shows every channel (up to 256). **Colour** switches Heat (black through red to yellow) and Grayscale; the scale runs from 0 to the 99.5th percentile, so one bright pixel does not darken the rest. The count of dead channels (zero everywhere) is under the maps. Click a map to enlarge it in the Inspector. |
| **Units** | A flat output | Each unit's value for the sample; hover a bar for its number. |
| **Values** | Every layer | A histogram of every output value (log counts), with min, max, mean, the share of zeros and any NaN or Inf. |
| **Weights**, **Gradients** | Layers with parameters | The main parameter as a heatmap, blue (negative) to red (positive). Conv weights [kh, kw, cin, cout] show one row per output channel over kernel × input channels; Dense weights [out, in] one row per output. **Slice** shows one kernel position instead: output channels by input channels at, say, (1, 2) of the 3 × 3 kernel. Hover a cell for its index and value; − and + page through the rows of a large matrix. The histogram and numbers are under it. |
| **Filters** | The first convolution on an image (1 or 3 input channels) | Each output channel's kernel as a small picture, RGB for a colour image, grey for one channel, each scaled to its own range so its colour balance shows; the number under it is its weight norm. |
| **Contributions** | A Dense layer | How the output was computed for the sample: output = the sum of w × x + b, from the captured input and weights. |
| **Unit weights** | A Dense layer that reads the image flattened | Each unit's weights drawn as the image they read. |
| **Where it looks**, **Top images**, **Embedding** | See [See where the model looks](where-the-model-looks.md) | They run the model again on the data. |

### Filters

A filter is the pattern its channel responds to: where the image looks like
it, that channel's feature map is bright. A Local Debug run uses a fresh
model, so its filters are the starting weights (noise); trained filters
(colour edges and blobs) show in a capture from a training run (**Layer
capture: Training, every N epochs**). Only the first layer sees the image's
colours, so deeper convolutions have no Filters view.

### Contributions

On the model's head, **Explain** picks the output (the predicted class
first, then the others); on a hidden Dense layer, **Unit** picks the unit
(this sample's strongest first, or any unit from the list). The 8 largest
terms come first, with w, x and w × x (blue pulls the output down, red
pushes it up). The line under them gives the sum, the bias and the value,
then the other logits and the prediction (a head) or the activation after
it (a hidden layer), and what the other inputs add up to. Click a row: the
Inspector says where that input comes from (for a Flatten of a conv stage,
the layer, row, column and channel).

### Unit weights

Shown for a Dense layer with only Flatten, Dropout or Reshape before it and
as many inputs as the image has pixels and colours. A grey image is drawn
red where a bright pixel raises the unit and blue where it lowers it; a
colour image as the colours that raise it (bright) or lower it (dark); each
unit on its own scale. **Order** picks the largest weight norm first or
unit order; the first 32 units show, and **All N units** shows every one.
Click a unit for the Inspector: the unit large, its norm, bias and value
for this sample, and **Top images of unit N**. A data table (a CSV of pixel
columns) is not an image to the model, so its first Dense layer has no Unit
weights.

## The Inspector

The Inspector shows the sample as the first layer received it (after the
data steps), its name and label, and the prediction with how it was read,
for example *logit -0.5472 → sigmoid 0.3665* (the raw output when the loss
gives no rule). The prediction comes from the evaluation pass, with dropout
off and BatchNorm on its running statistics; only when the model cannot be
rebuilt does it come from the run's own training step, and then it says so.
A clicked feature map shows enlarged with its mean, maximum and rank.

## Defaults for the views

The colour and order are part of the session's settings (**Layer capture**
in the toolbar); **Save as my default** keeps them. The other views'
choices (Where it looks: Method, Show; Top images: Channels, Ranked by;
Embedding: Method, Colour by; Unit weights: Order) each have a **Save as my
default** link beside their controls. A change in a view lasts for the
session until it is saved. **Preferences > Studio Debugger > Layer views**
lists them all.

Everything is computed on the compute device; only what a view draws (the
shown maps, a page of heatmap rows, a histogram) is copied to the host.

## Related

- [Capture what every layer holds](layer-capture.md)
- [See where the model looks](where-the-model-looks.md)
- [Check a graph with Local Debug](studio-debugger.md)
- [Training](index.md)
