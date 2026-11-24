---
title: See where the model looks
description: Use the Studio Debugger's Where it looks, Top images and Embedding views to see what a trained model relies on and what each layer responds to.
type: how-to
area: training
audience: user
status: experimental
since: 1.0.8
owner: training
last_reviewed: 2026-11-25
related:
  - inspect-layers.md
  - layer-capture.md
  - studio-debugger.md
---

Three views of **Model > Layers** in the Studio Debugger run the captured
model again on your data: **Where it looks** (which parts of the input the
model relies on), **Top images** (the items that excite a channel most) and
**Embedding** (where the items land in a layer's features). Results are not
saved.

## Before you start

- A run with a layer capture that kept **Parameters**. See
  [Capture what every layer holds](layer-capture.md).
- For a trained model, a capture from a training run (**Layer capture:
  Training, every N epochs**). A Local Debug run's model has its starting
  weights, so its views show what an untrained model does.
- The graph of the run open on the canvas and compiling cleanly. These
  views rebuild the model from the capture's weights and the canvas's
  graph. With no graph open, or while it compiles or has errors, the view
  says which and continues by itself once the graph compiles. A saved run
  of a graph you changed since says so; run Local Debug again.

## Find where the model looks

1. In **Model > Layers**, pick the layer the model's input enters and open
   **Where it looks**.
2. Pick the **Method**: **Occlusion** (the default) or **Saliency**.
3. Pick the output in **Explain**: the predicted class, the true class, or
   any other class from the list.
4. Select **Compute** (occlusion). **Stop** ends it. Saliency is instant.
5. Read the map. The headline names the piece that moves the score most,
   with the score before and after (occlusion) or its value (saliency);
   hover the map or a chip for its value.

**Occlusion** covers each piece of the input in turn and runs the model
again; the view shows how the picked output's score moved. Blue means
covering that piece lowers the score, so the model relies on it; red means
it raises it. The forward passes run on the device, 64 at a time, off the
UI thread.

| Input | A piece is |
| --- | --- |
| Image | A 12 × 12 px patch every 4 px (196 positions on 64 × 64), filled with the image's mean per channel; the change is drawn over the picture. |
| Tokens (a text graph) | Each token replaced by the pad id, padding left out; one chip per token, by its id in the vocabulary. |
| Table features | Each feature replaced by its mean over up to 512 of the graph's prepared rows; a bar per feature. |

**Saliency** is the gradient of the picked score with respect to the input:
how much the score moves for a tiny change of each pixel, token or feature.
It takes one forward and one backward pass. **Of** says what is
differentiated: the class's logit (the score before the softmax, which does
not saturate near 100 %); for one sigmoid output, the logit, or minus it
for the other class. **Show** picks what is drawn:

- **Gradient size** (the default): where a small change moves the score
  most, in the Heat scale. An image: per pixel, the largest of its
  channels' gradients. Tokens: the length of the gradient at each token's
  embedding (token ids have no gradient, so it is taken at the Embedding's
  output); padding is left out. Table features: the gradient's size.
- **Gradient × input**: signed, in occlusion's blue and red: does this
  element push the score up or pull it down, given its value? An image:
  summed over the channels. Tokens: the gradient · the embedding.
  Features: the gradient × the value.

Occlusion and saliency answer different questions (removing a whole piece
against a tiny change), so they often agree but need not. Saliency is
checked against PyTorch by the Engine's tests.

## Find the items a channel responds to

**Top images** (named **Top samples** when the model's input is not
pictures) is offered on every layer with channels or units.

1. Pick the layer and open **Top images**.
2. Pick **Channels**:
   - **The Dense head's strongest votes**, when the model's last layer is a
     Dense reading these channels (after a Global Avg Pool): for each class,
     the two channels with the largest weight towards it;
   - **The clicked channel**: the one last clicked in Feature maps or
     Units;
   - **The 16 strongest for this sample** (conv layers).
3. On a conv layer, pick **Ranked by**: a channel's **Maximum** over the map
   (it fires somewhere) or its **Mean** (it fires all over).
4. Select **Compute**. It takes 320 items spread over the data (every k-th,
   as validation prepares them, so a folder sorted by class gives both
   classes) and runs them through the run's model in batches on the device.
   A card shows the steps: **Load the data** (a reopened project's Data
   Inputs load first), **Prepare 320 images**, and **Run this snapshot's
   model** with a bar and the count. **Stop** ends it.

The result is one card per channel: how it was picked, a summary (*4 ×
class 0, 5 × class 1 · 15.9 to 11.7*) and the 9 strongest items, each with
its rank, label and value. Click a picture to see it large in the Inspector
with its file, label, rank and value, and **Make it the picked sample** (run
Local Debug to see it).

## See how a layer separates the classes

**Embedding** is offered on a flat output with more than one feature, such
as a Flatten, a Dense or a pooled vector.

1. Pick the layer and open **Embedding**.
2. Pick the **Method**: **PCA** (in a moment; it keeps the large distances,
   and the caption says how much of the variance the two components hold),
   **t-SNE** or **UMAP** (a few seconds; they keep neighbours together, and
   the gaps between their clusters mean little).
3. Pick **Colour by**: **True class** or **Predicted class**. With the
   predicted class, the items the model gets wrong are ringed in amber.
4. Select **Compute**. It takes 320 items spread over the data, runs them
   through the snapshot's model and projects the layer's output to 2-D; the
   steps show in a card as for Top images.

The white ring is the picked sample. Hover a point for its item, label and
prediction; click it to see it in the Inspector, with **Make it the picked
sample**. Separate clouds per class mean the layer already tells the classes
apart; mixed ones mean the layers after it still have work to do.

## Keep your choices

Each view has a **Save as my default** link beside its controls (Method and
Show; Channels and Ranked by; Method and Colour by). **Preferences > Studio
Debugger > Layer views** lists them all. A layer that does not offer the
saved Top images channels starts on the first choice it has.

## Related

- [Inspect the layers](inspect-layers.md)
- [Capture what every layer holds](layer-capture.md)
- [Check a graph with Local Debug](studio-debugger.md)
- [Training](index.md)
