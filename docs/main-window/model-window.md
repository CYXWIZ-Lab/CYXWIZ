---
title: The Model window
description: The Tools > Model window and its five tabs, Summary, Learning rate, Explain, Health and Projector, with what each figure means.
type: reference
area: evaluation
audience: user
status: experimental
since: 1.0.8
owner: evaluation
last_reviewed: 2026-11-25
related:
  - ../user-guide/evaluation/test-results.md
  - ../user-guide/evaluation/feature-importance.md
  - training-dashboard.md
redirect_from:
  - model-window.md
---

**Tools > Model** opens one window about the graph's model, with five tabs.
Each item of the **Tools > Model** menu opens the window on its tab.

| Tab | What it answers | Menu items |
| --- | --- | --- |
| Summary | Each layer's output shape, parameters and multiply-adds, and the model's memory | Model Summary, FLOPs Calculator |
| Learning rate | Which learning rate to train with (a range test) | Learning Rate Finder |
| Explain | Which pixels made the model say what it said, for one test row | Grad-CAM Visualization, Saliency Maps |
| Health | Whether each layer is still learning: weight size, gradient size, dead units | Training Health |
| Projector | How a layer's features group the test rows (PCA, t-SNE, UMAP) | Embedding Projector |

How good the trained model is lives in Data Studio's **Results > Test**
(see [Read test results](../user-guide/evaluation/test-results.md)).

The example numbers on this page are from
`examples/cyxgraph/model_tools/mnist_cnn.cyxgraph`: a small CNN on the
10,000 MNIST test digits (8,000 train, 1,000 validation, 1,007 test),
trained for 3 epochs. The folder's README says how to get the data.

## Summary

The table lists every layer of the compiled graph: its name, type, output
shape, parameters (trainable and frozen) and multiply-adds, with the
formula on hover ("26·26·32 · 3·3·1 = 194,688"). Clicking a row selects
its node; **Show on canvas** goes to it.

| Figure | Meaning | Example |
| --- | --- | --- |
| **Parameters** | The count and their size in float32 | 225,034 (879.0 KB) |
| **Multiply-adds** | For one sample. Counted as PyTorch's `torch.utils.flop_counter` counts FLOPs, halved (one multiply-add = two FLOPs). Attention counts as its plain matrix products; recurrent layers as their weight matrices times the sequence length. | 2.63 M |
| **Activations** | The memory of every layer's output for a batch (**Batch for the activations**, the graph's batch size by default), kept for the backward pass | 16.73 MB at 64 |

A line under the table names the layer doing the most work and the one
holding the most weights. Running statistics (BatchNorm's running mean and
variance) are not parameters and are not counted. **Copy as text** and
**Export CSV...** save the table.

The counts are checked against PyTorch: five models (this CNN, a mixed
image model, a residual block, an LSTM and an attention block) are built
in torch and must match exactly.

## Learning rate

A learning-rate range test (Leslie Smith's). The window trains a copy of
the model on the Train rows, raising the learning rate on a log scale from
**From** to **To** over **Steps** steps, and records the loss after each.
It uses the graph's batch size and optimizer; the trained model in use,
its checkpoints and the Training Dashboard's runs are not changed. It stops
early once the smoothed loss (smoothing 0.05) passes 4 x its lowest: the
rate is then far too high.

The plot is the smoothed loss against the rate. The suggested rate comes
from one of two rules:

| Rule | Suggests |
| --- | --- |
| **Steepest fall** (default) | Where the loss falls fastest (`numpy.gradient` of the smoothed loss against log10 of the rate) |
| **Lowest / 10** | A tenth of the rate with the lowest loss |

The button **Use *rate* in *optimizer*** (for example **Use 0.0085 in Adam
0.001**) writes the rate into the graph's optimizer node; **Copy the
curve** copies the rates and losses. The example stops at 85 of 100 steps,
lowest loss 1.56 at 0.0955, and suggests 0.0085 by the steepest fall. The
range test is checked against PyTorch.

## Explain

Pick a test row on the left (**All** or **Wrong only**). A test keeps the
inputs of every row it got wrong and of its first 64 rows, up to 64 MB.
The window shows four pictures:

| Picture | What it shows |
| --- | --- |
| Input | The row as the model sees it after the graph's steps. |
| Grad-CAM | Where the last convolution found what raised the score: its channels weighted by their mean gradient, ReLU, resized to the input (bilinear, as `torch.nn.functional.interpolate` with `align_corners=False`). Layer stacks with a convolution only. |
| Saliency | \|d score / d pixel\|: the pixels whose change moves the score most. Noisy by nature. |
| Integrated Gradients | Each pixel's share of the score, summed along the path from the baseline (the value 0 before the graph's Normalize: a black image) in **Integrated Gradients steps** steps. The shares add up to the score's change from the baseline; the line under it says how close. |

**Explain the score of** picks the class: the predicted class (default),
the true class, or any class. Saliency and Integrated Gradients use a
colour scale at the 99th percentile of |value|, as Captum draws them. All
three are computed on the model's own device and checked against Captum.
**Save the images...** writes the four PNG files. In the example, row 1014
is a 6 that the model reads as 5 with 98.8 %.

Results > Test > Errors has **Explain the first in the Model window**, and
an **Explain** link under each mistake.

Explain runs the model, so it does not start while Train, Test, Feature
Importance or generation from this session's model is running, and those
wait while Explain runs. Whichever is refused says which one is running.
Train can start meanwhile: it builds a new model.

## Health

Recorded during every training, once an epoch, at no extra pass:

| Chart | What it shows |
| --- | --- |
| Weight size by epoch | \|\|W\|\| of each layer's weights |
| Mean gradient by epoch | The mean size of its gradients over the epoch's steps |
| Weights, last epoch | A histogram of the weights |
| Dead after its ReLU | Units that gave 0 for every validation row |

The worst layer is selected first, and **What to try** suggests fixes. In
the example, 47 of Dense 128's 128 units are dead (37 %; 53 -> 47 -> 47 by
epoch), so it suggests LeakyReLU in place of ReLU after Dense 128, a lower
learning rate, or a narrower layer (81 live units do the work). Gradients
above 1e3 (or not finite) are named exploding, below 1e-7 vanishing. The
figures are checked against the same sums in PyTorch. The Training
Dashboard shows the same warning during training, with **Open Model >
Health**.

## Projector

After a test, each weight layer's output for the test rows is kept (images
averaged per channel; up to 64 MB). Pick a **Layer**, a **Method** (PCA,
t-SNE, UMAP) and **Colour by** (True class, Predicted, Right / wrong). In
the example, Dense 128 under PCA: the first two components explain 19.6 %
and 17.7 %. **Send to Projection tab** opens the features in Data Studio's
Explore > Projection (see
[Dimensionality reduction](../user-guide/data-studio/dimensionality-reduction.md)).

## From Python

`pycyxwiz.model` runs the same code on a `pycyxwiz.Sequential`: `summary`
(the same multiply-add counter), `lr_range_test` and `LearningRateRange`,
`saliency`, `smooth_grad`, `integrated_gradients`, and the calibration fits
(`reliability`, `fit_temperature`, `fit_platt`, `fit_isotonic` and their
`apply` functions). Grad-CAM needs a convolution, which a Python
`Sequential` cannot hold. `examples/python/model_tools.py` runs all of it
on a small network; the reference is
[pycyxwiz](../../python_doc/pycyxwiz.md) (`pycyxwiz.model`).

## Related

- [Read test results](../user-guide/evaluation/test-results.md)
- [Measure feature importance](../user-guide/evaluation/feature-importance.md)
- [Training Dashboard](training-dashboard.md)
