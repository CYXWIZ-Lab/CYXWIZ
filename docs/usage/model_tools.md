# The Model window

**Tools > Model** opens one window about the graph's model, with five tabs:

| Tab | What it answers | Menu item |
| --- | --- | --- |
| Summary | Each layer's output shape, parameters and multiply-adds, and the model's memory | Model Summary, FLOPs Calculator |
| Learning rate | Which learning rate to train with (a range test) | Learning Rate Finder |
| Explain | Which pixels made the model say what it said, for one test row | Grad-CAM Visualization, Saliency Maps |
| Health | Whether each layer is still learning: weight size, gradient size, dead units | Training Health |
| Projector | How a layer's features group the test rows (PCA, t-SNE, UMAP) | Embedding Projector |

How good the trained model is lives in Data Studio's **Results > Test**:
its Errors, Threshold and Calibration sections (see
[Testing a trained model](testing.md)). Python has the same work in
`pycyxwiz.model` (see the end of this page).

The numbers below are from the example graph
`examples/cyxgraph/model_tools/mnist_cnn.cyxgraph`: a small CNN on the 10,000
MNIST test digits (8,000 train, 1,000 validation, 1,007 test), trained for
3 epochs. Its README says how to get the data.

## Summary

The table lists every layer of the compiled graph: its name, type, output
shape, parameters (trainable and frozen) and multiply-adds, with the formula
on hover ("26·26·32 · 3·3·1 = 194,688"). Clicking a row selects its node;
**Show on canvas** goes to it. Above the table:

- **Parameters**: 225,034 for the example, and their size (879.0 KB in
  float32).
- **Multiply-adds** for one sample: 2.63 M. They are counted as PyTorch's
  `torch.utils.flop_counter` counts FLOPs, halved (one multiply-add = two
  FLOPs). Attention is counted as its plain matrix products; recurrent
  layers by their weight matrices times the sequence length.
- **Activations**: the memory of every layer's output for a batch
  (**Batch for the activations**, the graph's 64 by default), kept for the
  backward pass: 16.73 MB. A line under the table says which layer does
  the most work and which holds the most weights.

Running statistics (BatchNorm's running mean and variance) are not
parameters and are not counted. **Copy as text** and **Export CSV...** save
the table.

The figures are checked against PyTorch: the same five models (this CNN, a
mixed image model, a residual block, an LSTM and an attention block) are
built in torch and the counts must match exactly (ctest
`model_summary_pytorch_parity`). The range test is checked the same way
(`learning_rate_range_pytorch_parity`).

## Learning rate

A learning-rate range test (Leslie Smith's): the window trains a copy of the
model on the Train rows, raising the learning rate on a log scale from
**From** to **To** over **Steps** steps, and records the loss after each.
The graph's batch size and optimizer are used; the trained model in use, its
checkpoints and the Dashboard's runs are not changed. It stops early once
the smoothed loss (0.05) passes 4 x its lowest: the rate is then far too
high.

The plot is the smoothed loss against the rate. Two ways to suggest a rate:

- **Steepest fall** (the default): where the loss falls fastest
  (`numpy.gradient` of the smoothed loss against log10 of the rate).
- **Lowest / 10**: a tenth of the rate with the lowest loss.

The example: 85 of 100 steps (stopped early), lowest loss 1.56 at 0.0955,
suggested 0.0085 by the steepest fall. **Use 0.0085 in Adam 0.001** writes
it into the graph's optimizer node; **Copy the curve** copies the rates and
losses.

## Explain

Pick a test row on the left (**All** or **Wrong only**; Test keeps the
inputs of every row it got wrong and of its first 64 rows, up to 64 MB) and
the window shows four pictures:

- **Input**: the row as the model sees it after the graph's steps.
- **Grad-CAM**: where the last convolution found what raised the score: its
  channels weighted by their mean gradient, ReLU, resized to the input
  (bilinear, as `torch.nn.functional.interpolate` with
  `align_corners=False`). Layer stacks with a convolution only.
- **Saliency**: |d score / d pixel|, the pixels whose change moves the
  score most. Noisy by nature.
- **Integrated Gradients**: each pixel's share of the score, summed along
  the path from the baseline (the value 0 before the graph's Normalize: a
  black image) in **Integrated Gradients steps** steps. The shares add up
  to the score's change from the baseline; the line under it says how
  close.

**Explain the score of**: the predicted class (default), the true class, or
any class. Saliency and Integrated Gradients are drawn with their colour
scale at the 99th percentile of |value|, as Captum draws them. All three are
computed on the model's own device and checked against Captum (ctest
`model_explain_pytorch_parity`). **Save the images...** writes the four
PNGs.

The example: row 1014 is a 6 that the model reads as 5 with 98.8 %.

Results > Test > Errors has **Explain the first in the Model window**, and
an **Explain** link under each mistake it shows.

Explain runs the model, so it takes turns with Train, Test, Feature
Importance and generation from this session's model: whichever starts
second says which one is running and waits.

## Health

Recorded during every training, once an epoch, at no extra pass:

- **Weight size by epoch**: ||W|| of each layer's weights.
- **Mean gradient by epoch**: the mean size of its gradients over the
  epoch's steps (summed on the device).
- **Weights, last epoch**: a histogram of the weights.
- **Dead after its ReLU**: units that gave 0 for every validation row.

The worst layer is selected first. The example: 47 of Dense 128's 128 units
are dead (37 %; 53 -> 47 -> 47 by epoch), so "What to try" suggests
LeakyReLU in place of ReLU after Dense 128, a lower learning rate, or a
narrower layer (81 live units do the work). Gradients above 1e3 (or not
finite) are named exploding, below 1e-7 vanishing. All of it is checked
against the same sums in PyTorch (ctest `training_health_parity`).
The Training Dashboard shows the same warning during training, with **Open
Model > Health**.

## Projector

After a test, each weight layer's output for the test rows (images averaged
per channel; up to 64 MB) is kept. Pick a **Layer**, a **Method** (PCA,
t-SNE, UMAP) and **Colour by** (True class, Predicted, Right / wrong). The
example's Dense 128 under PCA: the first two components explain 19.6 % and
17.7 %. **Send to Projection tab** opens the features in Data Studio's
Explore > Projection for more.

## From Python

`pycyxwiz.model` runs the same code on a `pycyxwiz.Sequential`: `summary`
(the same multiply-add counter), `lr_range_test` and `LearningRateRange`,
`saliency`, `smooth_grad`, `integrated_gradients`, and the calibration fits
(`reliability`, `fit_temperature`, `fit_platt`, `fit_isotonic` and their
`apply` functions). Grad-CAM needs a convolution, which a Python
`Sequential` cannot hold yet. `examples/python/model_tools.py` runs all of it
on a small network; the reference is section 6.5 of
`python_doc/pycyxwiz.md`, checked against PyTorch and scikit-learn by
`cyxwiz-backend/python/tests/test_python_model.py` (ctest
`pycyxwiz_python_model`).
