# A small CNN from modern small-model parts

`cats_dogs_se_hardswish.cyxgraph` is a cats-vs-dogs classifier built with the
pieces MobileNetV3-style networks use: **Hardswish** activations,
**Squeeze-Excitation** after the wider convolution blocks, and **SGD** with
Nesterov momentum, weight decay and a **one-cycle** learning rate.

```
Data Input -> Split 80/10/10 -> Loader (batch 64, 8 epochs) -> Resize 64
  -> Horizontal Flip -> Normalize (same: mean 0.5, std 0.5)
  -> Conv 16 -> BN -> Hardswish
  -> Conv 32 /2 -> BN -> Hardswish -> Squeeze-Excitation (reduction 4)
  -> Conv 64 /2 -> BN -> Hardswish -> Squeeze-Excitation (reduction 4)
  -> Global Avg Pool -> Dense 1 -> BCE with Logits
  -> SGD (learning rate 0.05, momentum 0.9, Nesterov, weight decay 0.0005
          skipping biases and BatchNorm, one_cycle with 30 % rise) -> Output
```

## Data

Any folder of labelled images with two classes. It was made for the Kaggle
"Dogs vs. Cats" set (25,000 JPEGs with a CSV of `image,labels`); the example
saves no path.

## Run it

1. Open the graph (**File > Open**, or the Asset Browser).
2. Double-click **Cats_Dogs Data**, pick your image folder and how its labels
   are given, then **OK**.
3. **Train**. The Training Dashboard's learning-rate line rises from 0.002 to
   0.05 over the first 30 % of the updates, then falls to almost 0.

## What each part does

- **Hardswish** (`x * relu6(x + 3) / 6`) is a cheap stand-in for Swish.
- **Squeeze-Excitation** takes each channel's mean over the feature map and
  learns one value between 0 and 1 per channel to multiply the map by, so the
  model can weigh its channels for each image. `reduction` 4 keeps the gate's
  hidden width at a quarter of the channels.
- **Nesterov** steps along the gradient plus the momentum's look-ahead.
  **Weight decay** pulls the convolution and Dense weights towards 0;
  `norms_and_biases` leaves biases and BatchNorm scales out.
- **one_cycle** needs no scheduler node: it is the optimizer's LR Schedule,
  with Learning Rate as the peak and Warmup Ratio as the rising share.

## Try

- Swap a Hardswish for ReLU, or remove a Squeeze-Excitation node, and compare
  validation accuracy.
- `reduction` 16 (the paper's value) makes the gate narrower.
- LR Schedule `warmup_cosine` with Warmup Ratio 0.05, for a gentler curve.

More: `docs/usage/cnn_layers.md` (Squeeze-Excitation),
`docs/usage/lr_schedulers.md` (the optimizer's own schedule),
`docs/usage/regularization.md` (SGD's weight decay and Nesterov).
