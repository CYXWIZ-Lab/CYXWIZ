# An image classifier from scratch: a small ResNet

`examples/cyxgraph/cats_dogs_resnet_scratch.cyxgraph` trains a small residual
network (ResNet) on a folder of labelled images, from preparation to a test
score, with nothing but nodes. It was built for the cats vs dogs set (25,000
JPEGs and a CSV of labels), but it works for any two-class image folder; for
more classes see [Changing it](#changing-it).

## Open it and point it at your images

1. Open the example (Asset Browser or **File > Open**). The canvas shows six
   named groups, one per row.
2. Open **Cats_Dogs Data** (double-click). The example saves no path: pick your
   image folder, and under **Labels from** either **Class folders**
   (`root/cat/*.jpg`, `root/dog/*.jpg`) or **Flat folder + labels CSV** (a CSV
   of `image,labels`). Click **OK**; it loads the image list.
3. Open **Quality Analyzer** and click **Analyze** (once per image set and
   Resize size, see [quality_analyzer.md](quality_analyzer.md)). Look at what it leaves out, then
   **OK**.
4. **Train**. When it finishes, **Train > Test > Run Test** (F8) scores the
   held-out test images in **Data Studio > Results > Test**.

## The groups

**Data and augmentation**
- Data Input -> Quality Analyzer -> Data Split (80 / 10 / 10, stratified,
  seed 42) -> Data Loader (batch 64, 20 epochs).
- Once, before training (written to the prepared-image file): Resize 72 x 72,
  Center Crop 64 x 64.
- Per batch, on the GPU, training images only: Random Resized Crop (scale
  0.35-1, ratio 3/4-4/3, back to 64 x 64), Horizontal Flip (p 0.5), Color
  Jitter (brightness, contrast, saturation 0.2).
- Normalize with the ImageNet per-channel mean and standard deviation.
  Data Studio > Prepare lists these steps in order
  ([image_transforms.md](image_transforms.md)).

**Stem + stage 1** (32 channels, 64 x 64)
- Stem: Conv2D 3 x 3 -> BatchNorm -> ReLU.
- A residual block: Conv 3 x 3 -> BN -> ReLU -> Conv 3 x 3 -> BN, then **Add**
  the block's input (the skip), then ReLU. The skip lets the block learn a
  change to its input instead of a whole new mapping, which is what makes deep
  CNNs train.

**Stages 2, 3 and 4** (64, 128 and 256 channels; 32, 16 and 8 pixels)
- The block's first Conv has stride 2 (padding `same` = 1 on a 3 x 3), which
  halves the height and width. The skip cannot be added as it is (different
  size and channel count), so it goes through a 1 x 1 stride-2 Conv -> BN (the
  projection) and the two meet at Add.

**Head and training**
- Global Avg Pool (one value per channel: 256) -> Dropout 0.2 -> Dense 1: one
  logit; above 0 means class 1.
- BCE with Logits, **label smoothing 0.1**: targets 0.05 / 0.95 instead of
  0 / 1, which keeps the model from becoming over-confident. The loss then
  cannot reach 0 (its floor is about 0.2), so read accuracy, not the loss
  value.
- AdamW (learning rate 0.002, weight decay 0.05) with **lr_schedule
  warmup_cosine**: the rate rises over the first 5 % of updates, then follows
  a cosine down to 1 % of 0.002 ([lr_schedulers.md](lr_schedulers.md)).

The model has about 1.2 million weights; the As compiled card on each node
shows its shapes (for example Stage 4's BatchNorm: [8, 8, 256] per sample).

## What it scored here

On the cats vs dogs set, on one GTX 1050 Ti (2026-10-31), 20 epochs of about
3.5 minutes:

| | |
|---|---|
| Images | 25,000; the Quality Analyzer (blur 300) left out 1,099 |
| Split | 19,120 train, 2,389 validation, 2,392 test |
| Validation accuracy | 64.3 % after epoch 1, 89.5 % after 10, 92.97 % after 20 |
| **Test accuracy** | **92.64 %** (loss 0.331) |
| Test AUC | 0.9795 |
| Test macro F1 | 0.926 |

Your numbers will move a little with another GPU or Quality Analyzer
setting; the split and the batch order are fixed by their seeds.

## Quality Analyzer cut-offs

The analyzer measures images at the Resize size. On this set the default blur
cut-off (700) left out 11.9 % of the images, more cats (14.9 %) than dogs
(8.9 %), which would tilt the classes; 300 leaves out 4.4 % with no lean and
still catches the placeholders, very dark photos and near-duplicates. Check
its "Rejections lean to one class" line on your own set before training.

## Changing it

- **More classes**: Dense units = the number of classes and Cross entropy
  (with its own label_smoothing) instead of BCE with Logits.
- **Bigger images**: change Resize and Center Crop (for example 144 -> 128);
  add a stage for every doubling to keep the last feature map near 8 x 8.
- **A quick try**: set the Data Loader's epochs to 2 or 3; the warmup and
  cosine scale to the shorter run (2 epochs reached 76 % validation here, in
  about 10 minutes).

## Reading the Training Dashboard

The loss and accuracy points are means over the last 10 batches (640
images), so they wiggle; the epoch-end values and validation (one point per
epoch) are the ones to compare. Train accuracy can sit below validation: training
images are augmented, validation images are not. See
[plotting.md](plotting.md#training-dashboard).

See also [cnn_layers.md](cnn_layers.md), [image_transforms.md](image_transforms.md),
[image_split.md](image_split.md), [testing.md](testing.md).
