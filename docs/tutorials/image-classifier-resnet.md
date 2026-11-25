---
title: Train a small ResNet image classifier from scratch
description: Train a residual network on a folder of labelled images with nodes only, from leaving out poor images to a test score.
type: tutorial
area: training
audience: user
status: experimental
since: 1.0.8
owner: docs
last_reviewed: 2026-11-25
related:
  - index.md
  - ../user-guide/data-studio/quality-analyzer.md
  - ../user-guide/evaluation/test-a-trained-model.md
  - ../main-window/training-dashboard.md
redirect_from:
  - ../usage/image_classifier_resnet.md
---

In this lesson you train a small residual network (ResNet) on a folder of
labelled images, from preparation to a test score, with nothing but nodes.
You learn what a residual block is, which image steps run once and which
run on every batch, and how to read a run that uses label smoothing.

**You need:** lesson 2 done, the graph
`examples/cyxgraph/cats_dogs_resnet_scratch.cyxgraph`, and a two-class
image folder. The graph was built for the 25,000 labelled photos of the
Dogs vs. Cats set published on Kaggle (JPEGs and a CSV of labels), which
is not shipped with the Engine; any two-class image folder works. A GPU
pack makes it practical: 20 epochs on 25,000 images take over an hour on a
mid-range GPU, much longer on a CPU. For more than two classes see
[Change the graph](#change-the-graph).

## 1. Open the graph and point it at your images

1. Open `examples/cyxgraph/cats_dogs_resnet_scratch.cyxgraph` from the
   Asset Browser or **File > Open**. The canvas shows six named groups,
   one per row.
2. Double-click **Cats_Dogs Data**. The example saves no path: pick your
   image folder.
3. Under **Labels from**, choose **Class folders** (`root/cat/*.jpg`,
   `root/dog/*.jpg`) or **Flat folder + labels CSV** (a CSV of
   `image,labels`).
4. Click **OK**. The node loads the image list.
5. Double-click **Quality Analyzer** and click **Analyze** (once per image
   set and Resize size). Look at what it leaves out, then click **OK**.

## 2. Follow the groups

**Data and augmentation**

- Data Input -> Quality Analyzer -> Data Split (80 / 10 / 10, stratified,
  seed 42) -> Data Loader (batch 64, 20 epochs).
- Once, before training (written to the prepared-image file): Resize
  72 x 72, Center Crop 64 x 64.
- On every batch, training images only: Random Resized Crop (scale
  0.35-1, ratio 3/4-4/3, back to 64 x 64), Horizontal Flip (p 0.5), Color
  Jitter (brightness, contrast, saturation 0.2).
- Normalize with the ImageNet per-channel mean and standard deviation.
  **Data Studio > Prepare** lists these steps in order (see
  [Image transforms](../usage/image_transforms.md)).

**Stem + stage 1** (32 channels, 64 x 64)

- Stem: Conv2D 3 x 3 -> BatchNorm -> ReLU.
- A residual block: Conv 3 x 3 -> BN -> ReLU -> Conv 3 x 3 -> BN, then
  **Add** the block's input (the skip), then ReLU. The skip lets the block
  learn a change to its input instead of a whole new mapping, which is what
  makes deep CNNs train.

**Stages 2, 3 and 4** (64, 128 and 256 channels; 32, 16 and 8 pixels)

- The block's first Conv has stride 2 (padding `same` = 1 on a 3 x 3),
  which halves the height and width. The skip cannot be added as it is
  (different size and channel count), so it goes through a 1 x 1 stride-2
  Conv -> BN (the projection), and the two meet at Add.

**Head and training**

- Global Avg Pool (one value per channel: 256) -> Dropout 0.2 -> Dense 1:
  one logit; above 0 means class 1.
- BCE with Logits, **label smoothing 0.1**: targets 0.05 / 0.95 instead of
  0 / 1, which keeps the model from becoming over-confident. The loss then
  cannot reach 0 (its floor is about 0.2), so read accuracy, not the loss
  value.
- AdamW (learning rate 0.002, weight decay 0.05) with **lr_schedule**
  *warmup_cosine*: the rate rises over the first 5 % of updates, then
  follows a cosine down to 1 % of 0.002 (see
  [Learning-rate schedulers](../usage/lr_schedulers.md)).

The model has about 1.2 million weights. The **As compiled** card on each
node shows its shapes, for example Stage 4's BatchNorm: [8, 8, 256] per
sample.

## 3. Train and test

1. Press **F7**. The compile report should say **Ready to train**.
2. Press **F5** to train.
3. Watch the Training Dashboard. The loss and accuracy points are means
   over the last 10 batches (640 images), so they wiggle; compare the
   epoch-end values and validation (one point per epoch). Train accuracy
   can sit below validation: training images are augmented, validation
   images are not. See
   [Charts in the Training Dashboard](../user-guide/evaluation/plotting.md#charts-in-the-training-dashboard).
4. When training finishes, choose **Train > Test > Run Test** (F8). The
   held-out test images are scored in **Data Studio > Results > Test**.

One run on the Dogs vs. Cats set, on an NVIDIA GTX 1050 Ti, took about 3.5
minutes per epoch:

| | |
| --- | --- |
| Images | 25,000; the Quality Analyzer (blur 300) left out 1,099 |
| Split | 19,120 train, 2,389 validation, 2,392 test |
| Validation accuracy | 64.3 % after epoch 1, 89.5 % after 10, 92.97 % after 20 |
| **Test accuracy** | **92.64 %** (loss 0.331) |
| Test AUC | 0.9795 |
| Test macro F1 | 0.926 |

Your numbers move a little with another GPU or Quality Analyzer setting;
the split and the batch order are fixed by their seeds.

## 4. Check the Quality Analyzer's cut-off

The analyzer measures images at the Resize size. The graph sets the blur
cut-off (**Reject blur below**) to 300, not the node's default of 700. On
the Dogs vs. Cats set, 700 left out 11.9 % of the images, more cats
(14.9 %) than dogs (8.9 %), which would tilt the classes; 300 leaves out
4.4 % with no lean and still catches the placeholders, very dark photos
and near-duplicates. Check its "Rejections lean to one class" line on your
own set before training.

## Change the graph

- **More classes**: set the last Dense's units to the number of classes
  and use **Cross Entropy** (with its own `label_smoothing`) instead of BCE
  with Logits.
- **Bigger images**: change Resize and Center Crop (for example 144 ->
  128); add a stage for every doubling to keep the last feature map near
  8 x 8.
- **A quick try**: set the Data Loader's epochs to 2 or 3; the warmup and
  cosine scale to the shorter run (2 epochs reached 76 % validation in the
  run above).

## Next steps

- [Leave poor images out with the Quality Analyzer](../user-guide/data-studio/quality-analyzer.md)
- [Split images and audio](../user-guide/data-studio/split-images-and-audio.md)
- [Test a trained model](../user-guide/evaluation/test-a-trained-model.md)
- [Convolutional layers](../usage/cnn_layers.md)
- [Fine-tune a published ResNet-18](../main-window/pattern-browser.md#starting-from-torchvisions-weights)
  instead of training from scratch
