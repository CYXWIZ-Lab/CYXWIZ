---
redirect_from:
  - ../../../usage/cnn_layers.md
related:
  - pooling.md
  - normalization.md
  - ../../../tutorials/image-classifier-resnet.md
  - ../../../user-guide/graph-editor/branched-graphs.md
---

## Convolutional networks

An image graph trains as a CNN when its first model layer is a convolution.
The layers run on whole images, `[H, W, C]` per sample, until a
[Flatten](../shape-ops/flatten.md), [Global Avg Pool](../pooling/global-avg-pool.md)
or [Global Max Pool](../pooling/global-max-pool.md) turns each sample into a
row for [Dense](../ml-layers/dense.md).

```text
Image Data Input -> Resize 64x64 -> Conv2D 16 -> ReLU -> MaxPool2D -> Conv2D 32 -> ReLU
                 -> Global Avg Pool (or Flatten) -> Dense 2 -> Cross Entropy -> Adam
```

The Resize node sets the input shape (`[height, width, 3]`); the **As
compiled** card in Properties shows each layer's shape before training
starts. [Conv1D](../ml-layers/conv1d.md) runs the same way on sequences and
[Conv3D](../ml-layers/conv3d.md) on volumes; their pages say how to feed them.

### Layers that run on the feature map

Only these layers run before the Flatten or global pool. Each follows the
PyTorch layer named.

| Node | Output shape | PyTorch |
| --- | --- | --- |
| [Conv2D](../ml-layers/conv2d.md) | `[floor((H + 2p - k) / s) + 1, ..., filters]` | `nn.Conv2d` |
| [Depthwise Conv2D](../ml-layers/depthwise-conv2d.md) | same formula, `C x M` channels (M = depth multiplier); channel `c*M + m` sees only input channel `c` | `nn.Conv2d(C, C*M, k, groups=C)` |
| [MaxPool2D](../pooling/maxpool2d.md), [AvgPool2D](../pooling/avgpool2d.md) | same formula, channels kept | `nn.MaxPool2d`, `nn.AvgPool2d` |
| [ConvTranspose2D](../upsampling/convtranspose2d.md) | `(H - 1) x s - 2p + k + output_padding` | `nn.ConvTranspose2d` |
| [BatchNorm](../normalization/batchnorm.md) | unchanged; one mean and variance per channel over the batch and every pixel | `nn.BatchNorm2d` (after Flatten the same node is `nn.BatchNorm1d`) |
| [GroupNorm](../normalization/groupnorm.md), [InstanceNorm](../normalization/instancenorm.md) | unchanged | `nn.GroupNorm`, `nn.InstanceNorm2d` |
| [Squeeze-Excitation](../normalization/squeeze-excitation.md) | unchanged; each channel multiplied by a learned value between 0 and 1 | `torchvision.ops.SqueezeExcitation` |
| [Adaptive Average Pool](../pooling/adaptive-average-pool.md) | `[s, s, C]` for any H, W | `nn.AdaptiveAvgPool2d(s)` |
| [Upsample](../upsampling/upsample.md), [Pixel Shuffle](../upsampling/pixel-shuffle.md) | scaled | `nn.Upsample`, `nn.PixelShuffle` |
| Activations, [Dropout](../regularization/dropout.md) | unchanged | PReLU only with one shared slope (`num_parameters` 1) |

Hardswish, Hardsigmoid and ReLU6 are the activations of MobileNet-style
models; see the [Activation nodes](activation.md).

### Settings a published network needs

- **Bias** on Conv2D, Depthwise Conv2D, Conv1D, Conv3D and ConvTranspose2D
  is on by default. Turn it off for a convolution that a BatchNorm follows:
  torchvision's ResNet and MobileNet are built that way (`bias=False`), and
  their weight files have no bias for those layers.
- **Padding** on MaxPool2D and AvgPool2D adds cells on every side before
  pooling (at most half the pool size). ResNet's first pool is size 3,
  stride 2, padding 1. AvgPool2D counts the added zeros in its average, as
  `torch.nn.AvgPool2d` does by default.
- Conv2D with `same` padding gives ResNet's paddings: 3 for the 7 x 7
  convolution, 1 for 3 x 3, 0 for 1 x 1.

### Residual blocks

A layer's output can feed two nodes, and [Add](../merge-ops/add.md) joins
them again, before Flatten as well as after it. That is a ResNet block:

```text
... -> ReLU --+--> Conv 3x3 (stride 2) -> BatchNorm -> ReLU -> Conv 3x3 -> BatchNorm --+
              |                                                                       Add -> ReLU -> ...
              +--> Conv 1x1 (stride 2) -> BatchNorm ----------------------------------+
```

- The two inputs of Add must have the same shape `[H, W, C]`. When the main
  path changes the size or the channels (stride 2, more filters), give the
  shortcut a 1x1 Conv with the same stride and filters, as above; when it
  does not, link the block's input straight to Add (identity shortcut).
- Each layer is built from its own input in the graph, so a shortcut Conv
  takes the block's input, not the main path's.
- BatchNorm learns while training (batch statistics; running mean and
  variance updated) and uses the running statistics when the model is tested
  or used.

### Checked against PyTorch

Every layer above, the global pools and Conv1D included, is replayed
against torch: forward output, input gradient and parameter gradients. A
residual CNN (stem, downsampling block with a projection shortcut, Global
Avg Pool, Dense) given torch's weights matches torch's logits, every layer's
gradients and BatchNorm's running statistics on CUDA, OpenCL and the CPU. See
[Numerical accuracy](../../../concepts/numerical-accuracy.md).

Which of these layers each code export writes is in
[Model formats](../../../concepts/model-formats.md#what-each-framework-writes).
