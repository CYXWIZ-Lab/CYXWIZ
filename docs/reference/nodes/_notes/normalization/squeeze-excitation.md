## How the block works

Squeeze-Excitation lets the model weigh its channels for each image. It
takes each channel's mean over the map (squeeze), passes the `C` means
through Dense(`C / reduction`), the inner activation, Dense(`C`) and the
gate (excite), and multiplies every channel of the map by its value between
0 and 1. It is `torchvision.ops.SqueezeExcitation(C, C // r)`; with the
Hardsigmoid gate and Hardswish inside, it is MobileNetV3's block.

Put it after a convolution block (Conv2D, BatchNorm, activation), as SENet,
MobileNetV3 and EfficientNet do. A reduction of 16 is the paper's value; use
4 where the layer has few channels (the hidden width is never below 1). It
adds about `2 x C x C / reduction` parameters.

- **Hidden width**: channels / reduction, or an exact number. Pretrained
  MobileNetV3 uses widths no reduction gives (32 for 120 channels).
- **Inner activation**: ReLU, Hardswish or Swish, between the two Dense
  steps.
- **Gate**: Sigmoid, or Hardsigmoid as in MobileNetV3.
- **Bias**: on or off, for both Dense steps.

The node is one box on the canvas with six steps inside. **Inside this
layer** in Properties lists them in order (mean over the map, Dense,
activation, Dense, gate, multiply) with each step's output for one image and
the weights of the two Dense steps. The defaults are the paper's block.

The code exports refuse a graph with Squeeze-Excitation and say so.
`examples/cyxgraph/cats_dogs_se_hardswish.cyxgraph` is a small CNN that uses
it.
