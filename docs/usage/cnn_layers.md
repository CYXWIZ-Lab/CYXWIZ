# Convolutional (CNN) layers

An image graph trains as a CNN when its first model layer is a convolution
layer. The layers run on whole images, `[H, W, C]` per sample, until a
**Flatten** or **Global Avg Pool** turns each sample into a row for Dense.

```
Image Data Input -> Resize 64x64 -> Conv2D 16 -> ReLU -> MaxPool2D -> Conv2D 32 -> ReLU
                 -> Global Avg Pool (or Flatten) -> Dense 2 -> Cross Entropy -> Adam
```

The Resize node sets the input shape (`[height, width, 3]`); the Properties
"As compiled" card shows each layer's shape before training starts.

## Layers before Flatten / Global Avg Pool

| Node | Output shape | PyTorch |
| --- | --- | --- |
| Conv2D (`filters`, `kernel_size`, `stride`, `padding`: a number, `same` or `valid`) | `[floor((H + 2p - k) / s) + 1, ..., filters]` | `nn.Conv2d` |
| MaxPool2D / AvgPool2D (`pool_size`, `stride`) | same formula, channels kept | `nn.MaxPool2d` / `nn.AvgPool2d` |
| ConvTranspose2D | `(H - 1) x s - 2p + k + output_padding` | `nn.ConvTranspose2d` |
| GroupNorm, InstanceNorm | unchanged | `nn.GroupNorm` / `nn.InstanceNorm2d` |
| Upsample, PixelShuffle | scaled | `nn.Upsample` / `nn.PixelShuffle` |
| Activations, Dropout | unchanged | PReLU only with one shared slope (`num_parameters` 1) |

## Ending the convolution section

| Node | Row per sample | Dense after it takes | PyTorch |
| --- | --- | --- | --- |
| Flatten | every value, `H x W x C` | `H x W x C` inputs | `torch.flatten(x, 1)` |
| Global Avg Pool | each channel's mean over H and W, `C` | `C` inputs | `adaptive_avg_pool2d(x, 1).flatten(1)` |

Global Avg Pool keeps the model small: after a `[16, 16, 32]` feature map,
Dense(2) needs 66 weights instead of Flatten's 16,386. Its gradient spreads
each channel's gradient evenly over the `H x W` positions.

Rules the compiler checks:

- Global Avg Pool must follow a convolution, pooling or normalisation layer
  (or their activations); after Flatten or Dense there is no `[H, W, C]`
  sample to average.
- No convolution layer after the Flatten or Global Avg Pool.
- Only the layers in the first table run before it.

## Checked against PyTorch

`spatial_layers_pytorch_parity` replays every layer above, Global Avg Pool
included, against torch (forward, input gradient and parameter gradients;
`tests/computation_truth/fixtures/spatial_layers_pytorch.json`).
`cnn_graph_training_contract` compiles the example graph with Flatten and with
Global Avg Pool, checks the shapes and parameter counts against torch's, and
runs it forward and backward.

The code export does not write CNN layers with their settings yet.
