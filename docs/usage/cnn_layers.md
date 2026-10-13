# Convolutional (CNN) layers

An image graph trains as a CNN when its first model layer is a convolution
layer. The layers run on whole images, `[H, W, C]` per sample, until a
**Flatten**, **Global Avg Pool** or **Global Max Pool** turns each sample into a
row for Dense.

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
| Depthwise Conv2D (`kernel_size`, `stride`, `padding`, `depth_multiplier` M) | same formula, `C x M` channels; channel `c*M + m` sees only input channel `c` | `nn.Conv2d(C, C*M, k, groups=C)` |
| MaxPool2D / AvgPool2D (`pool_size`, `stride`) | same formula, channels kept | `nn.MaxPool2d` / `nn.AvgPool2d` |
| ConvTranspose2D | `(H - 1) x s - 2p + k + output_padding` | `nn.ConvTranspose2d` |
| GroupNorm, InstanceNorm | unchanged | `nn.GroupNorm` / `nn.InstanceNorm2d` |
| Adaptive Avg Pool (`output_size` s) | `[s, s, C]` for any H, W (cell i averages rows `floor(i*H/s)` to `ceil((i+1)*H/s) - 1`) | `nn.AdaptiveAvgPool2d(s)` |
| Upsample, PixelShuffle | scaled | `nn.Upsample` / `nn.PixelShuffle` |
| Activations, Dropout | unchanged | PReLU only with one shared slope (`num_parameters` 1) |

## Ending the convolution section

| Node | Row per sample | Dense after it takes | PyTorch |
| --- | --- | --- | --- |
| Flatten | every value, `H x W x C` | `H x W x C` inputs | `torch.flatten(x, 1)` |
| Global Avg Pool | each channel's mean over H and W, `C` | `C` inputs | `adaptive_avg_pool2d(x, 1).flatten(1)` |
| Global Max Pool | each channel's maximum over H and W, `C` | `C` inputs | `adaptive_max_pool2d(x, 1).flatten(1)` |

Global Avg Pool keeps the model small: after a `[16, 16, 32]` feature map,
Dense(2) needs 66 weights instead of Flatten's 16,386. Its gradient spreads
each channel's gradient evenly over the `H x W` positions; Global Max Pool
sends it to the position holding the maximum (the first one, as torch).

Rules the compiler checks:

- A global pool must follow a convolution, pooling or normalisation layer
  (or their activations); after Flatten or Dense there is no `[H, W, C]`
  sample to average.
- No convolution layer after the Flatten or Global Avg Pool.
- Only the layers in the first table run before it.

## Conv1D: convolution over a sequence

Conv1D slides its kernel along one axis. It runs on `[L, C]` samples: a
length `L` with `C` channels, as `torch.nn.Conv1d` on `[N, C, L]`. Output:
`[floor((L + 2p - kernel_size) / stride) + 1, filters]`; `padding` is `same`
(keeps `L` at stride 1, odd kernels) or `valid` (none).

It gets its sequence one of two ways:

- **First model layer**: it reads each input row as channels laid end to
  end, torch's `x.view(N, C, L)`:

  | Data | Channels C | Length L |
  | --- | --- | --- |
  | Time Series Window (`input_width` W, `feature_cols`) | 1 + the feature columns | W |
  | Audio features (Spectrogram, Mel, MFCC) | frequency bins (rows) | frames |
  | Table rows of F numbers | 1 | F |

- **After an Embedding** (text): `[L, E]` token vectors, convolved over the
  tokens with one channel per embedding dimension, torch's
  `x.transpose(1, 2)`. Activations and Dropout may sit between them.

```
Text: Data Input -> Embedding -> Conv1D -> ReLU -> Global Max Pool -> Dense -> Cross Entropy
Series: Data Input -> Time Series Window -> Conv1D -> ReLU -> Flatten -> Dense -> MSE
```

Only Conv1D, activations and Dropout run on the sequence; end it with
**Flatten** (`C x L` values per sample, torch's `flatten` order), **Global
Avg Pool** (each channel's mean over `L`) or **Global Max Pool** (its
maximum: max-over-time pooling for text) before Dense. The compiler reports
on the node: Conv1D anywhere else (after Dense, after the Flatten), a Dense
straight after Conv1D, a kernel longer than the sequence.

## Checked against PyTorch

`spatial_layers_pytorch_parity` replays every layer above, the global pools
and Conv1D included, against torch (forward, input gradient and parameter gradients;
`tests/computation_truth/fixtures/spatial_layers_pytorch.json`).
`cnn_graph_training_contract` compiles the example graph with Flatten and with
Global Avg Pool, checks the shapes and parameter counts against torch's, and
runs it forward and backward. `conv1d_graph_pytorch_parity` builds a Conv1D
graph on table rows (Flatten) and on an Embedding (Global Avg Pool, and the
text CNN with Global Max Pool), sets
torch's parameters on the built model and matches torch's output and every
parameter gradient, then trains the rows graph with the TrainingExecutor.

The code export does not write CNN layers with their settings yet.
