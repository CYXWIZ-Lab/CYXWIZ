## Ending a convolution section

A CNN's feature map has to become one row per sample before
[Dense](../ml-layers/dense.md). Three nodes do that:

| Node | Row per sample | Dense after it takes | PyTorch |
| --- | --- | --- | --- |
| [Flatten](../shape-ops/flatten.md) | every value, `H x W x C` | `H x W x C` inputs | `torch.flatten(x, 1)` |
| [Global Avg Pool](../pooling/global-avg-pool.md) | each channel's mean over H and W, `C` | `C` inputs | `adaptive_avg_pool2d(x, 1).flatten(1)` |
| [Global Max Pool](../pooling/global-max-pool.md) | each channel's maximum over H and W, `C` | `C` inputs | `adaptive_max_pool2d(x, 1).flatten(1)` |

Global Avg Pool keeps the model small: after a `[16, 16, 32]` feature map,
Dense(2) needs 66 weights instead of Flatten's 16,386. Its gradient spreads
each channel's gradient evenly over the `H x W` positions; Global Max Pool
sends it to the position holding the maximum (the first one, as torch).
After a [Conv1D](../ml-layers/conv1d.md) the global pools work over the
sequence length instead.

Rules the compiler checks:

- A global pool needs a feature map: it follows a convolution, pooling or
  normalisation layer (or their activations). After Flatten or Dense there
  is no `[H, W, C]` sample to pool.
- No convolution layer comes after the Flatten or global pool.

See [Convolutional networks](ml-layers.md#convolutional-networks) for the
layers that run before it.
