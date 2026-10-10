# Conv3D

Conv3D is a convolution over **volumes**: CT or MRI scans, voxel grids, or
short video clips with time as the depth axis. It slides a cubic
`kernel x kernel x kernel` filter through depth, height and width, the way
Conv2D slides a square one over an image. It follows PyTorch's
`torch.nn.Conv3d` rules.

## The node

| Pin / setting | Meaning |
| --- | --- |
| Input | `[D, H, W, C]` volume per sample |
| Output | `[D', H', W', filters]` |
| `filters` (32) | output channels |
| `kernel_size` (3) | cubic kernel: 3 means 3x3x3 |
| `stride` (1) | step on every axis |
| `padding` (same) | `same` keeps D, H and W at stride 1 (odd kernels only); `valid` adds none |

Each output axis is `X' = (X + 2p - k) / s + 1` (rounded down), with
`p = (k - 1) / 2` for `same` and `0` for `valid`. The **As compiled** card
shows the shapes the model trains with.

## Feeding it volumes

Set the Data Input's **shape** to `[D, H, W, C]`, for example `[16, 16, 16, 1]`
for a 16-voxel cube with one channel. Each data row then holds one volume of
`C x D x H x W` numbers, **channel by channel**: all of channel 0 (depth-major,
then height, then width), then channel 1, and so on. This is the same order as
PyTorch's `x.view(N, C, D, H, W)`, so a NumPy array of shape `[N, C, D, H, W]`
saved with `reshape(N, -1)` is already in the right order. With one channel
the order is just depth, height, width.

## Wiring it

Conv3D opens a **volume section**, which ends at a Flatten:

```
Data Input (shape [D, H, W, C]) -> Conv3D -> ReLU -> Conv3D -> ReLU -> Flatten -> Dense -> loss
```

- Conv3D is the first model layer (activations or Dropout may come first), or
  follows another Conv3D.
- Inside the section only Conv3D, activations and Dropout run.
- End the section with Flatten before Dense. Use stride 2 on later Conv3D
  layers to shrink the volume first, which keeps the Dense head small.

The compiler refuses graphs that break these rules and names the node.

## Cost

A Conv3D layer has `filters x C x k^3` weights plus `filters` biases. Its work
is `2 x C x k^3 x filters x D' x H' x W'` per sample. A 3x3x3 kernel costs 3
times a 3x3 kernel per output voxel, and volumes have many voxels, so start
with few filters and small volumes.

On ArrayFire the column matrix comes from one sparse gather, built once for the
volume shape, and the convolution is one matrix product. Backward uses the
transposed gather, so training stays on the device. Builds without ArrayFire
use plain loops.

## Notes

- Code export (PyTorch / TensorFlow / Keras / PyCyxWiz) does not support
  Conv3D yet and says so; train it in Studio.
- The weight is stored as `[filters, C x k^3]`, PyTorch's
  `conv.weight.flatten(1)`.
- Checked against PyTorch: `conv3d_pytorch_parity` covers the compiled shapes,
  the forward output, the input gradient and every parameter gradient. Its
  cases use one, two and three channels, `same` and `valid`, stride 2, a 1x1x1
  kernel, and two Conv3D layers with a ReLU between them.
