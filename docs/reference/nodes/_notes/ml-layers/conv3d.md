---
redirect_from:
  - ../../../usage/conv3d.md
---

## Convolution over volumes

Conv3D convolves **volumes**: CT or MRI scans, voxel grids, or short video
clips with time as the depth axis. It slides a cubic
`kernel x kernel x kernel` filter through depth, height and width, the way
Conv2D slides a square one over an image, following `torch.nn.Conv3d`.

Each output axis is `X' = floor((X + 2p - k) / s) + 1`, with
`p = (k - 1) / 2` for `same` and `0` for `valid`; `same` keeps D, H and W at
stride 1 (odd kernels only). The **As compiled** card shows the shapes the
model trains with.

### Feed it volumes

Set the Data Input's **shape** to `[D, H, W, C]`, for example
`[16, 16, 16, 1]` for a 16-voxel cube with one channel. Each data row then
holds one volume of `C x D x H x W` numbers, channel by channel: all of
channel 0 (depth-major, then height, then width), then channel 1, and so on.
This is PyTorch's `x.view(N, C, D, H, W)` order, so a NumPy array of shape
`[N, C, D, H, W]` saved with `reshape(N, -1)` is already in the right order.

### Wire it

Conv3D opens a **volume section**, which ends at a Flatten:

```text
Data Input (shape [D, H, W, C]) -> Conv3D -> ReLU -> Conv3D -> ReLU -> Flatten -> Dense -> loss
```

- Conv3D is the first model layer (activations or Dropout may come first),
  or follows another Conv3D.
- Inside the section only Conv3D, activations and Dropout run.
- End the section with Flatten before Dense. Use stride 2 on later Conv3D
  layers to shrink the volume first, which keeps the Dense head small.

The compiler refuses a graph that breaks these rules and names the node.

### Example: shapes in a voxel grid

`examples/cyxgraph/volumes/conv3d_volume_shapes.cyxgraph` trains on
`examples/datasets/volume_shapes.csv`: 480 noisy 8x8x8 volumes, each holding
a ball, a rod or a plate at a random position, labelled in the `shape`
column (`generate_volume_shapes.py` next to the data rebuilds it).

```text
Data Input (shape [8, 8, 8, 1]) -> Split -> Loader -> Conv3D (8, same) -> ReLU
  -> Conv3D (16, stride 2, valid) -> ReLU -> Flatten -> Dense (32) -> ReLU
  -> Dense (3) -> Cross Entropy -> Adam -> Output
```

The shapes go `[8,8,8,1] -> [8,8,8,8] -> [3,3,3,16] -> 432 -> 32 -> 3`.

### Cost

A Conv3D layer has `filters x C x k^3` weights plus `filters` biases, and
does `2 x C x k^3 x filters x D' x H' x W'` operations per sample. A 3x3x3
kernel costs three times a 3x3 kernel per output voxel, and volumes have many
voxels, so start with few filters and small volumes. The convolution is one
gather and one matrix product on the device, forward and backward, so
training stays on the GPU (or on ArrayFire's CPU backend).

The weight is stored as `[filters, C x k^3]`, PyTorch's
`conv.weight.flatten(1)`. The code exports refuse a graph with Conv3D and
say so; train it in Studio. Shapes, forward output, input gradient and every
parameter gradient are checked against PyTorch for one to three channels,
`same` and `valid`, stride 2, a 1x1x1 kernel and two stacked layers.
