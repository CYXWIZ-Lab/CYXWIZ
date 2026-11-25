## Methods and directions

Colour images are made grey first.

| `method` | What it computes | Settings it reads |
| --- | --- | --- |
| `canny` | thin white edges on black (two thresholds, 0 to 255 scale) | `threshold1`, `threshold2` |
| `sobel` | the first derivative, smoothed | `kernel_size` (1, 3, 5 or 7), `direction` |
| `scharr` | the same with a more accurate 3 x 3 kernel | `direction` |
| `laplacian` | the second derivative: answers to any direction, and to noise | `kernel_size` |
| `log` | Laplacian of Gaussian: a blur of `sigma` first, so it answers to edges of that scale and not to noise | `sigma`, `kernel_size` |
| `dog` | Difference of Gaussians: blur `sigma` minus blur `sigma2`, the detail between the two scales (close to LoG when `sigma2` is 1.6 x `sigma`) | `sigma`, `sigma2` (greater than `sigma`) |

`direction` (Sobel and Scharr):

- `magnitude`: the mean of both directions' sizes, 0.5 |gx| + 0.5 |gy|.
- `x`: |gx|, changes from left to right, so vertical edges light up.
- `y`: |gy|, changes from top to bottom, so horizontal edges light up.
- `angle`: the gradient's direction, 0 to 360 degrees stored as 0 to 255. 0
  is brighter to the right, 64 brighter downwards, 128 brighter to the
  left, 191 brighter upwards; flat areas are 0. The value wraps (255 is next
  to 0), so it suits looking at more than feeding a model.

Strengths are stored as their size, cut at 255.

**Keep image** (`keep_image`) keeps the image and adds the edge map as one
more channel: an RGB image becomes 4 channels (R, G, B, edges), a grey one
2, and the model's first layer reads `[H, W, 4]`. See
[More than three channels](../categories/preprocessing.md#more-than-three-channels).

```text
Data Input (images) -> Resize 72 -> Image Edges (canny, keep image)
  -> Center Crop 64 -> Normalize (same 0.5 / 0.5) -> Conv2D -> ...
```

`examples/cyxgraph/cats_dogs_edges_channel.cyxgraph` is a small
cats-vs-dogs CNN that reads the photo and its edges.
