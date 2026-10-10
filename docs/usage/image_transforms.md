# Image transforms

Eight nodes change images on their way from the Data Input to the model:
**Center Crop**, **Random Crop**, **Horizontal Flip**, **Vertical Flip**,
**Image Rotate**, **Color Jitter**, **Image Gaussian Blur** and **Grayscale**.
They follow `torchvision.transforms`: each one gives the same pixels as the
torchvision function with the same settings.

## Where they go

Put them between **Resize** and **Normalize**, in the order you want them
applied:

```
Data Input (images) -> Split -> Loader -> Resize 72 -> Random Crop 64
  -> Horizontal Flip -> Color Jitter -> Normalize -> Conv2D -> ...
```

- Resize comes first: every image is decoded at the Resize size.
- Normalize comes last: the transforms work on pixels in `[0, 1]`.
- The compiler refuses a transform before Resize or after Normalize, and
  names the node.

## Training, validation and test

| Node | Training | Validation and test |
| --- | --- | --- |
| Random Crop | random position per image | centre crop of the same size |
| Horizontal / Vertical Flip | flipped with `probability` | unchanged |
| Image Rotate | random angle in `[-max_angle, max_angle]` with `probability` | unchanged |
| Color Jitter | random factors and order per image | unchanged |
| Center Crop | centre crop | centre crop |
| Image Gaussian Blur | blurred | blurred |
| Grayscale | one channel | one channel |

Random nodes make each training epoch see slightly different images, which
reduces overfitting. Validation and test always see the same images, so their
scores are comparable from epoch to epoch. The random choices come from the
Data Loader's seed, so a run repeats exactly with the same seed.

## The nodes

| Node | Settings | Notes |
| --- | --- | --- |
| Center Crop | `width`, `height` | Must fit the image. The model's input becomes `[height, width, C]`. |
| Random Crop | `width`, `height` | Resize a little larger than the crop so positions differ. |
| Horizontal Flip | `probability` (0.5) | Avoid it for text and digits. |
| Vertical Flip | `probability` (0.5) | For images with no natural up: satellite, microscopy. |
| Image Rotate | `max_angle` (15), `probability` (0.5), `interpolation` (nearest / bilinear) | Same size. Corners outside the image become 0. |
| Color Jitter | `brightness`, `contrast`, `saturation` (0.2), `hue` (0.1) | Factors come from `[max(0, 1 - v), 1 + v]`; hue shifts by `[-hue, hue]`, with `hue` at most 0.5. 0 turns an adjustment off. |
| Image Gaussian Blur | `kernel_size` (5, odd), `sigma` (1.0) | Edges are reflected. `kernel_size / 2` must be smaller than both sides. |
| Grayscale | none | `0.2989 R + 0.587 G + 0.114 B`. The model's input becomes `[H, W, 1]`. |

The **As compiled** card shows the input shape after the crops and Grayscale.

## Speed

The batcher reads and decodes the image files on the CPU and uploads each batch
to the GPU once. The transforms and Normalize then run on the whole batch on
the GPU, and the batch stays there for the model. Without a GPU the same code
runs on ArrayFire's CPU backend. Each image's random settings (flip or not,
crop position, angle, jitter factors) are a few numbers drawn on the CPU.

## Notes

- Code export (PyTorch / TensorFlow / Keras / PyCyxWiz) does not include image
  transforms yet and says so; train these graphs in Studio.
- They work on image Data Inputs (folders of image files). Tabular pixel rows
  do not go through them.
- Checked against torchvision: `image_transforms_torchvision_parity` compares
  every node with `torchvision.transforms.functional` on CUDA, OpenCL and the
  ArrayFire CPU backend, including 90-degree rotation on even sizes, bilinear
  rotation, every Color Jitter order and one-channel images.
  `image_transform_graph_contract` checks the compiled shapes, the refusals, and
  that random nodes change training batches only.
