# Image transforms

Ten nodes change images on their way from the Data Input to the model:
**Center Crop**, **Random Crop**, **Horizontal Flip**, **Vertical Flip**,
**Image Rotate**, **Color Jitter**, **Image Gaussian Blur**, **Grayscale**,
**Morphology Transform** and **Advanced Augment**. They follow `torchvision.transforms` (morphology
follows `kornia.morphology`): each one gives the same pixels as the reference
function with the same settings.

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
| Morphology Transform | applied | applied |
| Advanced Augment (cutout, random_erasing) | patch erased with `probability` | unchanged |
| Advanced Augment (mixup, cutmix) | batch and labels mixed with `probability` | unchanged |
| Advanced Augment (randaugment) | `num_ops` random ops per image with `probability` | unchanged |

Random nodes make each training epoch see slightly different images, which
reduces overfitting. Validation and test always see the same images, so their
scores are comparable from epoch to epoch. The random choices come from the
Data Loader's seed, so a run repeats exactly with the same seed.

## The nodes

| Node | Settings | Notes |
| --- | --- | --- |
| Center Crop | `width`, `height` | Must fit the image. The model's input becomes `[height, width, C]`. |
| Random Crop | `width`, `height`, `padding` (0) | Resize a little larger than the crop so positions differ, or pad: the CIFAR recipe is Random Crop 32 with `padding` 4 on 32 x 32 images. Validation takes the centre of the padded image, so a full-size crop leaves it unchanged. |
| Horizontal Flip | `probability` (0.5) | Avoid it for text and digits. |
| Vertical Flip | `probability` (0.5) | For images with no natural up: satellite, microscopy. |
| Image Rotate | `max_angle` (15), `probability` (0.5), `interpolation` (nearest / bilinear) | Same size. Corners outside the image become 0. |
| Color Jitter | `brightness`, `contrast`, `saturation` (0.2), `hue` (0.1) | Factors come from `[max(0, 1 - v), 1 + v]`; hue shifts by `[-hue, hue]`, with `hue` at most 0.5. 0 turns an adjustment off. |
| Image Gaussian Blur | `kernel_size` (5, odd), `sigma` (1.0) | Edges are reflected. `kernel_size / 2` must be smaller than both sides. |
| Grayscale | none | `0.2989 R + 0.587 G + 0.114 B`. The model's input becomes `[H, W, 1]`. |
| Morphology Transform | `operation` (open), `kernel_size` (3, odd) | Flat square kernel, pixels outside the image ignored. erode = local minimum, dilate = local maximum, open = dilate(erode), close = erode(dilate), gradient = dilate - erode, tophat = image - open, blackhat = close - image. |
| Advanced Augment | `method` (cutout / random_erasing / mixup / cutmix / randaugment), `probability` (0.5), `alpha` (1.0), `num_ops` (2), `magnitude` (9), `cutout_size` (16), `scale_min` / `scale_max` (0.02 / 0.33), `ratio_min` / `ratio_max` (0.3 / 3.3), `value` (0) | cutout: a square centred at a random pixel, clipped at the edges (DeVries & Taylor). random_erasing: a box of random area and aspect ratio (torchvision RandomErasing); after ten misses the image stays as it is. `value` is a pixel value, written before Normalize. mixup / cutmix and randaugment: see below. |

## MixUp and CutMix

Advanced Augment with `method` mixup or cutmix mixes whole training batches,
as `torchvision.transforms.v2.MixUp` / `CutMix` do:

- One weight `lambda` per batch, drawn from Beta(`alpha`, `alpha`); `alpha` 1.0
  draws it uniformly from 0 to 1.
- Each image is mixed with the image before it in the batch (the first with the
  last). mixup blends the two: `lambda * own + (1 - lambda) * other`. cutmix
  pastes a box from the other image covering `1 - lambda` of the area, and
  `lambda` becomes the share of the image left unpasted.
- The labels are mixed with the same `lambda`, so a label becomes, for example,
  70% cat and 30% dog. Cross Entropy learns from these soft labels directly, so
  the loss must be Cross Entropy; the compiler refuses other losses.
- `probability` is the chance that a batch is mixed. One mixup or cutmix per
  graph. Validation and test are never mixed.
- Training accuracy counts a mixed image as right when the model picks the
  class with the larger share of its label.

## RandAugment

Advanced Augment with `method` randaugment follows
`torchvision.transforms.v2.RandAugment`. Each training image (with
`probability`; set it to 1 for torchvision's behaviour) takes `num_ops` ops,
each picked at random from 14:

| Op | Strength at `magnitude` m (of 30) |
| --- | --- |
| Identity | none |
| Shear X / Y | shear factor `0.3 m / 30`, either way, about the top-left corner |
| Translate X / Y | `150 / 331` of the width / height `x m / 30` pixels, either way |
| Rotate | `30 m / 30` degrees, either way |
| Brightness / Color / Contrast / Sharpness | factor `1 +/- 0.9 m / 30` |
| Posterize | keeps `8 - round(m / 7.5)` bits |
| Solarize | inverts pixels at or above `1 - m / 30` |
| AutoContrast | stretches each channel to 0..1 |
| Equalize | equalizes each channel's histogram (as uint8, like torchvision) |

The default `num_ops` 2 and `magnitude` 9 are torchvision's. Shear, translate
and rotate use nearest sampling and fill with 0.

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
  rotation, every Color Jitter order, one-channel images, padded Random Crop,
  all seven morphology operations and erasing (clipped boxes, any value).
  MixUp and CutMix are compared with torchvision's own `v2.MixUp` / `v2.CutMix`
  runs (images and labels) on the lambda and box they drew. Every RandAugment op
  is compared through torchvision's own dispatcher with both signs, on three and
  one channels, and the magnitude table matches torchvision's exactly.
  `image_transform_graph_contract` checks the compiled shapes, the refusals, and
  that random nodes change training batches only.
