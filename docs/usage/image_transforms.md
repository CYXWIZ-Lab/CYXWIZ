# Image transforms

Fifteen nodes change images on their way from the Data Input to the model:
**Center Crop**, **Random Crop**, **Random Resized Crop**, **Horizontal Flip**,
**Vertical Flip**, **Image Rotate**, **Random Affine**, **Random Perspective**,
**Color Jitter**, **Random Grayscale**, **Gaussian Noise**, **Image Gaussian
Blur**, **Grayscale**, **Morphology Transform** and **Advanced Augment**. They
follow `torchvision.transforms` (morphology follows `kornia.morphology`): each
one gives the same pixels as the torchvision transform with the same draws.

Nine more run OpenCV once, when the images are prepared:
**Image Contrast (CLAHE)**, **Image Denoise**, **Image Sharpen**,
**Image Edges**, **Image Smooth**, **Image Warp**, **Image Pyramid**,
**Image Filter** and **Image Gabor** (see
[OpenCV steps done once](#opencv-steps-done-once)). Node search marks them
`once`.

## Where they go

Put them between **Resize** and **Normalize**, in the order you want them
applied:

```
Data Input (images) -> Split -> Loader -> Resize 72 -> Random Crop 64
  -> Horizontal Flip -> Color Jitter -> Normalize -> Conv2D -> ...
```

- Resize comes first: every image is decoded once at the Resize size and
  mode (see [Decoded once](#decoded-once)).
- Normalize comes last: the transforms work on pixels in `[0, 1]`.
- The compiler refuses a transform before Resize or after Normalize, and
  names the node.

## Training, validation and test

| Node | Training | Validation and test |
| --- | --- | --- |
| Random Crop | random position per image | centre crop of the same size |
| Random Resized Crop | random box per image, resized | the centred box of the nearest allowed aspect, resized |
| Horizontal / Vertical Flip | flipped with `probability` | unchanged |
| Image Rotate | random angle in `[-max_angle, max_angle]` with `probability` | unchanged |
| Random Affine | random angle, shift, scale and shear per image | unchanged |
| Random Perspective | corners moved with `probability` | unchanged |
| Color Jitter | random factors and order per image | unchanged |
| Random Grayscale | gray with `probability` | unchanged |
| Gaussian Noise | noise on every pixel | unchanged |
| Center Crop | centre crop | centre crop |
| Image Gaussian Blur | blurred | blurred |
| Grayscale | one channel | one channel |
| Morphology Transform | applied | applied |
| The seven OpenCV steps | applied (done once) | applied (done once) |
| Advanced Augment (cutout, random_erasing) | patch erased with `probability` | unchanged |
| Advanced Augment (mixup, cutmix) | batch and labels mixed with `probability` | unchanged |
| Advanced Augment (randaugment) | `num_ops` random ops per image with `probability` | unchanged |
| Advanced Augment (autoaugment, trivialaugment, augmix) | the policy's ops per image with `probability` | unchanged |

Random nodes make each training epoch see slightly different images, which
reduces overfitting. Validation and test always see the same images, so their
scores are comparable from epoch to epoch. The random choices come from the
Data Loader's seed, so a run repeats exactly with the same seed. Which
images are train, validation and test is the Data Split's (see
image_split.md).

## The nodes

| Node | Settings | Notes |
| --- | --- | --- |
| Center Crop | `width`, `height` | Must fit the image. The model's input becomes `[height, width, C]`. |
| Random Crop | `width`, `height`, `padding` (0) | Resize a little larger than the crop so positions differ, or pad: the CIFAR recipe is Random Crop 32 with `padding` 4 on 32 x 32 images. Validation takes the centre of the padded image, so a full-size crop leaves it unchanged. |
| Horizontal Flip | `probability` (0.5) | Avoid it for text and digits. |
| Vertical Flip | `probability` (0.5) | For images with no natural up: satellite, microscopy. |
| Random Resized Crop | `width`, `height` (0 = the incoming size), `scale_min` / `scale_max` (0.08 / 1), `ratio_min` / `ratio_max` (0.75 / 1.3333), `interpolation` (bilinear / nearest) | A box covering `scale_min` to `scale_max` of the area, aspect ratio (width / height) from `ratio_min` to `ratio_max`, resized to `width` x `height`; after ten misses, the centred box of the nearest allowed ratio. Bilinear is antialiased when shrinking, as torchvision's resize. 0 x 0 keeps the size coming in, so adding the node does not change the model's input. |
| Image Rotate | `max_angle` (15), `probability` (0.5), `interpolation` (nearest / bilinear) | Same size. Corners outside the image become 0 (torchvision RandomRotation's fill). |
| Random Affine | `degrees` (10), `translate_x` / `translate_y` (0.1), `scale_min` / `scale_max` (0.9 / 1.1), `shear_x_min` / `shear_x_max`, `shear_y_min` / `shear_y_max` (0), `interpolation` (nearest), `fill` (0) | About the centre. The shift is whole pixels, up to `translate_x` of the width and `translate_y` of the height. Pixels from outside the image take `fill`. |
| Random Perspective | `distortion` (0.5), `probability` (0.5), `interpolation` (bilinear), `fill` (0) | Each corner moves inwards by up to `distortion` of half the side. |
| Random Grayscale | `probability` (0.1) | Gray in every channel, so the shape stays the same (unlike Grayscale). One-channel images are unchanged. |
| Gaussian Noise | `mean` (0), `sigma` (0.1), `clip` (on) | Noise from N(`mean`, `sigma`) on pixels in `[0, 1]`, drawn on the device for every batch; `clip` keeps the result in `[0, 1]`. |
| Color Jitter | `brightness`, `contrast`, `saturation` (0.2), `hue` (0.1) | Factors come from `[max(0, 1 - v), 1 + v]`; hue shifts by `[-hue, hue]`, with `hue` at most 0.5. 0 turns an adjustment off. |
| Image Gaussian Blur | `kernel_size` (5, odd), `sigma` (1.0) | Edges are reflected. `kernel_size / 2` must be smaller than both sides. |
| Grayscale | none | `0.2989 R + 0.587 G + 0.114 B`. The model's input becomes `[H, W, 1]`. |
| Morphology Transform | `operation` (open), `kernel_size` (3, odd) | Flat square kernel, pixels outside the image ignored. erode = local minimum, dilate = local maximum, open = dilate(erode), close = erode(dilate), gradient = dilate - erode, tophat = image - open, blackhat = close - image. |
| Advanced Augment | `method` (cutout / random_erasing / mixup / cutmix / randaugment / autoaugment / trivialaugment / augmix), `probability` (0.5), and the chosen method's own settings | cutout: a square centred at a random pixel, clipped at the edges (DeVries & Taylor): `cutout_size` (16), `value` (0). random_erasing: a box of random area and aspect ratio (torchvision RandomErasing); after ten misses the image stays as it is: `scale_min` / `scale_max` (0.02 / 0.33), `ratio_min` / `ratio_max` (0.3 / 3.3), `value`. `value` is a pixel value, written before Normalize. mixup / cutmix and the policies: see below. Properties shows only the chosen method's settings. |

## Augmentation Preset

**Augmentation Preset** is a standard recipe in one node. It compiles into the
nodes it stands for, so it behaves exactly as wiring them by hand:

| `preset` | Same as |
| --- | --- |
| flip | Horizontal Flip (0.5) |
| cifar | Random Crop of the image size with `padding` 4, then Horizontal Flip (0.5) |
| cifar_cutout | cifar, then Advanced Augment cutout 16 on every image |
| randaugment | Horizontal Flip (0.5), then Advanced Augment randaugment (2 ops, magnitude 9) on every image |

Put it between Resize and Normalize like the other nodes. To tune a step,
replace the preset with the single nodes.

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

## AutoAugment, TrivialAugment and AugMix

Three more Advanced Augment methods pick ops from the same list as
RandAugment, with their own strengths. Each runs on a training image with
`probability`; set it to 1 for torchvision's behaviour.

| `method` | Follows | What each image gets | Settings |
| --- | --- | --- | --- |
| autoaugment | `torchvision.transforms.v2.AutoAugment` | one of the policy's 25 learned pairs; each op of the pair runs with its own chance and strength (of 10). The policies also invert images. | `policy`: imagenet, cifar10 or svhn |
| trivialaugment | `TrivialAugmentWide` | one op of 14 at a random strength bin, on wider scales (shear 0.99, shift 32 pixels, rotate 135 degrees, factors 1 +/- 0.99) | `magnitude_bins` (31) |
| augmix | `AugMix` | `mixture_width` chains of `chain_depth` ops (-1: 1 to 3) at strength bins below `severity` (of 10), mixed with Dirichlet(`alpha`) weights and blended with the image by Beta(`alpha`, `alpha`); never inverts | `severity` (3), `mixture_width` (3), `chain_depth` (-1), `alpha` (1.0) |

autoaugment and trivialaugment sample with nearest pixels, augmix with
bilinear ones, as torchvision does. All three keep the image's shape.

The **As compiled** card shows the input shape after the crops and Grayscale.

## Decoded once

When training starts, every image of the Data Input is decoded once, on all
CPU cores, at the Resize size and mode, and kept as 8-bit RGB in one file in
the project's `cache/images` folder. The Console says
`Preparing images: 12500 of 25000 decoded` while it runs, and Stop stops it.
The next Train with the same images, size and mode opens that file and starts
at once (`Training reads the 25000 prepared images from ...`). Changing,
adding or removing an image file, the labels CSV, the size or the mode
prepares the images again. The folder keeps sets for 30 days and at most
10 GB in all, oldest first.

Resize modes:

| Mode | What you get |
| --- | --- |
| exact | stretched to the width and height |
| fit | scaled to fit inside, black bars on the short side |
| fill | scaled to cover, the centre kept (the sides cropped) |
| center | not scaled: the centre cut out (black around a smaller image) |

Grey, RGBA and 16-bit files become RGB. A file that cannot be read is left
out of training and named in the Console.

Transforms that give every image the same result in every epoch and split
(**Center Crop**, **Grayscale**, **Image Gaussian Blur**, **Morphology
Transform**), when they come before the first random transform, are done once
too: right after decoding, on the GPU, with the same code the batches use, and
kept as a second set beside the decoded one (`Prepared images: 2 fixed
transforms done once on the device (72x72x3 -> 64x64x3)`). The rest run on each
batch. The done-once pixels are 8-bit again: exact for crops and morphology,
within half a step for Grayscale and blur, as torchvision gives on 8-bit
images. Put random transforms after them to keep this: a Center Crop after a
Random Flip runs on every batch.

Each batch then takes its images from that file: when the whole set fits a
quarter of the GPU's memory it is uploaded once and a batch is picked on the
GPU (`ImageDatasetBatcher: 25000 images (293.8 MB) on the device for the
run`); otherwise each batch's images are copied and uploaded once. The
transforms and Normalize run on the whole batch on the GPU, and the batch
stays there for the model. Without a GPU the same code runs on ArrayFire's
CPU backend. Each image's random settings (flip or not, crop position, angle,
jitter factors) are a few numbers drawn on the CPU.

## Normalize per channel

Normalize comes last, after the image steps, and runs on each batch on the GPU:
every value becomes `(x - mean) / std`. Its **Statistics** setting picks the
numbers:

| Statistics | Settings | For |
| --- | --- | --- |
| Same for every channel | `mean` (0.0), `std` (1.0) | Tables, audio, text features, and images when one pair is enough. Saved graphs open as this. |
| Per channel | `channel_mean`, `channel_std`: one typed row per channel (R, G, B, or grey) | Images, with each channel's own statistics. Data Studio > Explore > Profile > Channel statistics > **Use in Normalize** fills them from the set this graph trains from. |
| ImageNet | none: 0.485, 0.456, 0.406 and 0.229, 0.224, 0.225 (torchvision), shown read-only | Models pretrained on ImageNet. Pick Per channel to change a value; the numbers carry over. |

- The rows follow the images' channels where Normalize sits: after Grayscale or
  Image Edges there is one (grey).
- The compiler refuses Per channel and ImageNet on data that is not images
  (*Per channel needs images: this graph's data is a table. Use Same for every
  channel.*), a count that does not match (*3 means for 1 channel: the images
  have 1 here (Grayscale or Image Edges before Normalize). Give one mean and
  one std per channel (Use in Normalize does).*), ImageNet on images that are
  not RGB, and a std that is not above 0 (*Std G is 0: every std must be
  greater than 0.*). ImageNet adds an Info note: for training from scratch,
  the data's own statistics centre it better.
- The Compile popup's setup list shows *Normalize · per channel, mean 0.528 /
  0.492 / 0.453, std 0.235 / 0.227 / 0.230*; the AS COMPILED card shows the
  formula, the shapes and the values under Details.
- After **Use in Normalize**, Profile says *Normalize uses these statistics:
  the model's inputs centre at 0.00 (R, G, B) with spread 1.00.* with a
  **Show Normalize on canvas** link, and Properties names where the values came
  from under the rows (*From Data Studio: cat_dog, the 64 × 64 set this graph
  trains from, measured 2026-10-29*) with **Measure again in Data Studio**.
  Typing a value or changing Statistics drops that line. Undo takes the whole
  change back.
- The Data Input dialog's Preprocessing list names the same values
  (*Normalized by Normalize per channel, mean 0.528 / ...*).

## OpenCV steps done once

These seven nodes call the Engine's OpenCV classes. Each one gives every image
the same result in every epoch, so it runs once, when the images are
prepared, on every CPU core, in canvas order with the GPU steps that are done
once. Every epoch, and every later run with the same settings, reads the
result from `cache/images`.

```
Data Input (images) -> Resize 64 -> Image Contrast (CLAHE) -> Center Crop 56
  -> Horizontal Flip -> Normalize -> Conv2D -> ...
```

Here Resize and CLAHE run on the CPU and Center Crop on the GPU, all once;
Horizontal Flip and Normalize run on each batch.

| Node | Settings (default) | What it does | Shape |
| --- | --- | --- | --- |
| Image Contrast (CLAHE) | `clip_limit` (2.0), `tile_grid` (8), `colour` (lightness / each_channel) | Contrast Limited Adaptive Histogram Equalization on each tile. `lightness` changes only Lab L, so colours stay; `each_channel` equalizes R, G and B alone. | same |
| Image Denoise | `strength` (10), `colour_strength` (10), `patch_size` (7, odd), `search_size` (21, odd) | Non-local means (`fastNlMeansDenoisingColored`; `fastNlMeansDenoising` for grey). Slow: about 45 ms an image at 64 x 64, so the compiler shows the time; it is spent once. | same |
| Image Sharpen | `amount` (1.0), `radius` (1.0), `threshold` (0, from 0 to 1) | Unsharp mask: image + amount x (image - Gaussian blur of sigma `radius`). With a threshold, only pixels whose detail is above it change. | same |
| Image Edges | `method` (canny / sobel / laplacian / scharr / log / dog), `threshold1` / `threshold2` (50 / 150, Canny), `kernel_size` (3; Sobel 1, 3, 5 or 7; Laplacian and LoG odd up to 31), `direction` (magnitude / x / y / angle, Sobel and Scharr), `sigma` (1.0, LoG and DoG), `sigma2` (1.6, DoG), `keep_image` (false) | Colour images are made grey first. Canny gives white edges on black; the others the edge strength. See [Image Edges](#image-edges). | one channel: `[H, W, 1]`; with `keep_image`, `[H, W, C + 1]` |
| Image Smooth | `method` (box / median / bilateral), `kernel_size` (5, odd), `sigma_color` / `sigma_space` (75 / 75, bilateral) | Box averages, median takes the middle value (removes specks, keeps edges), bilateral mixes only similar colours. A Gaussian blur is Image Gaussian Blur, on the GPU. | same |
| Image Warp | `method` (four_corners / three_points / log_polar / linear_polar), `source_points`, `target_points` (`0,0, 1,0, 1,1, 0,1`), `interpolation` (linear), `width` / `height` (0), `center_x` / `center_y` (0.5), `max_radius` (0.5), `inverse` | The same warp for every image. Points are `x, y` fractions of the image, top-left first, clockwise: four for four_corners, three for three_points. Polar warps unroll the image around the centre, so a rotation becomes a shift; they keep the size. | same, or `width` x `height` |
| Image Pyramid | `operation` (down / up / detail), `levels` (1) | down: Gaussian smoothing then half the size, per level; up: double the size, per level; detail: the Laplacian level `levels - 1` halvings down (the image minus its smoothed copy). | down halves each side per level (rounding up); detail at level 1 keeps it |
| Image Gabor | `orientations` (4; 1 to 16), `wavelengths` (`3, 5`, pixels per cycle), `bandwidth` (1.0 octave; 0.25 to 4), `aspect` (0.5; 0.1 to 1), `response` (energy / even / odd), `keep_image` (false) | A bank of striped kernels over the grey image, one channel per orientation and wavelength. See [Image Gabor](#image-gabor). | `[H, W, orientations x wavelengths]`; with `keep_image`, the image's channels first |
| Image Filter | `preset` (sharpen; custom / identity / box / emboss / outline / prewitt_x / prewitt_y / roberts_x / roberts_y / kirsch / robinson), `direction` (n; ne / e / se / s / sw / w / nw, Kirsch and Robinson), `kernel` (`0 -1 0; -1 5 -1; 0 -1 0`, custom), `normalize` (false), `border` (reflect_101 / reflect / replicate / zero), `output` (clip / absolute / offset) | One fixed kernel over every channel (`cv::filter2D`): each output pixel is the sum of its neighbours times the kernel, the kernel's centre on the pixel. See [Image Filter](#image-filter). | same |

- The prepared images are 8-bit, so each step's result is stored as 0 to 255:
  edge strength as its magnitude, pyramid detail around mid-grey (128 = no
  detail).
- They must come before the first random transform (flips, Random Crop, Image
  Rotate, Color Jitter, Advanced Augment, Augmentation Preset). After one, the
  graph does not compile: `Runs once on the prepared images, so it must come
  before the first random step: move it before Horizontal Flip (p 0.5)`.
- The compiler checks each step on the size it receives: `4 levels down from
  56 x 56 gives 4 x 4: at most 3 levels here`, `Corner 3 of the source is
  outside the image: points are fractions from 0 to 1`, a tile grid larger
  than the image, even kernels. Image Denoise adds a warning with its time;
  Image Edges says the images become one channel.
- The set's key includes every step and setting: changing one prepares a new
  set, and the old one stays in Data Studio > Explore > Profile's cached sets
  (`64 x 64 · CLAHE 2.0 / 8, Center Crop 56 x 56`) until it is pruned. The
  Console line names both kinds: `Prepared images: 2 fixed steps done once
  (1 on the device, 1 OpenCV on the CPU; 64x64x3 -> 56x56x3)`.
- Data Studio > Explore > Images > What the model sees shows each step on the
  picked image, an OpenCV step as `56 x 56, once on the CPU`. The Properties
  panel's AS COMPILED card shows the step's input and output shapes.
- Same result in training, validation and test.

### Image Edges

| `method` | What it computes | Settings it reads |
| --- | --- | --- |
| `canny` | thin white edges on black (two thresholds, 0 to 255 scale) | `threshold1`, `threshold2` |
| `sobel` | the first derivative, smoothed | `kernel_size` (1, 3, 5 or 7), `direction` |
| `scharr` | the same with a more accurate 3 x 3 kernel | `direction` |
| `laplacian` | the second derivative: answers to any direction, and to noise | `kernel_size` |
| `log` | Laplacian of Gaussian: a blur of `sigma` first, so it answers to edges of that scale and not to noise | `sigma`, `kernel_size` |
| `dog` | Difference of Gaussians: blur `sigma` minus blur `sigma2`, the detail between the two scales (close to LoG when `sigma2` is 1.6 x `sigma`) | `sigma`, `sigma2` (greater than `sigma`) |

`direction` (Sobel and Scharr):

- `magnitude` (default): the mean of both directions' sizes, 0.5 |gx| + 0.5 |gy|.
- `x`: |gx|, changes from left to right, so vertical edges light up.
- `y`: |gy|, changes from top to bottom, so horizontal edges light up.
- `angle`: the gradient's direction, 0 to 360 degrees stored as 0 to 255. 0 is
  brighter to the right, 64 brighter downwards, 128 brighter to the left, 191
  brighter upwards. Flat areas are 0. The value wraps (255 is next to 0), so
  it suits looking at more than feeding a model.

Strengths are stored as their size, cut at 255. Sets prepared before these
settings existed keep their key and are not prepared again.

**Keep image** (`keep_image`) keeps the image and adds the edge map as one
more channel: an RGB image becomes 4 channels (R, G, B, edges), a grey one 2.
The model's first layer then reads `[H, W, 4]`. See
[More than three channels](#more-than-three-channels).

### More than three channels

An image with channels added (Image Edges with Keep image, or an Image
Gabor bank) is neither grey nor RGB. What changes:

- **Steps after it run on each channel alone**: Image Contrast, Denoise,
  Sharpen, Smooth, Warp, Pyramid and Image Filter, and the transforms that
  move pixels (crops, flips, rotate, affine, blur, morphology, noise). Each
  channel gets the same crop, flip or warp.
- **Colour steps need RGB**: Color Jitter, Grayscale, Random Grayscale and
  the Advanced Augment policies do not compile after it (`Color Jitter needs
  a 1- or 3-channel image`). Put them before the step that adds the channel.
  A second Image Edges, or an Image Gabor, is refused the same way.
- **Normalize**: ImageNet statistics are for RGB and are refused (`the images
  have 4 channels here (R, G, B, edges)`). Use `same`, or Per channel with
  one mean and std per channel; the rows are named R, G, B, edges, and Data
  Studio > Explore > Profile > Use in Normalize fills all four.
- **Pictures**: wherever such an image is shown it is a sheet: the colour
  picture, then each further channel as a grey tile with its name, each on
  its own scale. Small thumbnails (Data Studio > Prepare) show the picture
  with `+1` (the count of further tiles); hover for the whole sheet. Data Studio > Explore > Images and
  the Studio Debugger's Inspector show every tile.
- **Studio Debugger**: Where it looks draws its heat over the colour
  picture and covers or differentiates every channel. Filters and Unit
  weights are not offered for a first layer that reads more than three
  channels.

```
Data Input (images) -> Resize 72 -> Image Edges (canny, keep image)
  -> Center Crop 64 -> Normalize (same 0.5 / 0.5) -> Conv2D -> ...
```

### Image Gabor

Image Gabor runs a bank of striped kernels over every image and gives one
channel per kernel. Each kernel is a small patch of stripes under a Gaussian
window; it answers where the image has stripes of that direction and
spacing, which is what fur, grass, fabric and the edges of one direction
look like up close.

- **Orientations** spread over 180 degrees: 4 gives 0, 45, 90 and 135. The
  angle is the direction the brightness changes in, so 0 degrees answers to
  vertical stripes and 90 to horizontal ones.
- **Wavelengths** are the stripes' spacing in pixels per cycle, split by
  commas or spaces, each 2 or more and given once. Every wavelength gets the
  whole set of orientations: `3, 5` with 4 orientations is 8 channels. At
  most 32.
- **Bandwidth** (octaves) sets the window's size from the wavelength: at 1
  the window's sigma is about 0.56 wavelengths. A smaller bandwidth is a
  wider window that answers to a narrower range of spacings. **Aspect** is
  the window's width across the stripes over its length along them.
- The kernel reaches two sigmas of the window's longer side each way. At
  bandwidth 1 and aspect 0.5 that is 15 x 15 for wavelength 3, 25 x 25 for
  5 and 29 x 29 for 6. A kernel larger than the image's shorter side is
  refused with the size it needs: use a shorter wavelength, a wider
  bandwidth or a rounder aspect.
- **Response**: `energy` is the size of both phases together (the even,
  cosine kernel and the odd, sine one), so it does not depend on where
  exactly a stripe falls. `even` and `odd` keep one phase with its sign,
  stored around mid-grey (128 = no response). Every kernel has zero mean, so
  a flat area answers 0, and absolute values that sum to 1; the stored
  values are scaled so the strongest possible answer just fills the range
  and nothing is cut.
- The bank runs once, before the random steps. A later flip or rotation
  moves each channel's picture but does not swap the channels: after a
  horizontal flip the `45 deg` channel holds what a bank on the flipped
  photo would put in `135 deg`. With flips in the graph, read the diagonal
  channels as a pair.
- Colour images are read as grey. **Keep image** keeps the image's channels
  and adds the bank's after them (`[H, W, 3 + 8]`); without it the model
  sees the bank alone.
- The channels are named `gabor 45 deg 3 px`: each wavelength, its
  orientations in turn. Normalize's Per channel rows and the sheet use the
  names. See [More than three channels](#more-than-three-channels).

```
Data Input (images) -> Resize 64 -> Image Gabor (4 orientations, 3 and 5 px, keep image)
  -> Normalize (same 0.5 / 0.5) -> Conv2D -> ...
```

`examples/cyxgraph/cats_dogs_gabor_bank.cyxgraph` is a small cats-vs-dogs CNN
that reads the photo and the bank; see
`examples/cyxgraph/cats_dogs_gabor_bank.md`.

### Image Filter

Image Filter runs one kernel you choose over every image, each channel alone.
It is what a Conv2D filter does, with numbers you set instead of learned
ones, so it is a good way to see what a kernel does before a network learns
its own (compare with Studio Debugger > Model > Layers > Filters).

| `preset` | Kernel | What you get |
| --- | --- | --- |
| `identity` | `0 0 0; 0 1 0; 0 0 0` | the image unchanged |
| `box` | nine times 1/9 | a 3 x 3 average (blur) |
| `sharpen` | `0 -1 0; -1 5 -1; 0 -1 0` | stronger edges, same brightness |
| `emboss` | `-2 -1 0; -1 1 1; 0 1 2` | a relief lit from the top-left |
| `outline` | `-1 -1 -1; -1 8 -1; -1 -1 -1` | edges in every direction |
| `prewitt_x`, `prewitt_y` | `-1 0 1` in each row; `-1 -1 -1; 0 0 0; 1 1 1` | left-right and top-bottom changes |
| `roberts_x`, `roberts_y` | `1 0; 0 -1` and `0 1; -1 0` | diagonal changes, 2 x 2 |
| `kirsch` | `5 5 5; -3 0 -3; -3 -3 -3` turned to `direction` | edges brighter on that side, strong |
| `robinson` | `1 2 1; 0 0 0; -1 -2 -1` turned to `direction` | the same, gentler (Sobel weights) |
| `custom` | the `kernel` you type | anything |

- **Typing a kernel.** Rows are split by `;` (or a new line), numbers by
  spaces or commas: `1 2 1; 2 4 2; 1 2 1`. Rows must have the same length; a
  kernel may be 1 to 31 numbers each way and need not be square, but not
  larger than the image. `normalize` divides it by its sum, so `1 2 1; 2 4 2;
  1 2 1` becomes a blur that keeps the brightness; a kernel that sums to 0 is
  left as typed.
- **It is a correlation**, as in OpenCV and in Conv2D: the kernel is laid on
  the image as written, not flipped. Flip your kernel yourself for a
  textbook convolution; symmetric kernels are the same either way.
- **`direction`** (Kirsch and Robinson) is the side the edge is brighter on:
  `n` answers to a bright band above a dark one, `e` to bright on the right,
  and so on, in steps of 45 degrees.
- **`border`** is what the kernel reads past the edge: `reflect_101` mirrors
  without repeating the edge pixel (OpenCV's default), `reflect` mirrors with
  it, `replicate` repeats the edge pixel, `zero` reads black.
- **`output`.** The images are stored as 0 to 255, and an edge kernel answers
  with negative numbers too. `clip` cuts below 0 and above 255 (right for
  blurs and sharpening); `absolute` keeps the size of the answer (an edge
  map, either sign); `offset` stores it around mid-grey, 128 = no answer
  (keeps the sign; this is how emboss is usually shown). With `clip` and a
  kernel that sums to 0 the compile report says so.
- The channels stay: a colour image stays colour. Put **Grayscale** before
  the filter for one channel.

```
Data Input (images) -> Resize 64 -> Image Filter (outline, absolute)
  -> Normalize -> Conv2D -> ...
```

`examples/cyxgraph/cats_dogs_image_filter.cyxgraph` is a small cats-vs-dogs
CNN with a typed kernel in front; see
`examples/cyxgraph/cats_dogs_image_filter.md`.

## Checked against torchvision

`cyxwiz-engine/tests/test_image_transforms.cpp` replays fixtures from
`tests/computation_truth/reference/generate_image_transform_fixtures.py` on
every ArrayFire backend. For the Random Resized Crop, Random Affine, Random
Perspective, Gaussian Noise and the three policies, the generator runs
torchvision's own transform class under a fixed seed and records what it
drew (the box, the angle and shift, the corners, the noise, the ops and their
strengths, AugMix's weights); the Engine applies the same draws and must give
the same pixels within 2e-5. Its own draws are checked for their ranges and
frequencies.

## Notes

- To leave blurry, dark, flat or duplicate images out of training, see
  [Quality Analyzer](quality_analyzer.md).
- Code export (PyTorch / TensorFlow / Keras / PyCyxWiz) does not include image
  transforms yet, the OpenCV steps included, and says so; train these graphs
  in Studio.
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
- The OpenCV steps are checked against OpenCV called directly
  (`image_cv_steps`: CLAHE, non-local means, Canny, Sobel, median, warps and
  pyramids on 8-bit images, exact or within one step), in a prepared set
  between GPU steps (`test_prepared_image_set`), in What the model sees
  (`image_pipeline_preview`) and in the compiler (`image_transform_graph_contract`).
