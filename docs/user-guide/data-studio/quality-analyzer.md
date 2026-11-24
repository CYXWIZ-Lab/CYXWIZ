---
title: Leave poor images out with the Quality Analyzer
description: Measure blur, brightness, contrast and near-duplicates of an image dataset and leave the images that fail out of training.
type: how-to
area: data-studio
audience: user
status: stable
since: 1.0.7
owner: data-studio
last_reviewed: 2026-11-25
related:
  - split-images-and-audio.md
  - explore-images.md
redirect_from:
  - ../../usage/quality_analyzer.md
---

The **Quality Analyzer** node leaves poor images out of training: blurry
ones, ones that are too dark or too bright, flat low-contrast ones, and
near-duplicates. It is a filter on the dataset: the files stay on disk, and
training skips them.

```text
Data Input (images) -> Quality Analyzer -> Data Split -> Data Loader
  -> Resize 64 -> ... -> Conv2D -> ...
```

Put it before the **Data Split**: it decides which images exist for
training, and the Split then shares out only the images that pass, so a
rejected image never lands in validation or test. Its pins carry the
dataset, like the Data Input's. A graph has one Quality Analyzer. It works on
image Data Inputs (folders of image files, with class subfolders or a labels
CSV).

## Before you start

- An image Data Input, loaded (its dialog's **Apply**).
- A **Resize** node in the graph: the images are measured at the size the
  model sees.

## Steps

1. On the canvas, add a **Quality Analyzer** between the Data Input and the
   Data Split.
2. Double-click the Quality Analyzer (or select **Open Dialog...** in
   Properties).
3. Select **Analyze**. Every image is decoded at the Resize size and
   measured once on the GPU. The work also shows in **Background Tasks**, so
   you can close the dialog and keep working; **Stop** ends it.
4. Read the result: how many images pass, a row per reason with the worst
   examples, a histogram per measurement with the cut marked, and a warning
   when one class loses a clearly larger share than another.
5. Optional: change a check. The verdict updates at once, because the
   measurements are kept. Only added or changed images, or a new Resize
   size, need **Analyze again**.
6. Select **OK** to save the checks. Training then leaves the rejected files
   out of every split.

## The checks

| Check | Measured as | Default |
| --- | --- | --- |
| Blur | Variance of the 3x3 Laplacian of the luminance; low = blurry | reject below 700 |
| Brightness | Mean luminance, 0 to 255 | reject below 50 or above 220 |
| Contrast | Standard deviation of the luminance / 255 | reject below 0.12 |
| Near-duplicates | 64-bit difference hash; images within 4 bits match | on; the first of each group stays |

The luminance is `0.299 R + 0.587 G + 0.114 B` (OpenCV's RGB to grey). The
measurements match OpenCV: `cv2.Laplacian(ksize=1)` with its default
reflect-101 border, and `cv2.resize(INTER_AREA)` to 9 x 8 for the hash.

The numbers depend on the image size. The blur threshold of 100 often quoted
for OpenCV is meant for full-size photos; at 64 x 64 almost every image is
far above it. The defaults come from measuring a real set at the Resize
size (see the example). Look at the blur histogram and the examples before
you move them.

An image that cannot be decoded measures as black and is left out as too
dark.

## Example: cats and dogs

300 images (150 cats, 150 dogs) at Resize 64 x 64, default checks:

- Measured in about 0.5 s on a GeForce GTX 1050 Ti (CUDA).
- 22 of 300 left out: 14 blurry, 9 low contrast, 2 too dark, 1 too bright;
  4 fail more than one check. No near-duplicates.
- Cats lose 16, dogs 6, so the dialog warns that rejections lean to one
  class.
- Blur at 64 x 64: 5th percentile 713, median 1778. Brightness: median 119.
  Contrast: median 0.211.

## Where the analysis is kept

In `<project>/cache/quality/` when a project is open, otherwise in
`<temp>/cyxwiz/quality/`. Each file is named after a key made from every
image's path, size and modification time and the Resize size, so a changed
dataset never reuses an old analysis. Deleting the folder only means
analyzing again.

## Limits

- Measured on the GPU with ArrayFire; without a GPU the same code runs on
  ArrayFire's CPU backend.
- Code export does not include the filter, and says so; train these graphs
  in CyxWiz Studio.
- The measurements are checked against OpenCV on CUDA, OpenCL and the
  ArrayFire CPU backend.

## If something goes wrong

Training refuses to start when the node has no analysis for the current
images at the current Resize size, and says so on the Quality Analyzer:

- **the images are not analyzed yet**: open the dialog and select
  **Analyze**.
- **the images or the Resize size changed since the analysis**: select
  **Analyze again**.

Training also refuses checks that leave out every image: loosen a check.

## Related

- [Split images and audio](split-images-and-audio.md)
- [Look at images in Explore](explore-images.md)
- [Explore your data](explore.md)
