---
title: Prepare images with the image recipe
description: See, set, add, remove and reorder an image graph's steps in Data Studio's Prepare job, on a sample image.
type: how-to
area: data-studio
audience: user
status: experimental
since: 1.0.8
owner: data-studio
last_reviewed: 2026-11-25
related:
  - prepare.md
  - explore-images.md
  - profile.md
  - ../../reference/nodes/categories/preprocessing.md
---

A graph whose images go through a **Resize** shows its **Image recipe** in
Prepare instead of the table steps: the graph's image steps in the order
training runs them, each row being the node itself. Open it with the
**Prepare** job of Data Studio.

The subtitle names the data, the model input and the counts (*cat_dog to
model input [64, 64, 3] · 2 steps once, 2 per batch*).

## The four groups

The rows come in four groups, the same split training makes:

| Group | Steps |
| --- | --- |
| **First** | The Resize |
| **Once, before training** | Steps done the same way to every image of every part when the images are prepared (kept in `cache/images`): Center Crop, Grayscale, Blur, Morphology and the OpenCV steps (CLAHE, Denoise, Sharpen, Edges, Smooth, Warp, Pyramid) |
| **Per batch, training images only** | Everything from the first random step on, drawn again for every batch on the GPU; validation and test images skip them. A Blur or Grayscale after a random step runs here too. |
| **Last** | The Normalize, on the GPU with each batch |

## Read a row

Each row shows the step's picture on a sample image (the image after that
step; a per-batch row shows several draws, and Normalize shows the picture
the model gets), its settings in short and the shape after it. **Next
sample** takes the next image of the index. The first pictures of a step
take a moment while its GPU code is built (*Drawing the pictures of ...*).

## Change, add and remove steps

- **Settings** opens the node's settings in the row, the same typed fields
  as Properties; a change applies to the node.
- **Show on canvas**, **Open dialog** and **Remove** (once-only and
  per-batch rows) act on the node; Ctrl+Z on the canvas undoes a removal.
- **+ Add a step** offers the steps done once; **+ Add an augmentation**
  offers the random ones (Random Crop, Random Resized Crop, Horizontal /
  Vertical Flip, Rotate, Random Affine, Random Perspective, Color Jitter,
  Random Grayscale, Gaussian Noise, Advanced Augment, Augmentation Preset)
  and, under **The same for every image**, the device steps that are not
  random. The new node goes after the group's last row.

## Reorder image steps

Drag a row by its handle. Over the upper half of a row the step goes before
that row, over the lower half after it, and a line on that side shows where
it lands (no line: the order would not change). The node moves on the
canvas with its links.

Resize stays first and Normalize last, and an OpenCV step cannot go after a
random one (*Image Contrast (CLAHE) is done once with the images: it must
come before the first random step.*). A refused move shows why while you
drag.

## Related

- [Prepare data with a recipe](prepare.md)
- [Look at an image dataset](explore-images.md)
- [Profile a dataset](profile.md)
- [Image transforms](../../reference/nodes/categories/preprocessing.md#image-transforms)
