---
title: Look at an image dataset
description: Browse an image dataset in Explore > Images as a grid, a list or one image at a time, and see what the model gets after the graph's image steps.
type: how-to
area: data-studio
audience: user
status: experimental
since: 1.0.8
owner: data-studio
last_reviewed: 2026-11-25
related:
  - explore.md
  - profile.md
  - bring-in-data.md
  - quality-analyzer.md
---

**Explore > Images** in Data Studio shows an image dataset's pictures as a
**Grid**, a **List** or one at a time in the **Viewer** (the first buttons
of its toolbar). The view appears when you pick an image dataset (its Data
Input in the rail). It works for any image dataset: the facts come from the
dataset's [index table](bring-in-data.md#the-image-index).

## Filter and page

- The chips filter by class (*All*, *Class 0*, *Class 1*, or the class
  names) and show the unreadable files when there are any. Once quality is
  measured ([Profile](profile.md#measure-image-quality)), there are also
  *Fails a check* and one chip per reason (*blurry*, *low contrast*,
  *dark*, *bright*, *duplicate*).
- The filter holds in all three views.
- The arrows at the right turn the pages (*1 to 60 of 25,000*).

## Grid

The tiles fill the panel and scroll; hovering shows the file name.

| Control | What it does |
| --- | --- |
| **Tile** | The tile size, 64 to 256 px (typed or with - / +) |
| **Fill** / **Fit** | Crop each image to its square, or show it whole |
| **Show** | What a tile carries: its **Class** (bottom left), **Name** (bottom right), **Size** in pixels (top left) and **Flags**, the first failed check (top right, *blurry +1*) |
| **Sort** | Dataset order, smallest first, largest first (by pixel count, unreadable files last) or, once measured, blurriest first |

Click a tile to pick it (a ring and a wash). The panel at the right shows it
larger with its row, label, original size and colour, size on disk and path
(and, when measured, its blur, brightness, contrast, verdict and the image it
duplicates), **Open in viewer**, and the links **Show in folder** and **Open
the index table** (Query on `<dataset>_index`).

The arrow keys move the pick (Up and Down a row); Enter or a double-click
opens it in the viewer. Thumbnails are read on two background threads, the
ones on screen first.

## List

A row per image: thumbnail, file, class, size, channels, size on disk and
quality (*not measured* until it is). Click a column to sort by it (File by
name, Class, Size smallest or largest first, On disk largest first, Quality
blurriest first); a third click goes back to dataset order. Double-click a
row to open it in the viewer.

## Viewer

One image at full size; nothing here changes the file.

| Control | What it does |
| --- | --- |
| **Zoom** | **Fit** (the whole image, enlarging a small one), **100 %**, a typed percent, and - / + through steps from 6 % to 1600 %. The mouse wheel zooms around the cursor, a drag pans, a double-click switches between Fit and 100 %. |
| **Channel** | RGB, R, G or B alone, Luminance (as the Grayscale node: 0.2989 R + 0.587 G + 0.114 B), and Alpha when the image has one |
| **View** | Rotate left or right, **Flip** left to right, **Pixel grid** (lines between pixels from 800 %), **Checkerboard** behind transparent pixels |
| **Compare** | **Original**, **Model sees** (the image after this graph's fixed image steps, at the size the model gets, *Gaussian Blur 3 x 3, sigma 0.8 · 546 %*) or **Side by side**, each fitted to its half |

- Under the picture: the filmstrip (the images before and after it in the
  current order; click one to go to it) and the status bar, with the pixel
  under the cursor (*x 189 y 190 · R 151 G 150 B 146 · #979692*) and the
  image's size, colour type, bits per channel and zoom.
- At the right: a **Histogram** per channel (red, green, blue, luminance,
  and alpha when there is one; 32 bins of 0 to 255, with the mean and
  standard deviation; the chosen channel stays bright), **Around the
  cursor** (the 12 x 8 pixels around the pointed one, one cell each), and
  the image's details.
- The arrows by the file name (or Left and Right) go to the previous and
  next image; Esc goes back to the grid or list.

The images are read with OpenCV, as training reads them: the pixels as
stored, without turning a photo by its EXIF orientation (a phone photo taken
sideways shows sideways, as the model gets it). 16-bit and float images are
scaled to 8 bits for the screen; the status bar names the file's own depth.

## See what the model sees

**What the model sees**, folded under the grid (click it to open), runs the
picked image through this graph's image steps with the code training runs,
on the ArrayFire device:

1. the Resize (*Resize 72 x 72, exact, once*);
2. each transform done once with the images (*Center Crop 64 x 64*,
   *Gaussian Blur 3 x 3, sigma 0.8*);
3. four draws of the random transforms done per batch, each saying what it
   drew (*flipped, rotated 7.5 degrees*).

A graph without random transforms says every epoch sees the same images.
Normalize is left out so the pictures stay viewable. The section runs again
when the graph's image steps change. To change the steps, see
[Prepare images with the image recipe](image-recipe.md).

## Related

- [Profile a dataset](profile.md)
- [Bring data into Data Studio](bring-in-data.md)
- [Quality Analyzer](quality-analyzer.md)
- [Image transforms](../../reference/nodes/categories/preprocessing.md#image-transforms)
