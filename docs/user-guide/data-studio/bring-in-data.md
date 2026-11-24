---
title: Bring data into Data Studio
description: Add a Data Input node from Data Studio's Bring in job, see what each one loaded, and read an image dataset's index table.
type: how-to
area: data-studio
audience: user
status: stable
since: 1.0.7
owner: data-studio
last_reviewed: 2026-11-25
related:
  - overview.md
  - explore.md
  - ../../main-window/data-input-dialog.md
---

Data reaches a graph only through a **Data Input** node. The **Bring in**
job of Data Studio adds those nodes and lists what each one loaded.

## Add a Data Input

1. In **Data Studio**, click the **Bring in** job.
2. Click **Add Data Input...**. The Engine adds a Data Input node to the
   graph and opens its dialog.
3. In the dialog, pick the file or folder, its format and the preview, then
   click **Apply**. See [The Data Input dialog](../../main-window/data-input-dialog.md).

The node is now listed in the job's table and in the rail's **Data
Inputs**.

## Read the Data Inputs table

The table lists the graph's Data Input nodes:

| Column | What it shows |
| --- | --- |
| Node | The Data Input's name |
| File | Its file as written in the graph |
| Size | Rows x columns, once loaded |
| State | *loaded*, *not loaded*, or *failed* with the error on hover |
| Actions | **Open dialog** opens the node's dialog on the canvas; **Explore** opens a loaded one in Explore > Profile |

Where the five columns do not fit, each Data Input is listed as a block with
the same facts and links.

An image Data Input shows its folder, where its labels come from (*image
folder, labels from cat_dog.csv* or *image folder, class folders*) and the
image count (*25,000 images*).

## Image types the Engine reads

JPEG (`.jpg` `.jpeg` `.jpe`), PNG, BMP (`.bmp` `.dib`), TIFF (`.tif`
`.tiff`), WebP, GIF (its first frame), the Netpbm types (`.ppm` `.pgm`
`.pbm` `.pnm`), Sun raster (`.sr` `.ras`), Radiance HDR (`.hdr` `.pic`) and
PFM.

The same list is used everywhere: the Data Input's folder scan, the image
index, training, Explore > Images and the asset previews. Other files in an
image folder are skipped. TGA, AVIF, HEIC, JPEG 2000, JPEG XL and EXR are
not read: convert them first.

## The image index

Loading an image dataset also builds its **index table**, one row per image,
named after the dataset with `_index` (`cat_dog_index`). It reads only each
file's header (JPEG, PNG and 16/24-bit BMP; other formats are decoded once),
on a background task, so the Data Input is loaded before the index is ready.
The 25,000 cats vs dogs images take under a second when the files are in the
system's file cache and about 30 seconds from a cold disk. The rail lists the
index under its Data Input (*indexing* while it builds).

| Column | What |
| --- | --- |
| `path` | The image file |
| `file` | Its file name |
| `label` | The class index (classes in sorted order, numbers as numbers) |
| `label_name` | The class name (the Data Input's class names when set) |
| `width`, `height` | The original size in pixels (empty when unreadable) |
| `channels` | What a decode gives: 1 grey, 3 colour, 4 with alpha |
| `bytes` | The file size |
| `readable` | `false` when the file cannot be read as an image |

Once quality is measured ([Profile > Quality](profile.md#an-image-datasets-profile)),
the index has five more columns: `blur`, `brightness`, `contrast`, `verdict`
(empty when the image passes, else its reasons, *blurry, low contrast*) and
`duplicate_of` (the file it repeats). They stay when the index is rebuilt
(renamed classes).

The index is an ordinary table: Query, the Explore views and the Table
Viewer all work on it.

```sql
SELECT label_name, COUNT(*) FROM "cat_dog_index" GROUP BY label_name
```

It is dropped when the images are unloaded and rebuilt when they are loaded
again.

Changed in 1.0.8: the image index, its quality columns and the image types
list shared by every reader.

## Related

- [The Data Studio window](overview.md)
- [Explore a dataset](explore.md)
- [Look at an image dataset](explore-images.md)
- [Split images and audio](split-images-and-audio.md)
