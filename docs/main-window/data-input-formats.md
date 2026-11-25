---
title: Data Input formats
description: How Data Input reads tables, plugin formats, HDF5 files, text, images and audio, and which node each kind of data needs next.
type: reference
area: graph-editor
audience: user
status: stable
since: 1.0.7
owner: graph-editor
last_reviewed: 2026-11-25
related:
  - data-input-dialog.md
  - node-dialogs.md
  - ../user-guide/graph-editor/compile-and-run.md
redirect_from: []
---

What the [Data Input dialog](data-input-dialog.md) does with each kind of
source, and the node each kind needs after it before it reaches a model.

## Table format options

The **Format options** under **Detected** on the Source tab:

- **Quote character** (default `"`): a value inside it may hold the
  delimiter and line breaks; a doubled quote inside is one quote. Leave it
  empty to read quotes as ordinary text. Preview, Apply and pipeline runs
  use the same character.
- Files are read as UTF-8; a byte order mark at the start is skipped.
- The delimiter, header row, decimal separator and missing-value markers
  are set beside it.

## Formats from plugins

A plugin can add a table format (for example ARFF, with the example plugin
*Example Data Formats*). While the plugin is loaded, its formats are in the
**Format** list as *&lt;name&gt; (plugin)*, in **Browse...**'s file types,
and picked by the file's extension, so a file of such a format behaves like
any table: Preview after Apply, Data Studio, SQL, training. A format the
Engine reads itself wins over a plugin's for the same extension; choose the
plugin's in **Format** to use it instead.

On Apply the Engine reads the file through the plugin once and keeps a
Parquet copy in the project's cache; it reads that copy again until the
file or the plugin's version changes. Large files stay on disk like a large
Parquet file. A graph that uses such a format needs the plugin: without it,
Apply says *This format comes from a plugin that is not loaded*. Remote
training receives the loaded table, so the remote computer does not need
the plugin.

## HDF5 files

An HDF5 file (`.h5`, `.hdf5`, `.hdf`) holds groups and datasets; Data Input
loads one numeric dataset as a table.

1. The Source tab shows the file's hierarchy (name, shape, type). Open a
   group to go into it; the arrow goes back up.
2. With **Data** chosen above it, click the dataset to load. With
   **Labels**, click the dataset of class labels (optional). The paths are
   also in **Data path** and **Label path (optional)**, to type.
3. **Preview** shows the selected rows, a page at a time. Apply stays
   unavailable until a dataset is chosen ("Choose the dataset to load in
   the HDF5 hierarchy").

What loads:

- The data is a rank-1 dataset (one column, named `value`) or a rank-2 one
  (rows x columns, named `col_0`, `col_1`, ...), of integers or float32 /
  float64. The labels are a rank-1 dataset with one value per row; they
  become the last column, `label`, and the node's label column.
- Apply reads the dataset into memory, at most 256 MB; a larger one is
  refused with the estimate. Training and graph runs read the same dataset
  from the node's settings, so a reopened project needs no new Apply.
- Not loaded (refused with the reason): rank 3 and up (image stacks such
  as N x H x W x C), strings and compound types, virtual or external
  storage, and labels whose row count differs from the data's.

Example: `examples/datasets/metric_blobs.h5` holds the `metric_blobs.csv`
table, the features in `/blobs/features` (480 x 8, float32) and the class
in `/blobs/labels`; `make_metric_blobs_h5.py` beside it writes it with
h5py. The template `examples/cyxgraph/blobs_hdf5_classifier.cyxgraph`
trains a classifier from it.

## Text needs a node that turns it into numbers

Data Input loads text and its label; a node after it turns the text into
numbers: a **Text Tokenizer** (word ids), a **TF-IDF** or **Count**
vectorizer, or **Word Embeddings**. Compile refuses a text input that
reaches the model without one (*CW-C-0203*, "gives text, and nothing turns
it into numbers") and offers **Add a Text Tokenizer**: it adds the node
after the Data Input, takes over its links, fills in the text and label
columns and opens its dialog.

Data Input's Text options set only the layout and the text and label
columns. Under **Preprocessing (done by nodes)** the dialog names the nodes
after it that turn the text into numbers (*Tokenized by Tokenizer (word),
vocabulary sentiment_analysis_vocab.txt, 64 ids per row*), with **Open it**
and **Show on canvas**, or offers **Add a Text Tokenizer**. A new Text
Tokenizer takes its text and label columns from the Data Input before it.
Apply reads the texts (an empty text is not a sample) and builds no
vocabulary.

## Images need a Resize, audio a feature node

Data Input only loads images and audio:

- A **Resize** after the Data Loader sets the size every image is decoded
  at. Compile refuses an image graph without one (*CW-C-0204*) and offers
  **Add a Resize**.
- Audio features come from a **Mel Spectrogram**, **Spectrogram** or
  **MFCC** node after the Data Loader. Compile refuses an audio graph
  without one (*CW-C-0205*) and offers the three.

The added node goes after the Data Loader's data output (the labels keep
their link) and opens its settings.

Data Input's Image options set only the layout, the folder and (flat
layout) the labels CSV; its Audio options the layout, folder, labels CSV,
sample rate and length ("seconds, padded or cut"). Audio is always mixed to
mono. Under **Preprocessing (done by nodes)** the dialog names the Resize
(*Resized by Resize 64x64, 64 x 64*) and Normalize nodes, or the audio
feature node, with **Show on canvas** (and **Open it** for nodes with a
dialog). When one is missing it offers **Add a Resize**, **Add Normalize**
(after the Resize) or **Add Mel Spectrogram / Spectrogram / MFCC**.

An image folder holds nothing in memory: when training starts, the images
are decoded once at the Resize size into `cache/images` (see
[Image transforms](../reference/nodes/categories/preprocessing.md#image-transforms)). The image types the
Engine reads are listed in
[Bring data into Data Studio](../user-guide/data-studio/bring-in-data.md).

## Related

- [Data Input dialog](data-input-dialog.md)
- [Compile and run a graph](../user-guide/graph-editor/compile-and-run.md)
- [Text Tokenizer and Embedding dialogs](text-node-dialogs.md)
