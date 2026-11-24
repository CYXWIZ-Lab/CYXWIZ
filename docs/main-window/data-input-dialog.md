---
title: Data Input dialog
description: The Data Input node's dialog, tab by tab: the source and how its rows are used, the data strip, Preview, Profile, Memory and Audit.
type: reference
area: graph-editor
audience: user
status: stable
since: 1.0.7
owner: graph-editor
last_reviewed: 2026-11-25
related:
  - node-dialogs.md
  - data-input-formats.md
  - ../user-guide/data-studio/bring-in-data.md
  - ../concepts/loaded-data.md
redirect_from: []
---

**Data Input** is the one way data enters a graph. It reads a file or a
folder: a table (CSV, TSV, JSON Lines, Parquet, Feather, Arrow / IPC, a ZIP
of text documents, a dataset in an HDF5 file), text, a token-tagging table,
a time series, a folder of images or a folder of audio. Databases come in
through Data Studio's [SQL Workbench](../user-guide/data-studio/sql-workbench.md),
not through Data Input. Open the dialog by double-clicking the node; the
buttons every dialog shares are in [Node dialogs](node-dialogs.md).

How each kind of source is read (table options, plugin formats, HDF5,
text, images and audio) is in [Data Input formats](data-input-formats.md).

## The tabs

| Tab | What it holds |
| --- | --- |
| **Source** | The file or folder and how its rows are used (next section) |
| **Preview** | The first rows; for an image folder, its first 24 images with their class once it is loaded (Data Studio's **Explore > Images** shows them all) |
| **Profile** | The loaded data's profile; for an image folder, the image profile (counts, classes, sizes, prepared sets; see [Profile](../user-guide/data-studio/profile.md)) |
| **Rows and columns** | Which columns load, the rows to skip and the row limit. Not shown for an image or audio folder, which has no rows to read. |
| **Memory** | How the data is held, and unloading it |
| **Audit** | A check of the loaded dataset |

## Source: the file or folder, then how its rows are used

**1. Source** has one field, **File or folder**, with **Browse file...**
and **Browse folder...**. Under it, **Detected** says what it is:

- a file: its format and columns (*CSV table, 30.0 MB, 3 columns:
  (unnamed), statement, status*), with **Format options** (format,
  delimiter, header row, decimal separator, quote character, missing
  values; a ZIP's member list). Change the format only when the extension
  is wrong.
- a folder: what its files are (*Image folder: flat folder, 25,000
  images*, or *2 class folders (cat, dog)*), with **Scan again**.

**2. Use the rows as** offers only the uses that fit the source:

| Use | For | Columns |
| --- | --- | --- |
| **Table** | A table file | A label column; every other column is an input feature |
| **Text** | A table file or a folder of text files | A text column and a label column |
| **Token tagging** | A table file | Token, tag, POS and sentence columns |
| **Time series** | A table file | A label column and a **Time column** |
| Images | A folder of images | **Labels from**: class folders, or a flat folder and a labels CSV |
| Audio | A folder of audio | Labels, sample rate, length |

For a time series, the rows are put in the **Time column**'s order on
load (missing times last), so windows and a chronological split follow
time. A column named date, time, timestamp, datetime or ds is offered with
**Use it**; with none, the rows keep the file's order. The strip then
reads *as sales (time series by date)*. A table kept on disk cannot be put
in order, and the load says so.

Training, validation and test roles come from the graph's Data Split, not
from this node.

### Class names of an image folder

Once an image folder is loaded, its Source tab has **Class names**: a row
per class (its label in the CSV, or its folder) with a name to type and a
**suggested** one, the file-name prefix every file of the class shares
(*cat* for `cat.1.jpg`, *dog* for `dog.1.jpg`) when each class has its own.
**Use suggested** fills them in. Apply saves them with the node (an empty
name keeps the label), and every screen and the index's `label_name` use
them at once, without loading the images again. Class indices do not
change: they follow the labels, sorted (numbers as numbers).

## The data strip

Above the tabs a strip says what the node loads and how far it got, on
every tab:

- the file or folder;
- its state: *Not loaded*, *Loading: ...* with the seconds so far, *Loaded
  in 1.79 s as sentiment_mental_health (text)*, or a red *Not loaded* with
  the reason and what to do;
- the counts (*52,681 samples, 7 classes, 30.0 MB on disk*);
- how the data is held: *30.4 MB in memory*, *kept on disk (Parquet
  ...)*, or *decoded once at the Resize size into cache/images when
  training starts*.

When an Apply fails while an earlier load stays, the strip keeps the
earlier load and says the last Apply failed.

## Rows and columns

- **Include all columns**, or pick the columns to load.
- **Skip first N source rows:**.
- **Limit dataset rows**: Apply loads at most that many rows; only those
  are split and trained on.

To keep only some rows by value, filter them in Data Studio's Prepare job
or with a [Row Filter](row-filter-dialog.md) or SQL Query node.

## Memory and Audit

- **Memory** shows **Current Status:** (*In Memory*, *On disk (Parquet)*,
  *Lazy-loaded*, *On Disk (streaming)* or *Not Loaded*), the rows and
  columns, and **Unload from memory**. Under **Advanced**, **Force
  disk-backed cache** keeps even a small table on disk instead of in
  memory. See [Loaded data](../concepts/loaded-data.md) and
  [Data bigger than memory](../concepts/data-bigger-than-memory.md).
- **Audit** (**Dataset audit**) checks the loaded dataset: *Failed*,
  *Warnings* or *Passed*, with the dataset name, rows, columns, what read
  it, and a table of errors and warnings. Before a load it says "Click
  Apply to load data and run the audit."

## Related

- [Data Input formats](data-input-formats.md)
- [Node dialogs](node-dialogs.md)
- [Bring data into Data Studio](../user-guide/data-studio/bring-in-data.md)
- [Loaded data](../concepts/loaded-data.md)
