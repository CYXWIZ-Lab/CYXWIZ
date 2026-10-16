# Node dialogs

Double-click a node (or right-click it and choose **Configure...**; Data Studio
has **Open dialog**) to open its dialog.
The node's settings are saved into the graph when you press **Apply** or
**OK**.

## The buttons every dialog shares

- **Cancel** puts back the settings the node had when the dialog opened and
  closes it. It is always available, also while the dialog is loading.
- **Apply** saves the settings and keeps the dialog open.
- **OK** saves and closes.

While a dialog is loading or running something (a Data Input load, a Data
Convert run), Apply and OK are off; hover them to see why. Dialogs that know
when you changed something (Embedding, Word Embeddings) keep Apply off until
you do ("Nothing changed").

A file field is a path box with **Browse...** beside it; typing a path works
as well as browsing.

## Data Input

Data Input reads a file or a folder: a table (CSV, TSV, JSON Lines, Parquet,
Feather, Arrow / IPC, a ZIP of text documents; HDF5 files can be inspected),
text, a token-tagging table, a time series, a folder of images or a folder of
audio. Databases come in through Data Studio's SQL Workbench, not through Data
Input.

### Table format options

- **Quote character** (default `"`): a value inside it may hold the
  delimiter and line breaks; a doubled quote inside is one quote. Leave it
  empty to read quotes as ordinary text. Preview, Apply and pipeline runs use
  the same character.
- Files are read as UTF-8; a byte order mark at the start is skipped.
- **Limit Rows** skips the first rows and caps how many are loaded. To keep
  only some rows, filter them in Data Studio's Prepare step or with a Filter
  or SQL node.
