---
title: Data Output and Data Convert dialogs
description: The dialogs that write a table to a CSV or Parquet file from the graph, and convert a table file to another format once.
type: reference
area: graph-editor
audience: user
status: stable
since: 1.0.7
owner: graph-editor
last_reviewed: 2026-11-25
related:
  - node-dialogs.md
  - ../user-guide/data-studio/publish-a-dataset.md
redirect_from: []
---

**Data Output** writes the table that reaches it to a file each time the
graph runs. **Data Convert** converts a table file to another format once.
The buttons every dialog shares are in [Node dialogs](node-dialogs.md).

## Data Output

| Field | What it sets |
| --- | --- |
| **File** | The file the step writes, with **Browse...**. Left empty, the step writes to the project's exports folder, named after the table it reads. |
| **Format** | CSV or Parquet; switching it also switches the file's extension |
| **Relative to** | Shown for a relative path: the project folder (like Data Input paths) or the exports folder |
| **Writes to** | The full path the run will write |
| **Last run** | After a run: the file it wrote and when, or why it failed |

## Data Convert

Converts a table file to another format once, with an optional sidecar
manifest. Six tabs:

| Tab | What it holds |
| --- | --- |
| **Source** | The input file with **Browse...** and its format (*Auto* reads it from the extension). An XLSX input adds the worksheet (blank is the first) and the start column; Plain Text gives one column named `text`. |
| **Output** | The output file with **Save as...**, **Use the input's name with this format's extension**, the format (picking one switches the extension; a mismatch is flagged), and **When writing**: overwrite an existing file, create missing folders, write a conversion manifest beside it |
| **Options** | *Reading the input* (delimiter, detected or typed; decimal separator; header; line breaks inside quoted values; skip rows) and *Writing Parquet* (compression, row group size). Settings that do not apply to the chosen formats are dimmed with the reason. |
| **Preview** | **Preview schema** reads the input in the background and shows its rows, columns, each column's type and whether it can be missing, and sample rows |
| **Run** | **Run conversion** (the tab's main button, with its progress), then the output path, rows written, columns, size and manifest. The write finishes even if the dialog closes; the footer's Apply saves the settings. |
| **Log** | The messages of this dialog session, with **Clear** |

## Related

- [Node dialogs](node-dialogs.md)
- [Publish a dataset](../user-guide/data-studio/publish-a-dataset.md)
