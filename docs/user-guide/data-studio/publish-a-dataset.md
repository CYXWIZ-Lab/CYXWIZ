---
title: Write the prepared dataset
description: Write the prepared data to a file with Data Studio's Publish job, and read the lineage file the Engine writes beside every export.
type: how-to
area: data-studio
audience: user
status: stable
since: 1.0.7
owner: data-studio
last_reviewed: 2026-11-25
related:
  - prepare.md
  - validate.md
  - overview.md
---

The **Publish** job of Data Studio writes the prepared data to a file in
your project (for example `datasets/published/<step>.parquet`), so other
graphs, scripts and tools can use it. Every write also leaves a lineage file
beside it that says which recipe, inputs and checks made the file. The file
stays on your computer; nothing is uploaded or shared.

## See the graph's outputs

**Publish** lists the graph's outputs (Export CSV, Export JSON, Export
Parquet, Data Output nodes) in one table:

| Column | What it shows |
| --- | --- |
| **output (node)** | The export node |
| **reads** | The step it writes, past any checks in front of it |
| **writes** | The file, relative to the project when inside it (the full path on hover) |
| **last written** | *15,349 rows - 15:19:58*, *not written yet*, or *failed* with the error |

Click a row to show its lineage; **Show** selects the node on the canvas.

## Write a dataset file

1. In **Data Studio**, click **Publish...** (in the Publish job or the
   header). The *Publish dataset* dialog opens.
2. In **Output**, pick one of the graph's outputs, or **New Parquet
   export**, which adds an Export Parquet node reading the step in
   **Reads** (for a step inside a recipe, the recipe's result).
3. Check **Name** (the node's name) and **Location** (the file, relative to
   the project unless it is a full path; suggested:
   `datasets/published/<step>.parquet`).
4. Click **Publish**. The Engine runs the output, writes the file and its
   lineage, and updates the table and the lineage card when the run ends.
   Ctrl+Z on the canvas undoes the node change.

To add a Data Output instead, click **Add Data Output**: it reads the step
chosen in the rail (or the last step), and you set its format and file in
its dialog.

## Read the lineage

Every time an output writes its file, the run also writes
`<file>.lineage.json` beside it, whether the run came from Publish, **Run
step** or **Execute Pipeline**. Publish shows the selected output's lineage:

| Row | What it says |
| --- | --- |
| **Made by** | The recipe and its revision (the first 12 hex digits of the SHA-256 of its steps' types and settings, so any change to a step changes it) and the number of steps |
| **Inputs** | Each Data Input's file with its size and SHA-256 |
| **Result** | Rows, columns, the file's size and SHA-256 (hashed while the file is written, so the export is not read back) |
| **Checks** | *4 of 4 pass*, or the first one that did not |
| **Written** | When, and from which graph file |

**Steps** opens the list of steps before the output. The lineage file also
holds each step's settings and outcome, the schema and the full digests.

How inputs are hashed:

- A file is hashed once; the digest is kept by path, size and write time in
  the project's `cache/ingestion/digests`.
- A CSV over the data budget's result threshold is hashed while it is
  converted, so it is not read again.
- A larger file that is never read in full (a Parquet file opened in place)
  shows *not hashed (large file; size and time identify it)*, and the
  lineage file holds its write time.

## Related

- [Prepare data with a recipe](prepare.md)
- [Check a dataset before training](validate.md)
- [Data bigger than memory](../../concepts/data-bigger-than-memory.md)
