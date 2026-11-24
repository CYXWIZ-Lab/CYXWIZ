---
title: Plot data on the canvas
description: Add a Plot node at any point of a graph to see the data there, and what its status lines and Properties mean.
type: how-to
area: evaluation
audience: user
status: stable
since: 1.0.7
owner: evaluation
last_reviewed: 2026-11-25
related:
  - plotting.md
  - model-result-plots.md
  - ../../reference/nodes/categories/visualization.md
redirect_from: []
---

A **Plot** node draws the data at its place in the graph: after a Data
Input, a filter, a transform, a Preparation Recipe or an evaluation node.
It opens the same Plot window as a table does ([Plot data](plotting.md)).
The node is not a pipeline step: training skips it.

## Add a Plot node

1. On the CyxWiz Studio canvas, search **plot** (or open the
   **Visualization** group of the Nodes panel) and add a **Plot** node.
2. Connect the table you want to look at to its **Data** input:

   ```text
   Data Input (mnist_784.csv) --> Plot
   ```

3. Double-click the node (or right-click it and choose **Configure...**,
   or click **Open Plot window** in Properties). The Plot window opens with
   a header: *Data at &lt;node&gt;*, rows × columns, the read time, and
   **Refresh** (**Cancel** while it reads).
4. The first plot is picked for you: the counts of a label column (class,
   label, target or y), otherwise of a text column, otherwise a histogram
   of the first number column. Change the type, columns and labels as you
   like; they are saved in the node.

## What the node reads

- A loaded Data Input is read as it is.
- A node after it (a filter, a transform) is run on its own, with only the
  nodes above it; progress and **Cancel** are in **Background Tasks**.
- A Preparation Recipe runs as its steps, as the Data Pipeline runs it: the
  plot reads the recipe's result, and the header names its last step
  (*Data at Check: nothing admitted*).
- A table kept on disk is plotted from a sample of its row groups; the
  node says *N of M rows (sample)*.
- The data itself is never saved in the graph; it is read again when the
  plot opens.

## The status lines under the node

| First line | Meaning |
| --- | --- |
| *Not connected* | Connect a table to plot it. |
| *Not available yet* | The node above cannot be plotted; the second line says why and names its input to plot instead. |
| *Not read yet* | Open the plot to read the data. |
| *Reading N%* | The nodes above are running. |
| The plot and its column ("Bar · class") | The rows and when they were read. |
| *Out of date* | A node above changed; refresh in the plot. |
| *Could not read the data* | The error is on the second line. |

## Keep the plot current

When a node above changes, the plot keeps its data and says *Out of date*;
press **Refresh** to read again. Changes beside the plot (for example a
Dense layer) do not make it out of date. An open Plot window follows Undo
and Redo.

## What Properties shows

**Properties** shows the node's saved settings in words: the plot type,
the data (for example "grid columns V1 .. V61 (61)"), the rows, the colour
and the 3D view. **Details** shows the saved text (`plot_spec`) read-only,
with **Copy**. The settings are changed in the Plot window, not in
Properties. A Dashboard node shows its widgets and filters the same way
(`dashboard_spec`).

## If something goes wrong

- **Nodes that run only inside training cannot be plotted yet**:
  Train/Val/Test Split, DataLoader, Normalize and the layers. The window
  says so and offers **Plot its input (&lt;node&gt;)**, which connects the
  Plot node to the node before it.
- **"&lt;Data Input&gt; has no data yet: open it, choose a file or folder
  and Apply."**: the Data Input above has no file. Set one in its dialog.
- **A saved graph does not load and names an old plot node**: the old
  per-type plot nodes (Bar Chart, Line Plot, Box Plot and the other "Coming
  Soon" plot entries) are gone; the Plot window picks the type. Remove the
  named node and add a Plot node.

## Related

- [Plot data](plotting.md)
- [Plots of model results](model-result-plots.md): Plot nodes after
  Confusion Matrix, ROC Curve and Precision-Recall Curve
- [Visualization nodes](../../reference/nodes/categories/visualization.md)
