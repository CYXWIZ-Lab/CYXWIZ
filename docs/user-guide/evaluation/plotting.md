---
title: Plot data
description: Where plots come from in the Engine, and how to open the Plot window, choose the rows and columns, style a plot and export it.
type: how-to
area: evaluation
audience: user
status: stable
since: 1.0.7
owner: evaluation
last_reviewed: 2026-11-25
related:
  - plot-types.md
  - plot-node.md
  - plots-from-python.md
  - model-result-plots.md
  - ../../reference/nodes/categories/visualization.md
redirect_from:
  - ../../usage/plotting.md
---

Every plot in the Engine is drawn by one **Plot window**: the same window
opens from a table, from a Plot node on the canvas, from a Dashboard widget
and from Python. This page shows how to open it and use its controls; the
options of each plot type are in [Plot types](plot-types.md).

| Where the plot starts | Page |
| --- | --- |
| A table in the Table Viewer | This page |
| A **Plot** node on the canvas, at any point of the graph | [Plot data on the canvas](plot-node.md) |
| A **Dashboard** node: many plots of one table | [Visualization nodes](../../reference/nodes/categories/visualization.md) |
| A script, notebook or the Console (`plt.show()`, `import cyxwiz`) | [Plots from Python](plots-from-python.md) |
| Data Studio's **Explore > Visualize** | [Visualize](../data-studio/visualize.md) |
| A trained model's results (confusion matrix, ROC, residuals) | [Plots of model results](model-result-plots.md) |

## Open the Plot window from a table

1. Open a table in the [Table Viewer](../../main-window/table-viewer.md):
   a dataset, a CSV, or a variable from the Variable Explorer.
2. Right-click a column header, choose **Plot**, then **Histogram**, **Line
   Chart**, **Bar Chart**, **Box Plot** or **Use as X-axis for Scatter**.
   Or press **Plot** under the column's statistics in the **Stats** sidebar.
3. The Plot window opens on that plot. Every other type is one click away
   in it.

## Choose the plot type

**Plot type**, on the left, groups the types:

| Group | Types |
| --- | --- |
| Basic | line, scatter, bar, histogram, area, step, stem, pie, polar |
| Distribution | box, violin, KDE, parallel coordinates, error bars |
| Grid and density | heatmap, matrix, pair plot, 2D histogram, hexbin, contour, filled contour |
| Images | image |
| Vector fields | quiver, stream |
| Model results | confusion matrix, ROC curve, precision-recall curve, calibration, residuals, learning curve, feature importance |
| Flows and hierarchies | Sankey, treemap |
| Maps | map: points, map: regions |
| 3D | scatter 3D, line 3D, surface, mesh |
| Graphs | network, tree |

Pick one; the columns that still fit are kept.

## Choose the rows

**Rows**, at the top of the right panel:

| Choice | Draws |
| --- | --- |
| **All** | Every row |
| **First** | The first N rows |
| **Range** | A range of rows, numbered from 1 as in the Table Viewer |
| **Filter** | Rows matching conditions *column = != < <= > >= contains value*, all of which must match ("class = 7" keeps 7,293 of MNIST's 70,000 rows) |

The label over the plot says which rows are drawn; exports and the Python
script use the same rows. To filter rows for the whole graph, use a Row
Filter node instead.

## Choose the columns

**Data** holds the X column (or the row number for lines), one or more Y
columns (one series each), **Colour by** a column, bins, median and mean
lines, density, smoothing (a moving average) and the title. **Values**
lists the count, missing, min, max, mean and median of the plotted column.

In a wide table, every column field opens a list with a search box:

- each column shows its type (# number, Aa text), its range and how often
  it is not 0 ("0 to 255 · 66.3% not 0"). Columns with one value are
  hidden unless you untick that;
- Up / Down move, Enter picks, Esc closes;
- for Y values, **Add all N matches** of the search, or a **Range** of
  columns ("pixel400" to "pixel409"). The chosen columns show as chips;
  click one to remove it. The legend lists 12 series; hover shows all.

**Colour by** a column with up to 12 values (class 0-9) gives one series
per value. On a scatter, a number column with more values colours each
point on the colour scale, with a colour bar named after the column
(values on both sides of 0 use the two-sided scale); **Groups / Scale**
switches. Other plot types split such a column into six equal ranges.

## Style the plot

**Colour** applies to every plot:

- **The scale**, when the plot colours by values (heatmaps, contours,
  surfaces, maps, scatter colour scales): **Theme** (the default),
  **Viridis**, **Plasma**, **Magma**, **Cividis**, **Turbo**, **Greys**, or
  **Cool-warm** for values around 0. The menu shows each gradient.
- **Reverse** runs the scale the other way. Untick **Range: data** to set
  the scale's ends yourself.
- **Series colours**: a swatch per series (pie: per slice). Click it for
  the theme's colours, or **Custom** for any colour; **Theme colour** goes
  back to the default.

The choices are saved with the plot, and SVG export uses them too. New
plots start from the theme (**Preferences > Theme**): the series colours,
the plot area (a shade of the window colour) and the text.

On a line or scatter plot, **Show the y = x line** draws a reference line
(chance, on a ROC curve).

## Read and move the plot

- Hover for values: the nearest x of every series, the scatter point, the
  histogram bin and its share, the category, the pie slice, the box
  statistics, the heatmap cell.
- Drag to pan, use the wheel to zoom, double-click or **Fit** to fit.
- **Log Y** and **Legend** in the toolbar. On a line or scatter chart the
  legend sits in the corner the data leaves free.
- The label next to the toolbar says what is drawn: *exact* (all values),
  *reduced* (long lines drawn with the lowest and highest value of each
  step; hover and exports use all points), *sampled* (an even sample of a
  large scatter; exports use all rows), or *first N rows* (a variable read
  with a row limit).

In a narrow place the toolbar's buttons move under the data label, and
wrap when even that is too narrow. The window icon opens the plot in its
own window.

## Export the plot

**Export** offers **Save image as PNG...**, **Save image as SVG...** (2D
plots), **Copy image**, **Save data as CSV...**, **Copy data**, and **Plot
with Python (copy script)**: a matplotlib script you paste into the Script
Editor and run to get the figure in Plot Output.

To keep plots of a dataset, or turn one into a Plot node, use Data
Studio's **Explore > Visualize**.

## Charts in the Training Dashboard

The Training Dashboard's charts use the same plot view: hover for the
values of every curve at an epoch (read from all points), the data label,
**Legend** and **Export** (PNG, SVG, copy image, CSV with every point, copy
data). Long runs stay smooth because the charts draw a reduced copy of each
series (up to 4,000 points, with every spike and dip kept).

A training point during an epoch is the mean loss (and accuracy) of the
batches since the previous point (every log interval), not the epoch's
running mean, so the curve does not jump at each epoch start. At each
epoch's end the whole-epoch values are kept too: the **best** line on the
training cards names the best whole epoch, never one batch. Validation is
one point per epoch. See [Training Dashboard](../../main-window/training-dashboard.md).

## Related

- [Plot types](plot-types.md)
- [Plot data on the canvas](plot-node.md)
- [Plots from Python](plots-from-python.md)
- [Plots of model results](model-result-plots.md)
- [Maps, flows and hierarchies](plot-maps-and-flows.md)
- [3D plots, networks and trees](plot-3d-and-networks.md)
