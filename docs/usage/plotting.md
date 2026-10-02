# Plotting in the Engine

What draws plots in the CyxWiz Engine today, and how to use each part. A
new plot system (one Plot window with a plot-type picker, shared with the
Dashboard) is being built in TOFIX134; this page describes what works now.

## Figures from Python (matplotlib)

`plt.show()` in a script, a notebook cell or the Console puts the figure in
the Engine instead of opening a separate window.

```python
import matplotlib.pyplot as plt

plt.plot([1, 3, 2, 4], label="loss")
plt.title("My figure")
plt.legend()
plt.show()
```

- **Script (Script Editor, F5):** the figure opens in the **Plot Output**
  window. Plot Output opens by itself if it was closed; it keeps every
  figure of the session (arrows or thumbnails to move between them).
- **Notebook cell:** the figure shows under the cell, in the notebook.
- **Console (Python REPL):** after a script has run in the session, the
  figure goes to Plot Output as well.

Plot Output toolbar: previous / next figure, zoom (or mouse wheel, drag to
pan), Fit, 100%, copy to the clipboard, save as PNG, close the figure,
clear all. Right-click a figure for Copy, Save as PNG and Close.

## The Plot window (Table Viewer)

Open a table in the Table Viewer (a dataset, a CSV, or a variable from the
Variable Explorer), then either right-click a column and choose **Plot**
(Histogram, Line Chart, Bar Chart, Box Plot, or Use as X-axis for Scatter)
or press **Plot** under the column's statistics. The Plot window opens:

- **Plot type** (left): Basic (line, scatter, bar, histogram, area, step,
  stem, pie), Distribution (box, violin, error bars), Grid and density
  (heatmap, 2D histogram). Pick one; the columns that still fit are kept.
- **Data** (right): the X column (or the row number for lines), one or
  more Y columns (one series each), **Colour by** a column (one series per
  value; the 11 largest groups, the rest as "other"), bins, median and
  mean lines, density, smoothing (moving average), the title. **Values**
  lists count, missing, min, max, mean and median of the plotted column.
- **The plot**: hover for values (the nearest x of every series, the
  scatter point, the histogram bin and its share, the category, the pie
  slice, the box statistics, the heatmap cell). Drag to pan, wheel to zoom,
  double-click or **Fit** to fit. **Log Y**, **Legend**.
- The label next to the toolbar says what is drawn: *exact* (all values),
  *reduced* (long lines drawn with the lowest and highest value of each
  step; hover and exports use all points), *sampled* (an even sample of a
  large scatter; exports use all rows), or *first N rows* (a variable read
  with a row limit).
- **Export**: save image as PNG or SVG, copy image, save data as CSV, copy
  data, Plot with Python (copies a matplotlib script; paste it in the
  Script Editor and run it to get the figure in Plot Output).
- The window icon opens the plot in its own window; **Open in Visualizer**
  sends the column to the Visualizer.

Plot colours come from the theme (Preferences > Theme): the series
colours, the plot area (a shade of the window colour) and the text.

## Training Dashboard

Opens with **Train** on the CyxWiz Studio canvas; in the window list it is
**Training** (Training group), docked at the bottom right by default. It shows the run status, KPI cards (loss, accuracy, time
remaining), live loss and accuracy charts, custom metrics, data
preparation progress, execution truth and run comparison. Long runs stay
smooth: charts draw a reduced copy of each series (up to 4000 points, with
every spike and dip kept). Use the window icon on a chart to open it in its
own window.

Only canvas training reports to this window.

## RL Training Dashboard

A separate window for reinforcement learning (episode reward and length,
policy loss, value loss, explained variance). It opens with **Train RL** on
the canvas (graphs with RL nodes), or when a script reports RL metrics:

```python
import pycyxwiz

for episode in range(100):
    reward = run_episode()          # your code
    pycyxwiz.rl_update_metric("episode_reward", reward)
    pycyxwiz.rl_update_metric("episode_length", 200)
```

Metric names the window shows: `episode_reward`, `episode_length`,
`policy_loss`, `value_loss`, `explained_variance`.

## Not available yet

- The `cyxwiz_plotting` Python module cannot open Engine windows;
  `show_plot` raises an error saying so. Use matplotlib as above.
- The plot entries in the canvas node search (Line Plot, Box Plot, Quiver
  Plot, ...) are placeholders; they are replaced by one Plot node whose
  window picks the plot type (TOFIX134 P2).
- Bar Chart node: plots the raw file of the first Data Input upstream, not
  the data at its position in the graph.
