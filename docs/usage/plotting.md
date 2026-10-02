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

## Table Viewer quick plots

Open a table in the Table Viewer (a dataset, a CSV, or a variable from the
Variable Explorer), right-click a column and choose **Plot**: histogram,
bar, line, scatter, box, pie, stairs, stem or area. The plot opens in the
Quick Plot window. **Plot with Python** copies a
complete matplotlib script for the same plot to the clipboard; paste it in
the Script Editor and run it to get the figure in Plot Output.

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
