---
title: Plots from Python
description: Show matplotlib figures inside the Engine and open Engine Plot windows from a script, a notebook or the Console with the cyxwiz module.
type: how-to
area: evaluation
audience: user
status: stable
since: 1.0.7
owner: evaluation
last_reviewed: 2026-11-25
related:
  - plotting.md
  - plot-types.md
  - ../../usage/notebooks.md
redirect_from: []
---

Python code running in the Engine (a script in the Script Editor, a
notebook cell or the Console) can plot in two ways: matplotlib figures,
shown in the **Plot Output** window, and Engine Plot windows opened with the
`cyxwiz` module. Both come with the Engine; nothing to install.

## Show matplotlib figures

`plt.show()` puts the figure in the Engine instead of opening a separate
window:

```python
import matplotlib.pyplot as plt

plt.plot([1, 3, 2, 4], label="loss")
plt.title("My figure")
plt.legend()
plt.show()
```

| Where the code runs | Where the figure goes |
| --- | --- |
| Script Editor (**F5**) | The **Plot Output** window, which opens by itself if it was closed and keeps every figure of the session |
| Notebook cell | Under the cell. **Open in window** under it shows the image in Plot Output (zoom, pan, save). |
| Console (Python REPL) | Plot Output, also before any script has run |

The Engine sets matplotlib's backend for you (`module://cyxwiz_mpl_backend`,
drawn with Agg). `matplotlib.use(...)` with another backend opens that
backend's own windows instead. In a notebook cell, `%matplotlib inline` (or
any `%matplotlib` line, as notebooks made in Jupyter have) sets the
Engine's backend again from that line on, so a notebook that first called
`matplotlib.use('Agg')` still shows its figures. Other `%` magics are not
supported and stop the cell with a syntax error.

### The Plot Output window

| Control | Does |
| --- | --- |
| Previous / next, thumbnails | Move between the figures |
| Zoom (or the mouse wheel; drag to pan), **Fit**, **100%** | Size the figure |
| Copy, save as PNG, close, clear all | Act on the figure or the list |
| **All** / **Figures** / **Python plots** | Choose what the list shows |

Right-click a figure for Copy, **Save as PNG...** and Close.

## Open Engine Plot windows with `import cyxwiz`

The `cyxwiz` module opens real Engine Plot windows, with one function per
plot kind, like pyplot. The Plot window is the one described in
[Plot data](plotting.md).

```python
import cyxwiz as cx
import pandas as pd

# examples/datasets/metric_blobs.csv from the Engine's examples
df = pd.read_csv("metric_blobs.csv")
cx.scatter("f0", "f1", data=df, color="class", title="Blobs")
cx.hist(df.f2, bins=40)        # arrays and Series work too
cx.bar("class", data=df)       # counts per category
```

- Each call opens a **Plot window**, the same window as a Plot node's:
  change the type, columns, rows, colours and labels there, and export. Its
  source line says where the data came from ("from Python · df · 480 rows ·
  line 6").
- **The same title again updates that window** instead of opening another,
  so running a script twice does not pile up windows. Without `title=`,
  the title is the function and its columns (`bar: class`).
- The plots are listed in **Plot Output** with the matplotlib figures
  ("Python plot · Scatter · live"). Select one for **Show window** / **Open
  window** and **Close plot**.

### Columns and options

- Columns: names with `data=` (a pandas or Polars DataFrame, or a dict),
  or arrays, lists and Series (a Series keeps its name).
- Options: `title`, `xlabel`, `ylabel`, `color` (a column), `bins`,
  `logx`, `logy`, `legend`, `density`, `smooth`, `rows` (the first N
  rows), and any Plot window setting by its saved name
  (`bar_layout="stacked"`, `scale="viridis"`, `graph_layout="layered"`).
- Each call returns a handle: `p.update(...)` plots again into the same
  window (new data, other options); `p.close()` or `cx.close(title)`
  closes it.
- A mistake is raised in Python with the Plot window's words, for example
  `ValueError: column 'nope' is not in the data (columns: a, b)` or
  `scatter needs y values`.

### Functions

| Group | Functions |
| --- | --- |
| Basic | `plot` / `line`, `scatter`, `bar`, `hist`, `area`, `step`, `stem`, `pie`, `polar`, `errorbar` |
| Distribution | `box` / `boxplot`, `violin` / `violinplot`, `kde`, `parallel`, `pairplot` |
| Grid and density | `heatmap(x, y, value)`, `matshow(*columns)`, `hist2d`, `hexbin`, `contour`, `contourf` |
| Vector fields, images | `quiver(x, y, u, v)`, `streamplot(x, y, u, v)`, `imshow(array)` |
| Model results | `confusion_matrix(actual, predicted)`, `roc_curve(actual, score)`, `pr_curve`, `calibration`, `residuals`, `learning_curve(x, *ys)`, `importance(feature, value)` |
| Flows, maps | `sankey(*steps, value=)`, `treemap(*levels, value=)`, `map_points(lon, lat)`, `map_regions(country, value)` |
| 3D | `scatter3d(x, y, z)`, `plot3d(x, y, z)`, `plot_surface(Z)` (a 2D array) or `plot_surface(x, y, z)`, `plot_trisurf(x, y, z)` |
| Graphs | `network(source, target, weight=)`, `tree(node, parent, value=)` |

## Report reinforcement-learning metrics from a script

A script can report RL metrics to the **RL Training Dashboard**
(**Train > RL Training Dashboard**), which opens when the first value
arrives:

```python
import pycyxwiz

for episode in range(100):
    reward = run_episode()          # your code
    pycyxwiz.rl_update_metric("episode_reward", reward)
    pycyxwiz.rl_update_metric("episode_length", 200)
```

The window shows `episode_reward`, `episode_length`, `policy_loss`,
`value_loss`, `explained_variance`, `entropy`, `approx_kl`,
`clip_fraction` and `action_std`. Training a policy on the canvas is in
[Training a policy](../../usage/rl_training.md#the-rl-training-dashboard).

## Related

- [Plot data](plotting.md)
- [Plot types](plot-types.md)
- [Notebooks](../../usage/notebooks.md)
