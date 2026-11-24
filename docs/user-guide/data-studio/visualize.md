---
title: Plot a dataset
description: Make quick plots of a dataset in Explore > Visualize, keep one as a Plot node in the graph, or add it to a dashboard.
type: how-to
area: data-studio
audience: user
status: stable
since: 1.0.7
owner: data-studio
last_reviewed: 2026-11-25
related:
  - explore.md
  - query.md
  - ../evaluation/plotting.md
  - ../../reference/nodes/categories/visualization.md
---

**Explore > Visualize** in Data Studio makes quick plots of the picked
dataset, with the same plot types, plot view and settings as the Dashboard.
The plots read the data; they do not change what training reads.

## Make a plot

1. In **Data Studio > Explore**, pick a dataset in the rail and click
   **Visualize**.
2. Click **New plot** and pick a plot type. The types are grouped (Basic,
   Distribution, Grid and density, ...). A new plot starts with fitting
   columns, the target first (Spotify: a Histogram opens on
   `track_popularity`).
3. On the right, set the title, the plot type, the columns (with their
   roles) and **Colour by**.

**PLOTS** on the left lists the plots made here with their types; click one
to show and edit it. Each dataset keeps its own list for the session.

| Control | What it does |
| --- | --- |
| Plot toolbar | Fit, Log Y, Legend, and export as PNG, SVG or CSV |
| **All settings in the Plot window** | Opens the plot in the Plot window, with every setting |
| **View SQL** | The query the plot runs, with **Open in Query tab** and **Copy** |
| **Remove**, **Clear all** | Remove the selected plot, or all of them |

The list of plot types scrolls inside the main window; a long note after an
action goes on its own line.

## Keep a plot in the graph

**Keep as Plot node** adds the selected plot to the graph as a Plot node
reading the node that made the data (a Data Input, a step, or a recipe's
result for a step inside a recipe), placed below it.

- The Explore strip lists it, and Ctrl+Z on the canvas undoes it.
- The Plot node is kept with the graph and reads the data again when it
  opens. Training does not read it.
- The button is off, with the reason on hover, when no plot is selected or
  the data is not a table of this graph's nodes.

## Add a plot to a dashboard

1. Click **Add to Dashboard**. The menu lists the dashboards of the open
   graph; one on other data is greyed, with the reason.
2. Pick a dashboard, or **New Dashboard node on <dataset>** (for example
   **New Dashboard node on Spotify**), which adds a Dashboard node connected
   to the Data Input with the automatic layout and this plot.

The dashboard opens with the plot selected. Query's **Add to Dashboard**
uses the same menu.

## Related

- [Plotting](../evaluation/plotting.md)
- [Dashboard node](../../reference/nodes/categories/visualization.md)
- [Query a dataset with SQL](query.md)
- [Explore a dataset](explore.md)
