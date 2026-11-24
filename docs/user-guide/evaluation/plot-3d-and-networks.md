---
title: 3D plots, networks and trees
description: Scatter, line, surface and mesh plots in 3D, network graphs, trees, and the splits of a trained decision tree or forest.
type: reference
area: evaluation
audience: user
status: stable
since: 1.0.7
owner: evaluation
last_reviewed: 2026-11-25
related:
  - plotting.md
  - plot-types.md
  - plot-maps-and-flows.md
redirect_from: []
---

The Plot window's **3D** and **Graphs** types. How to open the Plot window
and the controls every type shares are in [Plot data](plotting.md).

## 3D

| Type | Columns | What it draws |
| --- | --- | --- |
| Scatter 3D | **X**, **Y** and **Z** numbers, **Colour by** (a number column with many values makes a scale with a colour bar; text makes groups) and an optional **Size** | One point per row. Above 100,000 points it draws a reproducible sample; the label says so and exports use all rows. |
| Line 3D | **X**, **Y** and **Z** in row order | A path (a trajectory, a gradient descent), one line per **Colour by** group. |
| Surface | **Grid columns**, or **X, Y, Z columns** | A surface of heights (next section). |
| Mesh | **X**, **Y** and **Z** rows that are not on a grid (survey points, sensor readings) | The points joined into triangles (Delaunay on X and Y), coloured by Z. Above 5,000 points it joins a reproducible sample. The panel says how many triangles it made. |

### Surfaces

A surface is made from either:

- **Grid columns**: a table of numbers, such as the classic volcano height
  grid (87 rows by 61 columns). Its columns are X, its rows Y (row 1 at the
  front), and each cell is a height.
- **X, Y, Z columns**: one point per row. Rows on a regular grid keep
  their own cells. Scattered rows go on a grid (**Grid cells across**) with
  the mean Z per cell; **Leave cells with fewer rows open** (5) leaves thin
  cells open, and the panel says how many are open.

**Draw**: **Fill**, **Lines** (wireframe) or **Both**. **Shade** lights the
fill from the upper left so slopes show. **Contours on the floor** draws
the level lines under the surface. **Draw** and **Shade** work the same on
a mesh.

### Move around a 3D plot

- Drag to turn it, right-drag to move it, and use the wheel to zoom;
  double-click to fit.
- **Turn** (the default turned view), **Top**, **Front** and **Side** are
  in the toolbar, and **Fit** frames the data.
- The view you leave is saved with the plot (and in the Plot node).
- Hover shows the nearest point's or grid point's X, Y and Z.

Export: PNG of the view, or CSV of the points or cells. SVG is for 2D
plots only.

## Network

One row per link, with **Source** and **Target** columns (names or
numbers) and an optional **Weight**. Links between the same pair are added
up; a node linked to itself is left out. Up to 5,000 nodes.

| Control | Choices |
| --- | --- |
| **Layout** | **Force** (linked nodes pull together; the same picture every time), **Layered** (left to right along the links, for pipelines and model graphs; **Directed** draws arrows) or **Circle** (by group) |
| **Colour nodes by** | **Groups found** (nodes that link more to each other than to the rest), **A column** (pick it as **Node column**) or **One colour** |
| **Node size** | **Links**, **Weight** or **Same** |
| **Labels** | The **Top** N nodes (12), **All** or **None**; labels that would cover each other are left out |

Drag to pan, wheel to zoom. Drag a node to move it; **Re-layout** puts
every node back. Hover a node for its group and its links.

## Tree

One row per node, with **Node** and **Parent** columns (a parent that is
not a node, or an empty one, makes a root), an optional **Value** (shown
under the name) and **Label** (colours the boxes).

- **Direction**: **Top-down** or **Left-right**.
- A box is as wide as the room to its neighbours, so a long name ends in
  "..."; hover shows it in full. When a box is too narrow, its label drops
  the number before the colon.
- Click a box to fold its branch ("+N" shows how many are hidden); click
  again to unfold. Folds stay when you turn the tree.

### The splits of a trained tree model

1. Link the **Model** pin of a **Decision Tree**, **Random Forest** or
   **Gradient Boosting** node (the second pin on its right) to a Plot node.
2. Open the plot. The model trains in the background and the tree shows
   its splits: each box is a rule such as `0: album_total_tracks <= 1.5`,
   or a leaf with its class, sized by the training rows that reach it and
   coloured by class (Gradient Boosting: each leaf's value).
3. A forest or a boosted model opens on its first tree. **Tree in the
   model** steps through them (`<` and `>`, or type a number). It sets the
   `tree = N` row filter, so **Rows** shows it, and it is saved with the
   plot. Gradient Boosting has one tree per class each round ("Round 2 ·
   single").

The trainer saves the model in the project's `artifacts/tree_model/`
folder unless the node's **Save model path** names a file.

## Export

PNG, SVG, or CSV: a network's nodes with their group and links, then the
links; a tree's rows; a mesh's triangle corners.

## From Python

`cx.scatter3d(x, y, z)`, `cx.plot3d(x, y, z)`, `cx.plot_surface(Z)` (a 2D
array) or `cx.plot_surface(x, y, z)`, `cx.plot_trisurf(x, y, z)`,
`cx.network(source, target, weight=)` and `cx.tree(node, parent, value=)`
open the same plots; see [Plots from Python](plots-from-python.md).

## Related

- [Plot data](plotting.md)
- [Maps, flows and hierarchies](plot-maps-and-flows.md)
- [Classic machine learning nodes](../../usage/classic_ml.md)
