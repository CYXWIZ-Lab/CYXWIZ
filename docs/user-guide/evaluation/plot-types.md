---
title: Plot types
description: The options of each basic, distribution, grid, vector-field and image plot type in the Plot window.
type: reference
area: evaluation
audience: user
status: stable
since: 1.0.7
owner: evaluation
last_reviewed: 2026-11-25
related:
  - plotting.md
  - model-result-plots.md
  - plot-maps-and-flows.md
  - plot-3d-and-networks.md
redirect_from: []
---

The Plot window's controls that only some types have. The controls every
type shares (rows, columns, colour, export) are in [Plot data](plotting.md).
Model results, maps, flows, 3D and graphs have their own pages:
[Plots of model results](model-result-plots.md),
[Maps, flows and hierarchies](plot-maps-and-flows.md) and
[3D plots, networks and trees](plot-3d-and-networks.md).

## Basic

| Type | Options |
| --- | --- |
| Line, area, step | X (optional: the row number), one or more Y columns, **Colour by**. |
| Stem | X (optional: the row number), one or more Y columns. |
| Scatter | X and one or more Y columns, **Colour by**. |
| Bar, pie | **Categories** and **Values** (optional: rows are counted). |
| Bar with **Colour by** | One bar per group, **Grouped**, **Stacked** or **100%** (each category's shares). Hover lists every group. |
| Histogram | **Values**, bins, median and mean lines, density. |
| Pie | **Donut** shows the total in the middle. |
| Polar | An **Angle** column and a **Radius** column, one line per **Colour by** group (**Points instead of lines** draws points). **Angle in**: Auto (text columns are categories, number columns degrees), Degrees, Radians, or Categories, which share the turn (in numeric order when they are numbers) and close each line. 0 is at the top; angles go clockwise. |

Example of a polar plot: months as the angle, passengers as the radius,
colour by year draws one turn per year.

## Distribution

| Type | Options |
| --- | --- |
| Box, violin | One or more Y columns, **Colour by**. Hover gives the box statistics. |
| KDE | A smooth histogram, one curve per Y column or per **Colour by** group. **Bandwidth** multiplies Silverman's rule (1.00). |
| Parallel coordinates | Two or more number columns, each on its own vertical axis (its lowest and highest value at the ends), one line per row (1,000 at most) coloured by group. Hover highlights the nearest line and lists its values. |
| Error bars | **Groups** and **Values**: each group's mean with its spread. |

## Grid and density

| Type | Options |
| --- | --- |
| Heatmap | **Cell values** sums a column per cell; without it, rows are counted. Labels that are all numbers read in numeric order. |
| Matrix | Two or more number columns. Values: **Correlation (Pearson)**, **Correlation (Spearman, ranks)** (on a -1 to +1 two-sided scale, the value in each cell) or **The values** (the columns as a grid, one row per table row, up to 500 rows). |
| Pair plot | 2 to 6 number columns: every pair as a scatter (sampled to 2,000 rows) and each column alone on the diagonal (a KDE per **Colour by** group, or **Histograms on the diagonal**). |
| 2D histogram | X and Y values; rows are counted per cell. |
| Hexbin | Like the 2D histogram, with hexagons. **Hexagons across**; colour by the count (optionally **Colour by the log of the count**) or the mean of a column. Hover gives the rows in the hexagon and their share. |
| Contour, filled contour | X and Y numbers; **Z** is the density of rows (lightly smoothed) or the mean of a column per cell. **Grid cells across** and **Levels**. Hover gives the value at the mouse. |

## Vector fields

| Type | Options |
| --- | --- |
| Quiver | An arrow per row from **X**, **Y**. **Arrows from**: **u, v** (the arrow's x and y parts) or **Direction + length** (degrees clockwise from north and a length). Tick **Wind** when the direction says where the wind comes from (weather data), so the arrows point where it blows. Arrows are coloured by length (colour bar) and scaled to fit their spacing. A dense grid shows every Nth arrow (about 600 at most); the label says so, and hover and exports use every row. Hover gives the length, the direction it points to, and u, v. |
| Stream | The same columns as Quiver. Lines that follow the field (rows of a grid are used as they are; other tables are averaged onto a 40 x 40 grid), coloured by speed, with an arrowhead on each line. **Density** spaces the lines. Hover gives the speed and direction at the mouse. |

## Images

**Image** draws any table whose number columns make a picture:

| Option | Meaning |
| --- | --- |
| **Pixel columns** | Search "pixel" and **Add all matches**, or pick a range. |
| **Label** | Optional label column. |
| Shape | Square when the count is a square (784 columns: 28 x 28); 3 channels when the count / 3 is a square (3,072: 32 x 32 RGB, as CIFAR-10); tick **Three planes** when the columns are all red, then green, then blue. Otherwise set **Width**. |
| **Show** | **One row** (Previous / Next), a **Gallery** of the chosen rows (the Rows section picks them, for example class = 7; **Pictures at most**), or the **Mean per class** (the average picture of each label, with its count). |
| **Values** | Auto (lowest to highest), 0 to 255 or 0 to 1; **Grey** or the theme scale; **Invert**. |

Hover gives the row, the label, the pixel and its value. Only the rows
shown are read, so wide tables stay quick.

## Related

- [Plot data](plotting.md)
- [Plots of model results](model-result-plots.md)
- [Maps, flows and hierarchies](plot-maps-and-flows.md)
- [3D plots, networks and trees](plot-3d-and-networks.md)
