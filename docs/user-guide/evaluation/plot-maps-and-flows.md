---
title: Maps, flows and hierarchies
description: Sankey diagrams, treemaps, and maps of points or countries on the built-in world map.
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
  - plot-3d-and-networks.md
redirect_from: []
---

The Plot window's **Flows and hierarchies** and **Maps** types. How to open
the Plot window and the controls every type shares are in
[Plot data](plotting.md).

## Flows and hierarchies

| Type | Columns | What it draws |
| --- | --- | --- |
| Sankey | Two or more **Steps** columns, left to right (text or numbers as categories), and an optional **Value** | Each band joins a category of one step to one of the next; its width is the rows, or the sum of Value. A ready flow table works the same way: Steps = source, target and Value = the flow. **Largest per step** (8) keeps the biggest categories and joins the rest as *other*. Hover a node for its total and share, a band for its value and its share of where it starts. |
| Treemap | One to four **Groups** columns (outer to inner) and a **Size** (summed; empty: rows) | Rectangles whose areas follow the size. **Colour** takes a number column (the mean per rectangle on the colour scale); leave it empty to colour by the top group. Click a group to zoom into it; right-click, or click the path at the top ("All > Asia"), to go back. The 3,000 largest groups are drawn; the label says when there are more. |

## Maps

The maps use a built-in world map: the Natural Earth 1:110m country
outlines (public domain), shipped with the Engine, so they work offline.

| Type | Columns | What it draws |
| --- | --- | --- |
| Map: points | **Longitude** and **Latitude** (degrees), an optional **Size** and **Colour by** (a number makes a scale, text makes groups) | One point per row. The map frames the points; pan and zoom like any plot. Hover gives the row's values and the country under it. |
| Map: regions | A **Country** column (names such as "France" or "United States", ISO codes FRA or FR, and common names like UK or Viet Nam) and a **Value** | Each country coloured by its value. Several rows for one country: **Sum** or **Mean**. **Colour by the log of the value** for values across orders of magnitude. Countries with no rows stay grey. Hover gives the country, its ISO code, the value and the rows. |

A longitude outside -180 to 180 or a latitude outside -90 to 90 is refused
with a hint (columns swapped?). Country names that match no country are
listed under the controls and in the CSV export.

Examples: a table of countries with population and GDP as a treemap of
continent > country, sized by population and coloured by GDP per person,
or as **Map: regions** of GDP per person (log); a feed of earthquakes as
**Map: points**, sized by magnitude and coloured by depth.

## From Python

`cx.sankey(*steps, value=)`, `cx.treemap(*levels, value=)`,
`cx.map_points(lon, lat)` and `cx.map_regions(country, value)` open the
same plots; see [Plots from Python](plots-from-python.md).

## Related

- [Plot data](plotting.md)
- [3D plots, networks and trees](plot-3d-and-networks.md)
