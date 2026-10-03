# Dashboard node

A Dashboard shows the data at its place in the graph as a summary and a grid
of widgets that starts by itself and that you can change. It belongs to the
same family as the Plot node: every plot type is a widget.

```
Data Input (Spotify) --> Dashboard
```

Search **Dashboard** on the canvas (or the Visualization group), connect a
table to it, and double-click it (or right-click > Configure...). The node
shows its state underneath: *Not read yet*, *Reading ...*, *Dashboard · 12
widgets*, *Out of date*.

## Data

The line at the top says where the data comes from ("Data at Spotify ·
8,582 rows × 15 columns · read 15:53"), with **Refresh** (or **Cancel**
while reading). Only the nodes above the dashboard run, in the background
(Task View); a loaded Data Input is read directly. When a node above
changes, the dashboard says it is out of date until you refresh; a node that
runs only inside training says so instead of showing old charts.

## The automatic layout

The first time it opens, the dashboard profiles the data (the same profile
as Data Studio's Profile tab) and builds:

- the **target** (the Data Input's label column): a histogram for a number,
  bars for a category;
- a **bar chart per category column** with up to 30 values;
- a **histogram per number column**;
- a **correlation matrix** of the number columns;
- the **target by its most correlated column** (a scatter, or a box per
  class for a category target).

**Regenerate** rebuilds these automatic widgets from the current data and
roles; widgets you added stay.

## Summary strip

Rows (filtered, *of* all rows), columns, missing cells, duplicate rows and
the target's mean (or most frequent value), with *all* for the unfiltered
figure when a filter is on.

## Filters (cross-filtering)

Click a bar, a pie slice or a histogram bin: every **other** widget shows
only those rows (the widget you clicked keeps showing all its values, so you
can pick another). The filter bar lists the conditions ("album_type =
single · 2,219 of 8,582 rows"); click a condition to remove it, or **Clear
all**. Clicking the same bar again removes its filter. Filters are saved in
the node.

Example (Spotify): click *single* in album_type: 2,219 of 8,582 rows, the
target mean 46.36 (all rows 52.36).

## Fields and roles

The left column lists the fields with their roles (ID, Target, Numeric,
Category, Date, Text, ...). Roles are edited in one place: **Edit roles in
Data Studio** opens Data Studio's Profile tab on this data. The contract
(the Data Input label) comes first, then roles set in Data Studio, then
inferred ones.

## Widgets

- **Add widget**: KPI, Table, or any plot type (grouped as in the Plot
  window). A new widget starts with fitting columns.
- Click a widget's title to edit it on the right: title, plot type, the
  columns (X, Y, Colour by), or a KPI's measure (rows, sum, mean, median,
  min, max, distinct, missing) and field, or a table's columns and rows.
  **All settings in the Plot window** opens the full Plot window on the
  same data; changes there go back into the widget.
- **Size and place**: width (1 to 12 columns) and height, **Earlier** /
  **Later** to move it, **Remove widget**.
- **Re-query** runs every widget's query again.

Widgets read only the columns they need, through the shared query service,
and plots are prepared in the background; very large tables are drawn from
a reproducible sample of 1,000,000 rows (the widget says *sampled*).

## When the data changes

- New values or rows: every widget updates; the layout stays.
- A column a widget uses is gone: the widget says so; when one new column of
  the same type appeared (a rename), **Rebind to** it, or **Remove**.
- A column became text where a widget needs numbers: the widget says so and
  suggests a Bar.
- New columns: existing widgets keep working; **Regenerate** adds cards for
  them.
