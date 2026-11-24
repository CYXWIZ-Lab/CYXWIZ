---
title: Query a dataset with SQL
description: Run read-only SQL over loaded datasets in Explore > Query, then open, plot, save or turn the result into a recipe step.
type: how-to
area: data-studio
audience: user
status: stable
since: 1.0.7
owner: data-studio
last_reviewed: 2026-11-25
related:
  - explore.md
  - sql-workbench.md
  - prepare.md
  - ../../main-window/table-viewer.md
---

**Explore > Query** in Data Studio runs SQL over the loaded datasets and
shows the rows. Queries only read: changing data belongs in the graph (an
SQL step, Row Filter, ...).

## Run a query

1. In **Data Studio > Explore**, pick a dataset in the rail and click
   **Query**. Until you type, the editor follows the dataset you pick.
2. Type a query. Name tables by their names, in double quotes when the
   name has spaces or capitals:

   ```sql
   SELECT album_type, count(*) AS tracks, round(avg(track_popularity), 1) AS popularity
   FROM "Spotify"
   GROUP BY album_type
   ORDER BY tracks DESC
   ```

3. Click **Run**, or press **Ctrl+Enter** while the editor has focus. The
   query runs in the background: the Engine stays responsive, **Background
   Tasks** shows it, and **Cancel** stops it. One query runs at a time.

| album_type | tracks | popularity |
| --- | --- | --- |
| album | 5856 | 55.7 |
| single | 2219 | 46.4 |
| compilation | 507 | 40.5 |

## The editor

It is the SQL Workbench's editor: highlighting, completion (type, or
**Ctrl+Space**) with the picked dataset's columns, **Ctrl+F** to find, and
**Format** (keywords upper case).

- **Examples** fills in queries for the picked dataset with its real
  columns: first rows, count, rows per value, average per value, a filter,
  columns and their types.
- **Clear** empties the editor.
- The line under the editor lists the tables you can name now.

## What a query may do

- One `SELECT` (or `WITH ... SELECT`). Queries cannot change data, read
  files or reach the network.
- Several tables in one query: `JOIN "Spotify" s ... JOIN "World countries" w`.
- Large datasets kept on disk (Parquet) can be queried too; see
  [Data bigger than memory](../../concepts/data-bigger-than-memory.md).

## Read the result

The result line says **exact** (every row shown) or **first 100,000 rows**
(more rows than are shown), then the rows, columns, the tables read (with
their rows) and the time.

The rows are in the table grid: sort by a header, select and copy cells,
`null` for empty cells, a long value whole on hover (see
[Table Viewer](../../main-window/table-viewer.md)). Decimals are numbers and
dates stay dates.

Errors show under the buttons in plain words (for example *Only reading is
allowed here*, or DuckDB's message for a typo). A long DuckDB message shows
its first line, and **Details** the rest.

## Use the result

| Button | What it does |
| --- | --- |
| **Open in Table Viewer** | The result as a table you can sort, filter and plot from |
| **Plot** | Opens the Plot window on the result: a category and a number open as bars, two numbers as a scatter, one number as a histogram; change the type there |
| **Add to Dashboard** | The query becomes a widget of a dashboard on the picked dataset (the menu is the one described in [Plot a dataset](visualize.md)). The widget runs the query over the dashboard's rows, so its filters apply. The query has to read the picked dataset. |
| **Save as Dataset** | Runs the query again over all rows and saves the result under a name; it then appears in the Dataset list and can be named in other queries. Training does not read it. |
| **Turn into SQL step** | Puts the query into the graph (see the next section) |

## Turn a query into a recipe step

**Turn into SQL step** puts the last result's query into the graph: a new
SQL step after the SQL step whose result it reads, in that step's recipe (a
collapsed recipe opens).

- It takes over what read that step (its checks and exports read the new
  step), shows on the canvas (Ctrl+Z there undoes it) and fills on the next
  run. Training reads its result.
- The query is kept as written; a first `WITH` part names the step's `input`
  as the table you queried.
- It is available when the query reads exactly one table and that table is
  an SQL step's result; otherwise the button says why.

## Related

- [SQL Workbench](sql-workbench.md)
- [Prepare data with a recipe](prepare.md)
- [Explore a dataset](explore.md)
- [Plotting](../evaluation/plotting.md)
