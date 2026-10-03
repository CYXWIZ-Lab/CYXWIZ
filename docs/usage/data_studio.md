# Data Studio

Data Studio is where you look at your data with SQL, a profile and quick
charts, without changing the graph. Open it from the Data Studio tab (or the
sidebar).

## Picking a dataset

The **Dataset** list at the top shows every dataset the Engine holds:

| Column | Meaning |
|---|---|
| name | What the graph calls it: the Data Input node's name ("Spotify"). Hover for the dataset's own name (`ds_datainput_1`); a query can use either. |
| origin | *Data Input*, *node result* (a node's output after a run), or *older dataset*. |
| kind | *table (in memory)*, *table (on disk, Parquet)* for large files, *sparse features*, *images*, *audio*, *text*. |
| size | rows (or files / clips) and columns or classes. |

A Data Input appears once it is loaded: open its dialog and Apply, or open a
Plot node fed by it. The list updates when a dataset is loaded, re-loaded or
removed. **Refresh** reads the picked dataset again; **Open Node Editor**
goes to CyxWiz Studio.

## Query

Write SQL that names tables by their names, in double quotes when the name
has spaces or capitals:

```sql
SELECT album_type, count(*) AS tracks, round(avg(track_popularity), 1) AS popularity
FROM "Spotify"
GROUP BY album_type
ORDER BY tracks DESC
```

| album_type | tracks | popularity |
|---|---|---|
| album | 5856 | 55.7 |
| single | 2219 | 46.4 |
| compilation | 507 | 40.5 |

- **Run** (or **Ctrl+Enter** while the editor has focus) runs the query in
  the background: the Engine stays responsive, Task View shows it, and
  **Cancel** stops it. One query runs at a time.
- Several tables in one query: `JOIN "Spotify" s ... JOIN "World countries" w`.
  Large datasets kept on disk (Parquet) can be queried too.
- The line under the editor lists the tables you can name now.
- Only reading is allowed: one `SELECT` (or `WITH ... SELECT`). Changing data
  belongs in the graph (the SQL node, Filter Rows, ...). Queries cannot read
  files or reach the network.
- **Examples** fills in queries for the picked dataset, using its real
  columns: first rows, count, rows per value, average per value, a filter,
  columns and their types.
- **Clear** empties the editor.

The result line says **exact** (every row shown) or **first 100,000 rows**
(more rows than are shown), then rows, columns, the tables read (with their
rows) and the time. Then:

- **Open in Table Viewer**: the result as a table you can sort, filter and
  plot from.
- **Plot**: opens the Plot window on the result (a category and a number
  open as bars, two numbers as a scatter, one number as a histogram; change
  the type there).
- **Save as Dataset**: runs the query again over all rows and saves the
  result under a name; it then appears in the Dataset list and can be named
  in other queries.

Errors are shown under the buttons in plain words (for example "Only reading
is allowed here" or DuckDB's message for a typo).

## Analyze and Visualize

Analyze and Visualize still work on the picked dataset as before; they move
onto the same profile and plot types next (Profile with column roles, and
Visualize as a one-chart builder).
