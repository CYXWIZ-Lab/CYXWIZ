# Table Viewer and the table grid

The **Table Viewer** shows a table held in memory, to sort, filter, format,
edit, plot or export. Its grid is also the results grid of the **SQL
Workbench** (Results), an SQL step's **Output preview** in Properties and
Data Studio's **Query** tab, so a table looks and behaves the same in all
four.

## Where tables come from

- **Open dataset** (or **Open a loaded dataset...** when nothing is open):
  the first rows of a dataset a Data Input loaded (10,000 by default); the
  dataset itself stays where its Data Input put it.
- **Open in Table Viewer** in Data Studio's Query tab, the SQL Workbench and
  an SQL step's Properties (input, output preview, checks). An output larger
  than 200,000 rows opens its first 200,000 and says so in the tab name.
- **View data** on a Python value in the Variable Explorer: the tab shows
  where the value comes from, **Read again**, and for arrays with more than
  two axes which slice is shown (Up / Down step the first index).
- A notebook or script that shows a table.

Each table opens in its own tab and comes to the front. Opening the same
in-memory table again (a step's output after a rerun) refreshes its tab.
Each table keeps its own column widths.

A dataset or a result is read in the background, so the Engine keeps
working while a large one opens: under the toolbar the viewer shows
*Opening <name>* with how far it is (*40,000 of 100,000 rows*) and
**Cancel**, and the same task is in the **Task viewer** (*Open <name> in
the Table Viewer*). The tab opens when the rows are read; a read that fails
or is cancelled says so in the status bar. MNIST's test set (10,000 rows,
785 columns) opens in about 3 seconds.

## The grid

- **One line per cell.** Line breaks and tabs show as spaces; a value longer
  than 300 characters shows its start. Hover a cut-off cell for its whole
  value. Text columns open at most 420 px wide; drag a column edge to widen.
- **Empty cells** show a faint `null`.
- **Numbers** show in their shortest exact form (`106.4`); a long one shows
  10 significant digits (`0.6666666667`). Decimal columns from SQL are
  numbers. Dates show as `2024-01-02`.
- The header icon says the column's kind: `#` number, `A` text.
- **Sort**: click a header (ascending, descending, then the table's own
  order). The row numbers stay frozen at the left while you scroll right.
- **Wide tables**: the grid shows up to 500 columns at a time. A table with
  more (MNIST has 785) shows the column pager under the grid, *Columns:
  1 - 500 of 785* with previous and next; the frozen columns (Lock) stay in
  every block, and the rows keep their sort and filter.
- **Select**: click a cell; Ctrl+click adds a cell, Shift+click a range.
- **Right-click a header**: sort, clear the filter, colour map (Viridis,
  Inferno, RdYlGn, Blues, Plasma), number formats (decimals, thousands
  separator, percentage, prefix / suffix, currency), **Plot** (Table Viewer
  only), copy the column.
- **Right-click a cell**: copy the cell, the row or the selection; select the
  row or the column; **Filter by This Value**; View Column Statistics (Table
  Viewer only).

In the Workbench, the preview and the Query tab the grid is read-only and
shows no data bars; it shows up to 500 rows (Workbench), 5 rows (preview) or
the query's rows (Query).

## Table Viewer toolbar

| Control | What it does |
|---|---|
| Open dataset | Another loaded dataset, in a new tab. |
| Stats | The Stats sidebar: the selected column's rows, missing, distinct, empty texts; numbers add min, max, mean, std, quartiles and a histogram (**Plot** opens it in the Plot window); text adds its most common values. Computed in the background (in the Task viewer as *Profile <name>*), again after an edit. It is the same profile as Data Studio's **Profile** tab: a tab showing a whole loaded dataset, unedited, is profiled exactly as Data Studio profiles it, and the answers either one saved serve the other (MNIST: 62 s the first time, 1.3 s for the other). A tab with part of a dataset, a result or an edited table is profiled on its own rows. |
| Data Bars | A bar in each number cell, its length the value within the column's range. |
| Line # | The row-number column. |
| Rows per page | 10 to 1,000 (100). Past one page the pager is under the grid: first, previous, page, next, last, and which rows are shown. |
| Filter (funnel) | Type and press Enter: matching cells are highlighted. Not case sensitive, any part of a cell (Find sets both). |
| Hide | Only the rows with a matching cell; sorting keeps them. No match shows "No row contains ..." under the grid. |
| Lock | How many data columns stay frozen at the left. |
| Find | A window over the same filter: **Case sensitive**, **Whole cell**, how many cells in how many rows match, **Clear**. |
| Save | Writes an edited table back to its file (CSV, TSV / TXT, HDF5); Ctrl+S. Not for a Python value: export it instead. |
| Export | **CSV**, **TSV**, **JSON** or **Parquet** to a file you pick, or **Copy as CSV**. **Only the shown rows** writes the grid's rows in its order (sorted, filtered); **Column names first** applies to CSV and TSV. The status bar says what was written. |
| X | Closes the tab. |

**Edit a cell**: double-click it, type, Enter (Escape cancels; a click
elsewhere keeps the edit). The tab gets a `*` until saved; the column's
kind, data bars and Stats follow the edit.

## Example

In Data Studio's Query tab:

```sql
SELECT i, 'Row ' || i AS label, DATE '2024-01-01' + CAST(i AS INTEGER) AS day, i * 0.5 AS half
FROM range(1, 8) t(i)
```

**Open in Table Viewer**, then **Export > CSV...** writes:

```
i,label,day,half
1,Row 1,2024-01-02,0.5
2,Row 2,2024-01-03,1
...
7,Row 7,2024-01-08,3.5
```

**Find** `row 3` matches 1 cell in 1 row; with **Case sensitive** on it
matches none.
