# Data Studio

Data Studio is the open graph's data work in one window: bringing data in,
exploring it with SQL, a profile and plots, preparing it, checking it,
getting it ready for training, publishing it and reading the results. Open
it from the Data Studio tab (or the sidebar). Everything in it is a node of
the graph: what you change shows on the canvas, and runs go through the
Data Pipeline executor.

## The window

- **Header**: the graph it works on (and its recipe, when it has one),
  **Show graph** (CyxWiz Studio) and **Publish...** (comes with the Publish
  job; disabled until then).
- **Rail** (left): the data.
  - **Data Inputs**: the graph's Data Input nodes with their rows, or *not
    loaded* (open the node's dialog and Apply, or run the graph).
  - **Node results**: what the last run made: a step's rows, a Row Count
    Check's *pass* or *fail*, *failed* with the error on hover.
  - **Published**: the files the export nodes write, with the rows written
    or *not written*.
  - **Other datasets**: loaded tables no node of this graph made (a query
    saved as a dataset, another graph's results).

  Click an entry to show it in Explore; double-click to show its node on
  the canvas. Data enters the graph only through a Data Input.
- **Jobs**: **Bring in**, **Explore**, **Prepare**, **Validate**, **ML
  readiness**, **Publish**, **Results**, each with a line saying where it
  stands (*1 Data Input*, *1 recipe*, *4 of 4 pass*, *no leakage*,
  *1 output*). Bring in, Explore, Prepare and ML readiness are built; the
  other jobs list their nodes and are being built (TOFIX148).
- **In the graph**: the nodes doing the open job, by kind (*Row Count Check
  x 4*, *Preparation Recipe tier1_paragraphs*, *step nodes Row Filter,
  ...*). Click one, or **Show on canvas**, to select and frame them in
  CyxWiz Studio; a collapsed recipe stands for its steps.
- **Status bar**: *Ready*, *Query running*, *Profiling* or *Comparing
  splits*; the last run
  (*tier1_paragraphs - 15,349 rows - 4/4 checks - 3.1 s - at 12:39:04*,
  orange when a check failed, red when a step failed); *Runs through the
  Data Pipeline executor*.

## Bring in

**Add Data Input...** adds a Data Input node to the graph and opens its
dialog, where you pick the file, its format and the preview, then Apply.
The table lists the graph's Data Input nodes: the node, its file as written
in the graph, rows x columns once loaded, and its state (*loaded*, *not
loaded*, or *failed* with the error on hover). **Open dialog** opens the
node's dialog on the canvas; **Explore** opens a loaded one in Explore >
Profile.

## Explore

Explore has three views: **Query**, **Profile** and **Visualize**. A
dashboard's "Open in Data Studio", "Edit roles", the Asset Browser's "Open
in Data Studio" and Data > Dataset Statistics open the right one.

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

- The editor is the SQL Workbench's: highlighting, completion (type, or
  Ctrl+Space) with the picked dataset's columns, **Ctrl+F** find, and
  **Format** (keywords upper case). Until you type, it follows the dataset
  you pick.
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
- **Turn into SQL step** puts the last result's query into the graph: a new
  SQL step after the SQL step whose result it reads, in that step's recipe
  (a collapsed recipe opens). It takes over what read that step (its checks
  and exports read the new step), shows on the canvas (Ctrl+Z there undoes
  it) and fills on the next run. The query is kept as written; a first
  `WITH` part names the step's `input` as the table you queried. It is
  available when the query reads exactly one table and that table is an SQL
  step's result; otherwise the button says why.

The result line says **exact** (every row shown) or **first 100,000 rows**
(more rows than are shown), then rows, columns, the tables read (with their
rows) and the time. The rows are in the table grid: sort by a header, select
and copy cells, `null` for empty cells, a long value whole on hover (see
[Table Viewer and the table grid](table_viewer.md)). Decimals are numbers and
dates stay dates. Then:

- **Open in Table Viewer**: the result as a table you can sort, filter and
  plot from.
- **Plot**: opens the Plot window on the result (a category and a number
  open as bars, two numbers as a scatter, one number as a histogram; change
  the type there).
- **Add to Dashboard**: the query becomes a widget of a dashboard on the
  picked dataset (see Visualize below for the menu). The widget runs the
  query over the dashboard's rows, so its filters apply. A category and a
  number become bars, two numbers a scatter, one number a histogram. The
  query has to read the picked dataset.
- **Save as Dataset**: runs the query again over all rows and saves the
  result under a name; it then appears in the Dataset list and can be named
  in other queries.

Errors are shown under the buttons in plain words (for example "Only reading
is allowed here" or DuckDB's message for a typo); a long DuckDB message
shows its first line, and **Details** the rest.

## Profile

Profile (it replaced Analyze) describes the picked table and decides what
each column is. It runs in the background as soon as a dataset is picked,
and again when the dataset is re-loaded; **Profile again** runs it by hand.

The first line: rows, columns, when it was profiled and how long it took,
*exact* (or *approximate* on very wide tables, beyond 64 columns, where
distinct counts and quartiles are estimated). The second line counts the
columns per role, then **missing cells**, **duplicate rows** and **data
quality** (100 minus twice the share of missing cells).

The table has a row per column: type, **role**, where the role comes from,
the values (range, dates from - to, or distinct values), missing (count and
share), mean, std, median and outliers (outside 1.5 times the interquartile
range).

### Column roles

Each column has one role: **ID**, **Group**, **Target**, **Numeric**,
**Category**, **Date**, **Text**, **File path**, **Weight** or **Ignore**.
**Group** marks rows that belong together and must stay in one split (a
family of passages, a document, a patient); it is never inferred, so set it
here. The *from* column says who decided:

- **contract**: the graph. The Data Input's label column is the Target. It
  cannot be changed here; change the label in the Data Input.
- **you**: set here. Click the role and pick another (or **Back to
  inferred**). Saved with the project in `datasets/column_roles.json` for
  this file, and used by dashboards, plots and Visualize.
- **inferred**: guessed from the values: text unique per row is an ID, a
  name ending in `_id` is an ID, dates are Date, yes/no and numbers with up
  to 12 values are Category, long or mostly different text is Text, image or
  audio file names are File path, other numbers are Numeric.

Contract first, then yours, then inferred. Setting a different role on the
graph's target shows a warning: training keeps the Data Input label.

Example (Spotify, 8,582 rows): track_id and album_id are ID, album_type,
explicit, artist_name and artist_genres are Category, track_name and
album_name are Text, album_release_date is Date (1952-09-12 to 2025-10-31),
the rest Numeric. Profiled in 0.3 s.

### A column's details

Click a column for its distribution (a histogram for numbers, the most
frequent values otherwise) and its figures (non-missing, missing, distinct,
min, quartiles, max, mean, std, outliers, text length, type).

**Treat as missing**: some files write missing values as text (`N/A`, `-`,
`none`). Type them, comma separated, and **Apply**: the profile counts them
as missing (saved with the project for this file). When a frequent value
looks like one, the panel suggests it. Spotify's artist_genres has `N/A` in
3,361 rows (39.2%); treated as missing, the table has 2.6% missing cells.

The strongest correlations between number columns (Pearson) are listed
under the details (Spotify: artist_popularity and artist_followers +0.64).

## Visualize

Quick plots of the picked dataset, with the same plot types, plot view and
settings as the Dashboard.

- **New plot** lists every plot type, grouped (Basic, Distribution, Grid and
  density, ...). A new plot starts with fitting columns, the target first
  (Spotify: a Histogram opens on track_popularity).
- **PLOTS** on the left lists the plots made here with their types; click
  one to show and edit it. Each dataset keeps its own list for the session.
- The plot has its toolbar (Fit, Log Y, Legend, Export as PNG, SVG or CSV).
- On the right: title, plot type, the columns (with their roles), Colour by,
  **All settings in the Plot window**, and **View SQL** (the query the plot
  runs, Open in Query tab, Copy).
- **Remove** and **Clear all** remove the selected plot or all of them.
- **Add to Dashboard** copies the plot into a Dashboard node on the same
  data: the dashboards of the open graph are listed (one on other data is
  greyed, with the reason), plus **New Dashboard node on Spotify**, which
  adds a Dashboard connected to the Data Input with the automatic layout
  and this plot. The dashboard opens with the plot selected.

## Prepare

The graph's Preparation Recipe, step by step. On the left, **RECIPE STEPS
(NODES)**: each step node in run order with its kind and its last run
(*SQL Query - 15,349 rows - 3.3 s*, *Row Count Check - pass*, *failed* in red
with the error on hover, *not run*). Click a step to show it; double-click to
show it on the canvas. A graph with several recipes lists them first; nodes
that prepare data outside a recipe are listed as *Steps outside a recipe*.

**An SQL step** opens in the SQL Workbench's editor (highlighting,
completion with its inputs' columns once they are loaded, find, Format).
The header names the step, *SQL Query node* and what it reads.

- **Run step** (Ctrl+Enter in the editor) saves the query to the step, then
  runs the Data Pipeline through it and its Row Count Checks. The rows it
  made show below (*15,349 rows - 13 columns - at 21:16:55*, the first 1,000
  in the grid) with the rows each part of the query keeps (*src 7 > parts
  17,083 > kept 15,385 > result 15,349*). The checks' *pass* / *fail* show in
  the step list.
- **Save to step** writes the query into the node without running it, also
  inside a collapsed recipe; Ctrl+Z on the canvas undoes it. *not saved*
  shows while the editor differs from the step; **Reset** goes back to it.
- **Open in Workbench** opens the step's SQL Workbench dialog (Plan,
  Parts, Guided mode); a collapsed recipe opens on the canvas first.
- **Add a step after** adds an **SQL step** or a **Guided step** after this
  one in the recipe (on the canvas, which opens a collapsed recipe).

**Guided | SQL** above the editor switches how the step is edited. A guided
step opens in Guided mode: its operations as cards (the rows after each one
once the step has run) and the selected operation's fields beside them, or
below them when the window is narrow. **Add an operation** lists the
operations by group (type to find one) and adds it after the selected card;
**Start from a template**, **Undo** and **Redo** work on the operations, as in
the Workbench (see [guided steps](sql-workbench.md#guided-steps)). **Save to
step** writes the operations and the SQL they make; **Run step** saves, then
runs. An operation marked in red must be fixed before the step saves or runs.

In SQL mode a guided step's query is read-only; **Edit as SQL** (it asks first)
turns it into an ordinary SQL step when you save. An SQL step in Guided mode
offers **Start a guided step instead**, which replaces the query when you save
(Reset brings it back until then).

**Other steps** (a Row Count Check, a step node) show their last outcome,
their settings as saved in the graph, **Run step**, **Open dialog** and
**Show on canvas**; settings are changed in the node's dialog.

**+ Add a step** under the step list (and **Add a step after** beside an SQL
step) adds after the selected step: an **SQL step**, a **Guided step**, a
**Row Filter**, **Joiner**, **GroupBy** or **Text Clean**, or any other
preparation node from **More nodes**. The node goes on the canvas to the
step's right, the nodes after it move right to make room (a collapsed recipe
opens first), and it is selected; Ctrl+Z on the canvas undoes it.

**Checks.** A Row Count Check node checks one of: a row count (exactly, at
least, at most; all rows or the rows where a column is true or has a value),
**allowed values** of a column, **unique** columns, a column **not empty**, or a
numeric column in a **range**. A failing check stops the run with the count and
the first offending value; nothing after it runs, and its result from an
earlier run is dropped. A guided step's **Add checks** become these nodes when
the step is saved.

## ML readiness

The splits a model trains and is scored on, side by side, before training.

**Which splits.** A Data Split node's inputs, in pin order: train,
validation, test (a graph that feeds separate Data Inputs into them, like a
frozen family split). Without a Data Split, loaded datasets named
`<name>_train`, `<name>_validation` (or `_val`, `_valid`, `_dev`) and
`<name>_test` make a set called `<name>`. Several sets show as chips to
pick from. A Data Split that splits one input by ratio does that during
training; it is not previewed here. A Data Split input that is not loaded is
named in an orange line.

**When.** Opening the job compares the splits in the background (Task View
shows it); so do a change to a split's data and **Check again**. The tab's
line then says what it found (*no leakage*, *leakage found*, *not
checked*).

**The header** names the set and its splits, the verdict (*No leakage
found*, *Leakage: 3 groups in two splits* in red, *Leakage not checked* when
no Group, ID or Text column is set, *One split only*), **Show the Data
Split** on the canvas and **Set roles in Profile** (Explore > Profile on the
train split).

**The cards:**

| Card | What it shows |
| --- | --- |
| Roles | The Text, ID, Group and Target columns, read from the train split's roles (yours first, then the graph's, then inferred). Group is never inferred: set it in Profile. |
| Split sizes | Rows per split, distinct groups and IDs; a split whose rows repeat an ID says how many. |
| Leakage across splits | Groups, IDs and identical texts found in two splits (exact matches), with the pairs when there are any. |
| Text length (words) | Median and mean words per row of the Text column; a split 10% or more shorter or longer than train is named (kjv80: *Validation and test are about 14% shorter than train.*). |
| Class balance | With a Target: the largest classes' share per split, the train split's largest and smallest class, and a class whose share moves 10 points or more between train and another split. The Data Loader's balance mode. |

Everything is counted with SQL on the whole tables; the data is not
changed. The train split's roles come from its profile, cached in the
project's `cache/query_results` like Explore > Profile's. When several
columns read as Text (a hash column does too), the longest one is used;
set Text on another column in Profile to choose it.
