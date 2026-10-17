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
- **Rail** (left): the data. In a narrow Data Studio (under about 670
  pixels) it sits above the jobs instead, so the job keeps the window's
  width.
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
  *1 output*).
- **In the graph**: the nodes doing the open job, by kind (*Check
  x 4*, *Preparation Recipe tier1_paragraphs*, *step nodes Row Filter,
  ...*). Click one, or **Show on canvas**, to select and frame them in
  CyxWiz Studio; a collapsed recipe stands for its steps. When the line is
  too narrow, **Show on canvas** goes under the nodes.
- **Status bar**: *Ready*, *Query running*, *Profiling*, *Comparing
  splits* or *Data Pipeline running*; the last run
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
Profile. Where the five columns do not fit, each Data Input is listed as a
block with the same facts and links.

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
  Large datasets kept on disk (Parquet) can be queried too. How large tables are kept and read: [big_data.md](big_data.md).
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
- **Keep as Plot node** adds the selected plot to the graph as a Plot node
  reading the node that made the data (a Data Input, a step, or a recipe's
  result for a step inside a recipe), placed below it; the Explore strip lists
  it and Ctrl+Z on the canvas undoes it. The Plot node is kept with the graph
  and reads the data again when it opens. Off, with the reason on hover, when
  no plot is selected or the data is not a table of this graph's nodes.
- **Remove** and **Clear all** remove the selected plot or all of them.
- The list of plot types scrolls inside the main window; a long note after
  an action goes on its own line.
- **Add to Dashboard** copies the plot into a Dashboard node on the same
  data: the dashboards of the open graph are listed (one on other data is
  greyed, with the reason), plus **New Dashboard node on Spotify**, which
  adds a Dashboard connected to the Data Input with the automatic layout
  and this plot. The dashboard opens with the plot selected.

## Prepare

The graph's Preparation Recipe, step by step. On the left, **RECIPE STEPS
(NODES)**: each step node in run order with its kind and its last run
(*SQL Query - 15,349 rows - 3.3 s*, *Check - pass*, *failed* in red
with the error on hover, *not run*). Click a step to show it; double-click to
show it on the canvas. A graph with several recipes lists them first; nodes
that prepare data outside a recipe are listed as *Steps outside a recipe*.

**An SQL step** opens in the SQL Workbench's editor (highlighting,
completion with its inputs' columns once they are loaded, find, Format).
The header names the step, *SQL Query node* and what it reads.

- **Run step** (Ctrl+Enter in the editor) saves the query to the step, then
  runs the Data Pipeline through it and its Check nodes. The rows it
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

**Other steps** (a Check, a step node) show their last outcome,
their settings as saved in the graph, **Run step**, **Open dialog** and
**Show on canvas**; settings are changed in the node's dialog.

**+ Add a step** under the step list (and **Add a step after** beside an SQL
step) adds after the selected step: an **SQL step**, a **Guided step**, a
**Row Filter**, **Joiner**, **GroupBy** or **Text Clean**, or any other
preparation node from **More nodes**. The node goes on the canvas to the
step's right, the nodes after it move right to make room (a collapsed recipe
opens first), and it is selected there; Data Studio stays in front, and the
step joins the list. Ctrl+Z on the canvas undoes it.

**Checks.** A Check node checks one of: a row count (exactly, at
least, at most; all rows or the rows where a column is true or has a value),
**allowed values** of a column, **unique** columns, a column **not empty**, or a
numeric column in a **range**. A failing check stops the run with the count and
the first offending value; nothing after it runs, and its result from an
earlier run is dropped. A guided step's **Add checks** become these nodes when
the step is saved. **Validate** lists them all and adds new ones.

## Validate

The graph's checks in one table: **check (node)**, what it **reads** (the step
whose rows it checks, past any checks in front of it), its **rule** in words
(*exactly 15,349 rows*, *exactly 3,712 rows where source_id = pg41633*,
*exactly 0 rows where training_admitted is true*, *role is one of book,
chapter*, *paragraph_id is unique*, *text is never empty*, *word_count from 1
to 900*) and its **last run**: *pass - 15,349 rows* (a row count shows what it
counted), *pass*, *fail* with the message under the rule, *could not run* (for
example a column that is not there), or *not run* (not run yet in this
session, or after a check that failed). The line above says *4 of 4 checks
pass*. Double-click a check's name, or **Show**, to show its node on the
canvas.

- **Run checks** runs the Data Pipeline through every check and the steps
  before them (an export after the checks is not written).
- **Add a check**: **Row count**, **Allowed values**, **Unique**, **Not
  empty** or **Range** opens the form below the table. **After** picks the
  step it checks (the dataset chosen in the rail, else the step the other
  checks read); **Name** is shown when it fails (by default the rule). The
  rule's fields pick from the step's columns once it has run (type them
  before). *Rule:* shows the rule as it will read, and **Add check** stays off,
  with the reason on hover, until the form is complete. The check is added as
  a Check node right after the step (a collapsed recipe opens on
  the canvas; Data Studio stays in front); Ctrl+Z on the canvas undoes it.
- **Edit** in a check's row opens the same form on that node: change the kind
  or the rule; **Save check** writes the node (Ctrl+Z on the canvas undoes it).
- **Data Validator...** opens the validator's form: **Reads** picks the step,
  and **Required columns**, **No missing value** and **No repeated value** pick
  columns (checkboxes once the step has run). **Add validator** adds a Data
  Validator node reading the step, beside what already reads it; for a step
  inside a recipe it reads the recipe's result, since every recipe step must
  lead to the recipe's output. It reports the rows that break a rule as an
  issues table rather than stopping the run; its row shows its rules and *no
  issues* / *15,345 issues* (the table is its node result in the rail), and
  **Edit** opens the same form on it.
- When the graph cannot run, a red line under the summary says why and **Run
  checks** is off.
- In a narrow Data Studio, what a check reads goes under its rule and **Edit**
  / **Show** under its last run.

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

## Publish

The graph's outputs (Export CSV, Export JSON, Export Parquet, Data Output) in
one table: **output (node)**, what it **reads** (the step, past any checks in
front of it), the file it **writes** (relative to the project when inside it;
the full path on hover) and when it was **last written** (*15,349 rows -
15:19:58*, *not written yet*, *failed* with the error). Click a row to show its
lineage; **Show** selects the node on the canvas.

**Lineage beside the file.** Every time an output writes its file, the run
also writes `<file>.lineage.json` beside it, whether the run came from
Publish, Run step or Execute Pipeline. Publish shows the selected output's:

- **Made by**: the recipe and its revision (the first 12 hex digits of the
  SHA-256 of its steps' types and settings, so any change to a step changes
  it) and the number of steps;
- **Inputs**: each Data Input's file with its size and SHA-256. A file is
  hashed once (the digest is kept by path, size and write time in the
  project's `cache/ingestion/digests`). A CSV over the data budget's result
  threshold is hashed while it is converted, so it is not read again. A
  larger file never read in full (a Parquet file opened in place) shows *not
  hashed (large file; size and time identify it)*, and the lineage file holds
  its write time;
- **Result**: rows, columns, the file's size and SHA-256 (hashed while the
  file is written, so the export is not read back);
- **Checks**: *4 of 4 pass*, or the first one that did not;
- **Written**: when, from which graph file.

**Steps** opens the list of steps above the output. The file itself also holds
each step's settings and outcome, the schema, and the full digests.

**Publish...** (here or in the header) opens *Publish dataset*:

- **Output**: one of the graph's outputs (its name and file are set and it
  runs), or **New Parquet export**, which adds an Export Parquet node reading
  **Reads** (for a step inside a recipe, the recipe's result) and runs it.
- **Name** is the node's name; **Location** is the file, relative to the
  project unless it starts with a drive (suggested:
  `datasets/published/<step>.parquet`).
- **Publish** writes the file and its lineage; the table and the card update
  when the run ends. Ctrl+Z on the canvas undoes the node change.

**Add Data Output** adds a Data Output node reading the step chosen in the
rail (or the last step); its format and file are set in its dialog.

## Results

A model's results next to the data. Pick **Language model previews** or
**Test run**. The job's line counts what there is to show (*test: 72 rows ·
2 preview runs*, and any evaluation nodes in the graph).

**Language model previews** reads the generation previews a causal language
model's training writes each epoch (`checkpoints/<graph>/<run>/
generation_previews/epoch_<n>.json`). **Run** lists every run of the project
that wrote previews, newest first; **Epoch** picks one of its epochs.

- The line under them: the number of preview prompts, the sampling, the token
  limit.
- Tiles: **Right book**, **Right chapter**, **Right verse** (when the prompts
  carry a `<reference>` such as *Psalm 34:13*: the model's first *Book C:V*
  citation is compared with it) and **Stopped by itself** (*1 of 4*; how many
  hit the token limit and how many loop).
- One row per prompt: the **question** (the prompt's `<question>`, else its
  last *User:* text), the **evidence** reference, what the model **cited**,
  book / chapter / verse marks, and how it **stopped** (*end*, *token limit*;
  *loops* when a three-word phrase repeats three times or more). Click a row to
  read the question and the full answer under the table.
- **Where it goes wrong (slices)**: rows and right verses per stop reason and
  for the looping answers.

These are the run's few preview prompts; the full table of an evaluation set
comes from batch evaluation (TOFIX149 item 4).

**Test run** reads the latest test run's per-row predictions (the
`test_predictions` table, published when a model is tested with Train > Test).
Its `row_id` column is each test row's row in the tested table (counted from
0), and the table's name is kept with it, so the results join back to the
data. A classifier:

- **Accuracy** and the number of wrong rows;
- **Confusion**: true classes down, predicted across (the class names come from
  the score columns), right answers green;
- **precision** and **recall** per class (recall under 60% in amber);
- **Worst rows**: the wrong rows the model was most sure of, by their **dataset
  row** (counted from 1, as the Table Viewer counts).

A regression model (its `test_predictions` has `true_value`,
`predicted_value` and `error` = predicted - true; with several targets
`true_0`, `predicted_0`, `error_0`, ...):

- **MAE**, **RMSE** and **Bias** (the mean error: positive predicts high);
- **Largest errors**: the rows furthest off, by dataset row (the first target
  when there are several).

**Error by column value** (both kinds): pick a column of the tested table and
see where the model goes wrong. Each value of the column gets its test rows and
the share wrong (a classifier) or the mean |error| (regression), highest first;
a value above the overall error by a quarter or more is amber. A number column
with more than 20 values is cut into five ranges of about equal rows; a text
column keeps its 20 most common values and puts the rest under *(other
values)*. A table on disk is read one row group at a time (a task in Task
View). It needs the tested table still loaded; when it was re-loaded with fewer
rows, the run is refused as changed since the test. A test from a source that
does not report rows (images, audio, older runs) says so: test again.
