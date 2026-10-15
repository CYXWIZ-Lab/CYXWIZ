# SQL steps, the SQL Workbench and guided steps

An **SQL step** (node type *SQL Query*) transforms one or more input tables
with one read-only `SELECT`. It runs in DuckDB with no file or network
access, and its result must be flat columns that come back exactly (see
**Result types** below).
Steps are usually kept inside a **Preparation Recipe** (below), with **Row
Count Checks** after them.

## Result types

| DuckDB type of a result column | Comes back as | Accepted by an SQL step |
|---|---|---|
| BOOLEAN, integers, FLOAT, DOUBLE, VARCHAR | the same | yes |
| DATE, TIMESTAMP | a date, a timestamp (microseconds) | yes |
| DECIMAL up to 15 digits (`0.5`, `round(x, 2)` of a decimal) | a number (double), exactly | yes |
| DECIMAL past 15 digits, HUGEINT, TIMESTAMP WITH TIME ZONE, BLOB, lists, structs | text | no: CAST them in the query (for example to VARCHAR, BIGINT or DOUBLE) |

Inputs keep their types too: a date, a timestamp or a decimal column going
into a step is a DATE, a TIMESTAMP or a number there, so `d + 1`, `year(t)`
and arithmetic work on them. Data Studio's Query tab shows the text column
types instead of refusing them.

## In Properties

Select an SQL step on the canvas. Properties shows:

- **What the step does**: its name, the recipe it belongs to, and the
  query's opening notes (the `--` comment lines at the top).
- **Last run**: rows in and out, columns, time, and how many of the checks
  after it passed. "Not run in this session" until it runs.
- **Guided | SQL**: SQL shows the query with highlighting and line numbers.
  Guided explains a step written in SQL part by part. For a guided step, it
  lists its operations, each with a switch to turn it on or off.
- **Run step** runs the graph up to this step, the step, and the checks after
  it. **Preview output** runs the same thing and shows the first rows here.
- **What it does, part by part**: one line per `WITH` part (`meta`, `src`,
  `body`, ...), with the lines it covers and what it reads. Click a part to
  select its lines in the editor.
- **Input**, **Output preview** and **Checks after this step**, each with
  **Open in Table Viewer**.
- **Details**: every fact (node ID, type, SQL contract, input names, output
  dataset, outcome, parameters), the general sections, Truth Summary and the
  parameter editor. `sql_contract_version` is shown read-only: the engine
  sets it.

## The SQL Workbench

Open it with **Open in SQL Workbench** in Properties. It opens as a dockable
window; dock it beside the canvas or keep it floating.

**Load inputs** first: it runs the steps before this one, so the Workbench has
the step's real input tables. Nothing is written until **Save to step**.

### The editor

- Highlighting, line numbers and folding. Errors point at their line.
- **Ctrl+Enter** runs the whole query on the step's inputs. **Ctrl+Shift+Enter**
  (**Run selection**) runs only the selected `SELECT`.
- **Check** asks DuckDB to check the query against the inputs without running
  it; "Syntax OK" or the error and its line.
- **Ctrl+F** opens find in the query (Enter next, Shift+Enter previous, Esc
  closes; Match case; Regular expression).
- **Ctrl+Space** lists column names, table names and SQL words. Up and Down
  choose, Enter or Tab inserts, Esc closes.
- **Format** puts keywords in upper case and removes trailing spaces.
  **Compare with saved** shows the draft beside the step's saved query
  (**Changes only** or **Whole query**). **Reset to saved** goes back. Ctrl+Z
  undoes any of these.

### Side tabs

- **Inputs**: each input table, its rows and columns. Click a column to insert
  its name. **Open input in Table Viewer**.
- **Parts**: the rows after each part. Click a part to run the query up to it
  (**Run to part**). **Count rows per part** counts them all at once.
- **History**: runs, saves and errors in this session. Click one to bring its
  query back into the editor.
- **Snippets**: common queries (find rows, remove duplicates, ...) to insert
  at the cursor.

### Results

- **Results**: the first 500 rows in the table grid (sort by a header,
  select and copy cells, hover a long value; see
  [Table Viewer and the table grid](table_viewer.md)). **Copy shown rows**
  copies them in the grid's order; **Export** writes the whole result to CSV
  or Parquet.
- **Column profile**: per column, its type, missing, empty and distinct
  values, how many rows repeat an earlier value, and its most common values;
  numbers also get minimum, median, mean and maximum.
- **Rows removed**: the rows each part dropped compared with the part before
  it, so you can see what a filter took out.
- **Checks**: the step's Check nodes, run on the draft's result before
  you save it.
- **Messages**: what happened, newest last.

### Plan

**Explain (estimate)** shows how DuckDB will run the query, without running
it (instant). **Run with timing** runs it with `EXPLAIN ANALYZE` on the same
inputs as Run step and shows where the time goes.

- **Tree**: the operators in the order rows flow, each with its query part,
  estimated or actual rows, and (timed) its time and share of the total as a
  bar. Click one for its details (filters, projections, join keys).
- **Diagram**: the same operators as boxes, shaded by their share of the time.
- **Text**: DuckDB's own plan text.

After a timed run, summary cards give the time in operators, the rows out,
the operator count, the slowest operator and its share, and how many
estimates were more than 100x off. Operators taking 10% or more of the time are marked.
**Copy plan** copies DuckDB's text plan; **Export plan** writes the JSON
plan.

## Guided steps

A guided step is an SQL step built from operations instead of typed SQL. In
the Workbench, choose **Guided**, then **Start a guided step instead** (or
**Start from a template**). Data Studio > Prepare edits guided steps in place
with the same editor ([Data Studio](data_studio.md#prepare)).

- **Operations**, in five groups:
  - Rows: Keep rows where, Remove duplicate rows, Sort rows, Keep first /
    sample rows.
  - Columns: Choose and rename columns, Add a calculated column, Change type,
    Split or merge columns.
  - Text: Clean text, Find and replace, Keep text between markers, Split text
    into rows, Extract with a pattern.
  - Combine and summarise: Look up from another table, Append rows, Group and
    summarise, Pivot / unpivot.
  - Splits and checks: Assign train / validation / test, Number rows, Add
    checks.
- Each operation is a card. Select it to change it in plain fields (columns
  from the real input). Switch it off without deleting it, or move it up or
  down. Click an operation in the library to add it after the selected one.
- **Templates**:
  - Split documents into paragraphs.
  - Clean a text column.
  - Look up and keep matched rows.
  - Train / validation / test split.
- **Preview on sample** runs the operations on the first 1,000 rows (or a
  random 1,000, or all rows) and shows the rows after each operation. **Run
  step** (Ctrl+Enter) uses all rows.
- The generated SQL is shown read-only, one commented part per operation.
  **Edit as SQL** turns the step into an ordinary SQL step: it asks first,
  because the operations are not kept.
- An operation with a problem (a missing column, a bad pattern) is marked in
  red with the reason, and the step cannot run until it is fixed.
- **Add checks** goes last. While you edit, its checks run with the preview
  and a failing one stops it with its message. When the step is saved, each
  check becomes a **Check** node right after the step (row count,
  allowed values, unique, not empty) and the operation leaves the step; Ctrl+Z
  on the canvas undoes it. Placed before other operations, it is marked in red.

The step stores its operations (`guided_ops`) and the SQL they generate
(`query`). Graphs and runs use the SQL, so a guided step runs exactly like any
SQL step. In Properties, a guided step lists its operations with on / off
switches and **Edit in guided editor**.

## Preparation Recipes

A **Preparation Recipe** is a subgraph that runs as one stage: one dataset
in, its steps in order, one prepared dataset out. The result is the same as
running the steps as separate nodes. A **visual group** only tidies the
canvas, and its nodes do not run while it is collapsed.

- Make one: select two or more nodes, right-click the canvas, then **Actions >
  Create Subgraph** (Ctrl+Shift+U). Then choose **Make Preparation Recipe**,
  either in Properties or in the canvas menu. The steps are checked first. If
  they do not qualify, you are told why and nothing changes.
- Select the recipe to see in Properties:
  - its role and steps in run order, each with its latest rows or check
    result;
  - **Expand** / **Collapse**;
  - **Make visual group** / **Make Preparation Recipe**;
  - **Details**.
- Click a step to expand the recipe and select that step.
- An expanded recipe is a frame on the canvas with a **Collapse** button.

## Graphs without a model

A graph made only of data steps has nothing to train. Its nodes' AS COMPILED
card reads "No model to compile" instead of "Compile failed". The Compile
popup still lists what a training graph would need.
