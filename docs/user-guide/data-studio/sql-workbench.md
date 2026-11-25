---
title: Write SQL steps in the SQL Workbench
description: Transform tables with one read-only SELECT per SQL step, and write, check, time and save the query in the SQL Workbench.
type: how-to
area: data-studio
audience: user
status: stable
since: 1.0.7
owner: data-studio
last_reviewed: 2026-11-25
related:
  - guided-steps.md
  - prepare.md
  - query.md
  - ../../main-window/table-viewer.md
redirect_from:
  - sql-workbench.md
---

An **SQL step** (node type *SQL Query*) transforms one or more input tables
with one read-only `SELECT`. It runs in DuckDB with no file or network
access, and its result must be flat columns that come back exactly (see
[Result types](#result-types)). Steps usually sit inside a **Preparation
Recipe**, with **Row Count Checks** after them. You write and test the query
in the **SQL Workbench**; to explore a dataset with SQL without adding a
step, use Data Studio's [Query tab](query.md) instead.

## Before you start

- A graph with the SQL step's input: a loaded Data Input, or the steps
  before it.
- An SQL step on the canvas. Add one from the Nodes panel, or start a
  guided step (see [Build a step from operations](guided-steps.md)).

## Write and save a query

1. On the canvas, select the SQL step. Properties shows the step.
2. In Properties, select **Open in SQL Workbench**. The Workbench opens as a
   dockable window; dock it beside the canvas or keep it floating.
3. Select **Load inputs**. It runs the steps before this one, so the
   Workbench has the step's real input tables. Nothing is written to the
   step yet.
4. Write the query, then press **Ctrl+Enter** to run it on the inputs. The
   first 500 rows show under **Results**.
5. Optional: select **Check** to have DuckDB check the query against the
   inputs without running it ("Syntax OK", or the error and its line).
6. Open the **Checks** tab to see the step's Check nodes run on the draft's
   result.
7. Select **Save to step**. The query becomes the step's query; graph runs
   use it from now on.

## The editor

- Highlighting, line numbers and folding. Errors point at their line.
- **Ctrl+Enter** runs the whole query on the step's inputs.
  **Ctrl+Shift+Enter** (**Run selection**) runs only the selected `SELECT`.
- **Ctrl+F** opens find in the query (Enter next, Shift+Enter previous, Esc
  closes; Match case; Regular expression).
- **Ctrl+Space** lists column names, table names and SQL words. Up and Down
  choose, Enter or Tab inserts, Esc closes.
- **Format** puts keywords in upper case and removes trailing spaces.
  **Compare with saved** shows the draft beside the step's saved query
  (**Changes only** or **Whole query**). **Reset to saved** goes back.
  Ctrl+Z undoes any of these.

## Side tabs

| Tab | What it shows |
| --- | --- |
| **Inputs** | Each input table, its rows and columns. Click a column to insert its name. **Open input in Table Viewer**. |
| **Parts** | The rows after each `WITH` part. Click a part to run the query up to it (**Run to part**); **Count rows per part** counts them all at once. |
| **History** | Runs, saves and errors in this session. Click one to bring its query back into the editor. |
| **Snippets** | Common queries (find rows, remove duplicates, ...) to insert at the cursor. |

## Results

| Tab | What it shows |
| --- | --- |
| **Results** | The first 500 rows in the [table grid](../../main-window/table-viewer.md#the-grid) (sort by a header, select and copy cells, hover a long value). **Copy shown rows** copies them in the grid's order; **Export** writes the whole result to CSV or Parquet. |
| **Column profile** | Per column: its type, missing, empty and distinct values, how many rows repeat an earlier value, and its most common values; numbers also get minimum, median, mean and maximum. |
| **Rows removed** | The rows each part dropped compared with the part before it, so you can see what a filter took out. |
| **Checks** | The step's Check nodes, run on the draft's result before you save it. |
| **Messages** | What happened, newest last. |

## See how the query runs

**Explain (estimate)** shows how DuckDB will run the query, without running
it. **Run with timing** runs it with `EXPLAIN ANALYZE` on the same inputs as
Run step and shows where the time goes.

- **Tree**: the operators in the order rows flow, each with its query part,
  estimated or actual rows, and (timed) its time and share of the total as a
  bar. Click one for its details (filters, projections, join keys).
- **Diagram**: the same operators as boxes, shaded by their share of the time.
- **Text**: DuckDB's own plan text.

After a timed run, summary cards give the time in operators, the rows out,
the operator count, the slowest operator and its share, and how many
estimates were more than 100x off. Operators taking 10% or more of the time
are marked. **Copy plan** copies DuckDB's text plan; **Export plan** writes
the JSON plan.

## The step in Properties

Select an SQL step on the canvas. Properties shows:

- **What the step does**: its name, the recipe it belongs to, and the
  query's opening notes (the `--` comment lines at the top).
- **Last run**: rows in and out, columns, time, and how many of the checks
  after it passed; "Not run in this session" until it runs.
- **Guided | SQL**: SQL shows the query with highlighting and line numbers.
  Guided explains a step written in SQL part by part; for a guided step it
  lists the operations, each with a switch to turn it on or off.
- **Run step** runs the graph up to this step, the step, and the checks
  after it. **Preview output** runs the same and shows the first rows.
- **What it does, part by part**: one line per `WITH` part (`meta`, `src`,
  `body`, ...), with the lines it covers and what it reads. Click a part to
  select its lines in the editor.
- **Input**, **Output preview** and **Checks after this step**, each with
  **Open in Table Viewer**.
- **Details**: every fact (node ID, type, SQL contract, input names, output
  dataset, outcome, parameters), Truth Summary and the parameter editor.
  `sql_contract_version` is read-only: the Engine sets it.

## Result types

| DuckDB type of a result column | Comes back as | Accepted by an SQL step |
| --- | --- | --- |
| BOOLEAN, integers (at their own width: a UTINYINT pixel stays uint8), FLOAT, DOUBLE, VARCHAR | the same | yes |
| DATE, TIMESTAMP | a date, a timestamp (microseconds) | yes |
| DECIMAL up to 15 digits (`0.5`, `round(x, 2)` of a decimal) | a number (double), exactly | yes |
| DECIMAL past 15 digits, HUGEINT, TIMESTAMP WITH TIME ZONE, BLOB, lists, structs | text | no: CAST them in the query (for example to VARCHAR, BIGINT or DOUBLE) |

Inputs keep their types too: a date, a timestamp or a decimal column going
into a step is a DATE, a TIMESTAMP or a number there, so `d + 1`, `year(t)`
and arithmetic work on them. Data Studio's Query tab shows the text column
types instead of refusing them. A table in memory goes to DuckDB in one pass
of its Arrow scan (no copy on disk); a table on disk is read from its own
file.

## Group steps into a Preparation Recipe

A **Preparation Recipe** is a subgraph that runs as one stage: one dataset
in, its steps in order, one prepared dataset out, with the same result as
running the steps as separate nodes. A **visual group** only tidies the
canvas, and its nodes do not run while it is collapsed.

1. On the canvas, select two or more step nodes.
2. Right-click the canvas and select **Actions > Create Subgraph**
   (Ctrl+Shift+U).
3. In Properties or the canvas menu, select **Make Preparation Recipe**. The
   steps are checked first; if they do not qualify, you are told why and
   nothing changes.

Select the recipe to see its role and steps in run order (each with its
latest rows or check result), **Expand** / **Collapse**, **Make visual
group** / **Make Preparation Recipe** and **Details**. Click a step to
expand the recipe and select that step. An expanded recipe is a frame on the
canvas with a **Collapse** button. Editing a recipe in Data Studio is in
[Prepare data with a recipe](prepare.md).

## Graphs without a model

A graph made only of data steps has nothing to train. Its nodes' AS
COMPILED card reads "No model to compile" instead of "Compile failed". The
Compile popup still lists what a training graph would need.

## Related

- [Build a step from operations](guided-steps.md)
- [Prepare data with a recipe](prepare.md)
- [Explore with SQL in the Query tab](query.md)
- [Table Viewer](../../main-window/table-viewer.md)
- [Explore your data](explore.md)
