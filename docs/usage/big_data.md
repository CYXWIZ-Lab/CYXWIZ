# Data bigger than memory

The Engine keeps a table in memory only while it fits a **data budget**. A
larger table stays a Parquet file on disk, and the steps of a Data Pipeline
read it from there: SQL steps through DuckDB, streaming steps one row group
at a time. A step that needs every row at once stops before loading and says
how much memory it would need.

## The data budget

**Preferences > General > Data > Data budget (MB)**. 0 (the default) uses half
of this PC's memory; the line under the field shows the budget in use and the
table size above which tables stay on disk (a quarter of the budget). The
setting is Engine-wide and applies at once.

The budget decides:

- **Loading**: a Data Input whose table is larger than a quarter of the budget
  is kept on disk. A Parquet file is opened in place (never copied); a CSV is
  converted once to a Parquet file in the project's `cache/ingestion` folder,
  in blocks sized from the budget. Selecting columns of a large Parquet file
  writes only those columns, one row group at a time. The Data Input's Memory
  tab option **Force disk-backed cache** keeps any file on disk.
- **Results**: a step's result larger than a quarter of the budget stays a
  Parquet file in `cache/pipeline/results`; a smaller one is in memory as
  before.
- **DuckDB**: its memory limit is the budget, it uses one thread per 256 MB of
  budget, and sorts, joins and group-bys spill to `cache/duckdb_tmp` instead
  of failing.

## How each step reads a large table

| Steps | How | Memory |
| --- | --- | --- |
| SQL step, Filter Rows, Select Columns, Sort, Join, Group By, Remove Duplicates, Fill Missing, Value Counts, Binning, Text Clean, Polynomial Features, Rule Engine, Unit Converter, Unpivot, Row Appender, Cell Extractor / Updater, String Manipulation, Math Formula, Calculator | DuckDB reads the input file in place and writes a large result to a file | within the budget |
| Standard Scaler, Log Transform | one row group at a time (the scaler fits on all rows first, then scales) | one row group |
| Checks (row count, allowed values, not empty, range) | one row group at a time; the rows pass through uncopied | one row group |
| Models, clustering, PCA, Robust Scaler, the `unique` check, and other steps | read their input whole when it fits | stop before loading otherwise |

When a step must read its input whole and it does not fit, the run stops
before loading, for example:

> K-Means: its input (5,111,808 rows) is on disk and needs about 2.0 GB of
> memory to read, over the data budget of 256 MB. Select fewer columns in
> Data Input, or raise the data budget in Preferences > Data.

## Training on a large table

A Data Input kept on disk can feed training:

- **Steps before the Data Loader** (Standard and Min-Max Scaler, Log Transform, checks) run one
  row group at a time: each fitting step reads the file once to fit, then one
  last pass writes the prepared table to a Parquet file
  (`cache/materialized/streamed` in the project) that training reads. It is
  prepared again on each run (not cached), and the previous file is removed.
  A step there that needs every row at once (a tokenizer, a vectorizer, PCA,
  the Robust Scaler) stops the run before training with the reason, for
  example:

  > 'Text Tokenizer' needs every row at once, and 'corpus' is kept on disk.
  > Before the Data Loader a table on disk runs only steps that stream
  > (Standard Scaler, Min-Max Scaler, Log Transform, checks); raise the data
  > budget (Preferences > Data) to load it in memory.

- **Causal-LM token windows** (a `tokens` list column) on disk are read one
  row group at a time while training: memory holds one row group's windows.
  The row-group order is shuffled every epoch, and the windows within each
  group. Validation and Test windows come from datasets of their own (a Data
  Split with named inputs); a ratio split of a windows file on disk is
  refused.

## Running again

A graph run again reuses every step whose settings, inputs and source files
did not change ("Using cached result"), so only the changed part runs. A step
that writes a file (an export) always runs, and so does a step after a
changed one.

## What to expect

Memory stays about the same whatever the size of the data; time grows with
it. Measured with a 256 MB budget:

| | 1 GB CSV (5.1 M rows) | 5 GB (25 M rows) | 20 GB (100 M rows) |
| --- | --- | --- | --- |
| open the Parquet file | 0.1 s | 0.0 s | 0.1 s |
| SQL filter | 0.9 s, 131 MB | 4 s, 45 MB | 33 s, 91 MB |
| group by | 0.2 s, 29 MB | 0.7 s, 28 MB | 3 s, 60 MB |
| sort every row | 21 s, 354 MB | 98 s, 358 MB | |
| convert the CSV | 11 s, 210 MB | 52 s, 216 MB | 22 min, 274 MB |

The Standard Scaler, streamed with a 1 GB budget, peaks at about 1.15 GB at all
three sizes (most of it the device). Without a cap the steps run 3-4 times
faster than before this change.

- **The GPU or CPU device needs memory of its own**: starting ArrayFire (any
  backend) takes about 500 MB on top of the data, so device steps (scalers,
  transforms, training) need that much free beyond the budget.
- **Time grows with the data**, disk space too: a sort or a join of a large
  table writes temporary files about the size of its input.
- A tool for measuring: `cyxwiz-bench-out-of-core` (in the build's `bin`
  folder) generates test tables and runs these steps under a memory cap.
