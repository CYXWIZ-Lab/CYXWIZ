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

The compute device keeps its own share: starting it takes about 1 GB, so data
may use at most this PC's memory less 1 GB, whatever is set. A second line
under the field says so ("The compute device takes about 1.0 GB of this PC's
47.9 GB when it starts; data may use at most 46.9 GB."), and a budget above
that is used as the most, with a warning ("58.6 GB is more than that: data
gets 46.9 GB.").

The budget decides:

- **Loading**: a Data Input whose table is larger than a quarter of the budget
  is kept on disk. A Parquet file is opened in place (never copied); a CSV is
  converted once to a Parquet file in the project's `cache/ingestion` folder,
  in blocks sized from the budget. Selecting columns of a large Parquet file
  writes only those columns, one row group at a time. The Data Input's Memory
  tab option **Force disk-backed cache** keeps any file on disk.
- **One copy per file**: a table in memory is held once. When the same file is
  loaded again with the same options (Apply on the Data Input, then a run of
  the graph, or the other way round), the second load shares the table that is
  already there ("Shared with ... (already loaded)" in the log) instead of
  reading the file again. A changed file, or other options (columns, rows,
  delimiter, missing values), is read again.
- **Results**: a step's result larger than a quarter of the budget stays a
  Parquet file in `cache/pipeline/results`; a smaller one is in memory as
  before.
- **DuckDB**: its memory limit is the budget, it uses one thread per 256 MB of
  budget, and sorts, joins and group-bys spill to `cache/duckdb_tmp` instead
  of failing. Each DuckDB spills to a folder of its own there, removed when it
  closes.
- **Branches**: steps on different branches of a graph run at the same time
  (up to four at once): a step starts as soon as the steps it reads are done.
  Steps running at once share the budget: with two running, each gets half
  for DuckDB and for reading a table on disk whole. A step alone has all of
  it. When a step fails, nothing new starts; the steps already running finish
  and the run reports the failing step's error. Two branches reading the same
  table share one copy, and two exports to the same file write it one after
  the other.

## How each step reads a large table

| Steps | How | Memory |
| --- | --- | --- |
| SQL step, Filter Rows, Select Columns, Sort, Join, Group By, Remove Duplicates, Fill Missing, Value Counts, Binning, Text Clean, Polynomial Features, Rule Engine, Unit Converter, Unpivot, Row Appender, Cell Extractor / Updater, String Manipulation, Math Formula, Calculator | DuckDB reads the input file in place and writes a large result to a file | within the budget |
| Standard Scaler, Log Transform | one row group at a time (the scaler fits on all rows first, then scales) | one row group |
| Checks (row count, allowed values, not empty, range) | one row group at a time; the rows pass through uncopied | one row group |
| The `unique` check | one DuckDB query (group by the columns) over the file; the rows pass through uncopied | within the budget |
| Export CSV / JSON / Parquet, Data Output, and Export in the SQL Workbench | write the table one row group at a time (a table in memory in one go), hashing the file as it is written for the lineage | one row group |
| Models, clustering, PCA, Robust Scaler, and other steps | read their input whole when it fits | stop before loading otherwise |

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

- **A table on disk** trains a row group at a time, with the Data Split
  made row by row as in memory: without stratification Train is the first
  rows of the file, then Validation, then Test; with **Stratified** each
  class gets its share (in file order, or drawn by the split seed when
  shuffling). The split takes one byte of memory per row (100 MB for 100
  million rows). Class balancing in the Data Loader is not available on disk
  yet (a warning says so).
- **Causal-LM token windows** (a `tokens` list column) on disk are read one
  row group at a time while training: memory holds one row group's windows.
  The row-group order is shuffled every epoch, and the windows within each
  group. Validation and Test windows come from datasets of their own (a Data
  Split with named inputs). A share left to cut from the training set (for
  example a 10% test share beside a Validation input) is refused on disk with
  the reason; set it to 0 or give it an input of its own.
- **A server node** (remote training) does the same: a Parquet file a job
  brings that is larger than a quarter of the node's data budget stays on disk
  and is read a row group at a time, for tables and for token windows.

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
  backend) takes about 0.5-0.9 GB on top of the data. The budget never takes
  the last 1 GB of this PC's memory, so the device always has room to start.
- **Time grows with the data**, disk space too: a sort or a join of a large
  table writes temporary files about the size of its input.
- A tool for measuring: `cyxwiz-bench-out-of-core` (in the build's `bin`
  folder) generates test tables and runs these steps under a memory cap.
