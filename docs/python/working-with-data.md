---
title: Work with data in pycyxwiz
description: Bring data into pycyxwiz from NumPy, CSV and Parquet files through the DuckDB loader, SQL queries, batches and pandas, and reach the data the Engine works with.
type: how-to
area: python
audience: user
status: experimental
since: 1.0.8
owner: python
last_reviewed: 2026-11-25
related:
  - tensors-and-devices.md
  - reference/data-loading.md
  - engine-and-standalone.md
  - ../user-guide/scripting/variable-explorer.md
  - ../concepts/loaded-data.md
---

Use this when you get a dataset into tensors. Every layer and loss takes a
`pycyxwiz.Tensor`; the ways in are NumPy (`Tensor.from_numpy`) and the
DuckDB loader `pycyxwiz.DataLoader`, which reads CSV, Parquet and JSON files
and runs SQL over them.

> Checked against pycyxwiz 0.2.0. The samples on this page run, in order,
> as one script, in a folder that holds the repository's
> `examples/datasets` (Engine packages from 1.0.8 ship the same files in
> their `examples` folder): change the paths to where your copy is.

## From NumPy

```python
import numpy as np
import pycyxwiz as cx

cx.initialize()

data = np.loadtxt("examples/datasets/digits_8x8.csv", delimiter=",", skiprows=1)
X = cx.Tensor.from_numpy((data[:, :64] / 16.0).astype(np.float32))   # features as float32
y = cx.Tensor.from_numpy(data[:, 64].astype(np.int64))               # class indices as int64
print(X.shape(), X.get_data_type(), y.get_data_type())   # [1797, 64] DataType.Float32 DataType.Int64
```

Give features as `float32`: `np.loadtxt` and pandas give `float64`, and a
`float64` tensor stays `float64` while the layers compute in `float32`.
Give class labels as `int64` (or `int32`).

## From files, with the DuckDB loader

`DataLoader` reads files with DuckDB, built into pycyxwiz when the build
includes it (`cx.duckdb_available()` says whether), and returns tensors.

```python
loader = cx.DataLoader()
path = "examples/datasets/digits_8x8.csv"
print(loader.get_row_count(path), len(loader.get_columns(path)))   # 1797 65
print([(c.name, c.type) for c in loader.get_schema(path)][-1])     # ('class', 'BIGINT')

everything = loader.load_csv(path)
pixels = loader.load_csv(path, columns=["pixel_0", "pixel_1", "pixel_2"])
print(everything.shape(), pixels.shape())    # [1797, 65] [1797, 3]
```

A loaded tensor is `float32`, one column per file column.

**Text columns become 0.** `load_csv` gives every value that is not a
number as 0, without a warning: the `Month` column of
`airline_passengers.csv` loads as zeros. Leave such columns out (with
`columns=` or SQL) and turn text into numbers yourself.

### Ask with SQL

`query` runs any DuckDB SQL; name a file in quotes as a table:

```python
counts = loader.query(
    "SELECT class, COUNT(*) AS n FROM 'examples/datasets/digits_8x8.csv' GROUP BY class ORDER BY class")
print(counts.to_numpy()[:3].tolist())        # [[0.0, 178.0], [1.0, 182.0], [2.0, 177.0]]

features = loader.query("SELECT * EXCLUDE (class) FROM 'examples/datasets/digits_8x8.csv'")
print(features.shape())                      # [1797, 64]

passengers = loader.query("SELECT Passengers FROM 'examples/datasets/airline_passengers.csv'")
print(passengers.shape(), passengers.at(0, 0))   # [144, 1] 112.0
```

`query_columns(sql)` gives one tensor per column instead.

### Read in batches

For a file larger than you want in memory at once, iterate a query in
batches of rows:

```python
seen = 0
for batch in loader.create_batch_iterator("SELECT * FROM 'examples/datasets/digits_8x8.csv'", 512):
    seen += batch.shape()[0]                 # [512, 65], ... and the rest
print(seen)                                  # 1797
```

The loader's settings (`DataLoaderConfig`: `batch_size`, `memory_limit_mb`,
`num_threads`) are read with `get_config()` and changed with
`set_config(config)`.

### Parquet and JSON

```python
loader.convert_csv_to_parquet("examples/datasets/digits_8x8.csv", "digits.parquet")
print(loader.load_parquet("digits.parquet").shape())   # [1797, 65]
```

`load_json(path)` and `convert_json_to_parquet(json_path, parquet_path)`
work the same way.

## From pandas

pandas is not part of pycyxwiz; install it into the project's environment
([Python and project environments](../user-guide/scripting/python-environment.md)).
Go through NumPy with the types set:

<!-- not run: needs pandas, which the docs check does not install -->
```python
import pandas as pd

df = pd.read_csv("examples/datasets/digits_8x8.csv")
X = cx.Tensor.from_numpy(df.drop(columns="class").to_numpy(dtype=np.float32) / 16.0)
y = cx.Tensor.from_numpy(df["class"].to_numpy(dtype=np.int64))
print(X.shape(), y.get_data_type())          # [1797, 64] DataType.Int64
```

Polars works the same way (`df.to_numpy()`). In the Engine's Python
session, DuckDB and Polars come with short helpers (`sql()`, `read_csv()`,
`df()`) when they are installed; see
[Run in the Engine or standalone](engine-and-standalone.md).

## The data the Engine works with

pycyxwiz does not reach into the Engine's windows: a dataset loaded in Data
Studio or read by a graph's Data Input is not a Python object. To use the
same data in a script, read the same file, by the path the Data Input or
Data Studio shows. The other direction works: arrays, tensors and
DataFrames a script makes are listed in the
[Variable Explorer](../user-guide/scripting/variable-explorer.md), which
opens them in the Table Viewer.

## Other kinds of data

- **Text**: `cx.Tokenizer` and `cx.Vocabulary` turn text into token ids
  ([Tokenizers](reference/tokenizers.md)); `pycyxwiz.text` splits words and
  scores sentiment ([Signals, statistics and text](signals-statistics-and-text.md)).
- **Audio**: `cx.AudioProcessing` loads audio and computes spectrograms,
  mel spectrograms and MFCCs ([Audio](reference/audio.md)).
- **Time series**: `pycyxwiz.timeseries.create_windows` makes sliding
  windows ([pycyxwiz.timeseries](reference/timeseries.md)).

## Related

- [Data loading reference](reference/data-loading.md): every member of
  `DataLoader`, `DataLoaderConfig`, `ColumnInfo` and `BatchIterator`.
- Example datasets: `examples/datasets/README.md` lists the committed
  samples and their sources.
