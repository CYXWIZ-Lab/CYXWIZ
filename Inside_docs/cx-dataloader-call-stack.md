# cx.DataLoader Call Stack & Backend Flow

This document walks through what happens when a Python script invokes `cx.DataLoader` (e.g., `cx.DataLoader().load_csv("data.csv")`). The sequence spans four layers: Python <-> PyBind11 bindings, the C++ `DataLoader` wrapper, the embedded DuckDB engine, and the conversion back into a `cyxwiz::Tensor`. The diagram below shows the high-level flow; the sections afterward name the key steps and reference the relevant C++ functions.

```
Python app
   |
   v
pycyxwiz (
  PyBind11 binding
)
   |
   v
cyxwiz::DataLoader::LoadCSV
   |
   v
duckdb_query -> DuckDB
   |
   v
duckdb_result -> ResultToTensor()
   |
   v
Tensor -> PyBind11 -> Python
```

## 1. Python & PyBind11 (`cyxwiz-backend/python/bindings.cpp`)

1. Python calls `cx.DataLoader().load_csv(...)`.
2. PyBind11 exposes that call in the `pycyxwiz` module; see the `load_csv` lambda at `cyxwiz-backend/python/bindings.cpp:2368-2382`.
3. The lambda forwards the arguments (path, columns, delimiter, header) into the C++ `DataLoader::LoadCSV`.
4. Control returns to Python once the `Tensor` produced by the C++ layer is converted back through PyBind11.

## 2. C++ `DataLoader` (`cyxwiz-backend/src/core/data_loader.cpp`)

1. The constructor `DataLoader::DataLoader()` (lines 34‑48) allocates an in‑memory DuckDB instance and opens a connection (`Initialize()`).
2. `LoadCSV()` (lines 260‑303):
   - Normalizes the path (`NormalizePath`).
   - Builds a SQL query that wraps `read_csv(...)` with the requested columns and delimiter.
   - Logs the SQL when `config_.verbose` is true.
   - Calls `duckdb_query(connection_, sql)` to execute everything inside DuckDB.
3. Successful queries produce a `duckdb_result`. `ResultToTensor()` (lines 121‑214) walks the DuckDB chunks, reads each column vector, handles NULL validity, casts numeric types to `float`, and fills a contiguous row-major `std::vector<float>` buffer.
4. The helper then constructs a `cyxwiz::Tensor` using that buffer and returns it to the caller.

## 3. DuckDB & SQL execution

Inside `LoadCSV`/`LoadParquet`/`Query` the flow is:

- The generated SQL is submitted via `duckdb_query`. DuckDB parses the SQL, plans it, and streams data out in `duckdb_data_chunk`s.
- `ResultToTensor()` repeatedly calls `duckdb_result_get_chunk`, converts each chunk to native floats, and destroys the chunk when done.
- On failure, the `duckdb_result` is destroyed and an exception is thrown up the stack; PyBind11 converts that to a Python exception.

`BatchIterator` (lines 586‑825) reuses the same `duckdb_query` path but adds `LIMIT/OFFSET` clauses to stream large datasets without exhausting memory. Each `Next()` call does a fresh query for the next offset, converts the chunk(s) via the same chunk-to-tensor loop, and increments `current_batch_`.

## 4. Summary of matching code sites

- PyBind interface: `cyxwiz-backend/python/bindings.cpp:2338-2460`.
- DataLoader implementation: `cyxwiz-backend/src/core/data_loader.cpp:34-825`.
- DuckDB bridging: `duckdb_query`, `duckdb_result_get_chunk`, and the chunk iteration inside `ResultToTensor()`.
- Batch iteration: `DataLoader::BatchIterator::Next()` + `HasNext()` to limit/offset queries.

If you want a sequence diagram for a specific helper (e.g., `QueryColumns` returning per-column tensors), let me know and I can add it.
