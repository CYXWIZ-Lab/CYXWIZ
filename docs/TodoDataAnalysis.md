# DuckDB + Polars Integration Plan

## Overview
Integrate DuckDB (C++ backend) and Polars (Python scripting) for high-performance data analysis in CyxWiz.

## Implementation Order

### Phase 1: Build System (15 min)
1. **vcpkg.json** - Add `"duckdb"` to dependencies
2. **cyxwiz-backend/CMakeLists.txt** - Add DuckDB detection:
   ```cmake
   find_package(DuckDB CONFIG QUIET)
   if(DuckDB_FOUND)
       set(CYXWIZ_HAS_DUCKDB ON)
       target_link_libraries(cyxwiz-backend PRIVATE duckdb)
       target_compile_definitions(cyxwiz-backend PRIVATE CYXWIZ_HAS_DUCKDB)
   endif()
   ```

### Phase 2: C++ DataLoader Class (4-5 hours)
3. **cyxwiz-backend/include/cyxwiz/data_loader.h** - Header with:
   - `DataLoaderConfig` struct (batch_size, memory_limit_mb, num_threads)
   - `ColumnInfo` struct for schema inspection
   - `DataLoader` class with:
     - `LoadParquet(path, columns)` → Tensor
     - `LoadCSV(path, columns, delimiter, header)` → Tensor
     - `Query(sql)` → Tensor
     - `GetSchema(path)`, `GetColumns(path)`, `GetRowCount(path)`
     - `CreateBatchIterator(sql, batch_size)` → BatchIterator
     - `ConvertCSVToParquet(csv, parquet, compression)`
   - `BatchIterator` class with HasNext/Next/Reset

4. **cyxwiz-backend/src/core/data_loader.cpp** - Implementation:
   - DuckDB C API integration (duckdb_open, duckdb_connect, duckdb_query)
   - `ResultToTensor()` - Convert DuckDB result chunks to Tensor
   - Handle type conversion (FLOAT, DOUBLE, INTEGER, BIGINT → float)
   - BatchIterator with LIMIT/OFFSET pagination
   - All wrapped in `#ifdef CYXWIZ_HAS_DUCKDB`

### Phase 3: Python Bindings (1 hour)
5. **cyxwiz-backend/python/bindings.cpp** - Add:
   ```cpp
   py::class_<cyxwiz::DataLoader>(m, "DataLoader")
       .def("load_parquet", ...)
       .def("load_csv", ...)
       .def("query", ...)  // with py::gil_scoped_release
       .def("create_batch_iterator", ...)
       .def_static("is_available", ...);

   py::class_<cyxwiz::DataLoader::BatchIterator>(m, "BatchIterator")
       .def("__iter__", ...)
       .def("__next__", ...);  // Python iterator protocol
   ```

### Phase 4: Python Sandbox (15 min)
6. **cyxwiz-engine/src/scripting/python_sandbox.h** - Add to `allowed_modules`:
   ```cpp
   "polars", "duckdb", "pyarrow",
   "pyarrow.parquet", "pyarrow.csv",
   ```

7. **cyxwiz-engine/python/requirements.txt** (new file):
   ```
   polars[pyarrow]>=1.3.0
   duckdb>=1.2.0
   pyarrow>=15.0.0
   ```

### Phase 5: Testing (2 hours)
8. **tests/unit/test_data_loader.cpp** - C++ tests for:
   - IsAvailable(), construction
   - LoadCSV, Query, BatchIterator

9. **cyxwiz-backend/examples/python/data_loader_example.py** - Python examples

---

## Critical Files
| File | Purpose |
|------|---------|
| `vcpkg.json` | Add duckdb dependency |
| `cyxwiz-backend/CMakeLists.txt` | CYXWIZ_HAS_DUCKDB flag |
| `cyxwiz-backend/include/cyxwiz/data_loader.h` | DataLoader API |
| `cyxwiz-backend/src/core/data_loader.cpp` | DuckDB implementation |
| `cyxwiz-backend/python/bindings.cpp` | Python bindings |
| `cyxwiz-engine/src/scripting/python_sandbox.h` | Allow polars/duckdb |

---

## Key Implementation Details

### DuckDB Result → Tensor Conversion
```cpp
Tensor ResultToTensor(duckdb_result* result) {
    // Use duckdb_fetch_chunk() for streaming
    // Handle types: DUCKDB_TYPE_FLOAT, DOUBLE, INTEGER, BIGINT
    // Check validity mask for NULLs
    // Return Tensor with shape [rows, cols]
}
```

### BatchIterator Strategy
- Use `LIMIT X OFFSET Y` for pagination (simpler than streaming API)
- Pre-calculate total rows with `COUNT(*)` subquery
- Python iterator protocol via `__iter__`/`__next__`

### GIL Release for Long Queries
```cpp
.def("query", [](DataLoader& self, const std::string& sql) {
    py::gil_scoped_release release;
    return self.Query(sql);
})
```

---

## Pitfalls to Avoid
1. **Type conversion** - Handle all DuckDB types, document unsupported ones
2. **Memory** - Warn if file > memory_limit, recommend BatchIterator
3. **Paths** - Normalize to forward slashes, quote in SQL
4. **GIL** - Release during long operations to avoid freezing UI
5. **Polars imports** - May need additional submodules in whitelist

---

## Estimated Time: ~8-10 hours total

## Status
- [x] Phase 1: Build System (vcpkg.json + CMakeLists.txt)
- [x] Phase 2: C++ DataLoader Class (data_loader.h + data_loader.cpp)
- [x] Phase 3: Python Bindings (bindings.cpp)
- [x] Phase 4: Python Sandbox (python_sandbox.h + requirements.txt)
- [ ] Phase 5: Testing (pending first build with DuckDB)
