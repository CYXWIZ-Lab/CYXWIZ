# Data Studio Components in CyxWiz Studio (Unified Canvas)

These components were integrated from Data Studio into the unified CyxWiz Studio node editor.

## Execution Modes

| Mode | Purpose |
|------|---------|
| `CodeGeneration` | Generate PyTorch/TensorFlow/Keras code |
| `DuckDBPipeline` | Execute data transforms with DuckDB/Arrow |
| `LocalTraining` | Train models locally |

## Data Source Nodes

| Node | Purpose |
|------|---------|
| `CSVFile` | Load CSV file into Arrow table |
| `SQLQuery` | Execute SQL query, return Arrow table |
| `HDF5Dataset` | Load HDF5 dataset into Arrow |
| `ParquetFile` | Load Parquet file into Arrow |
| `JSONFile` | Load JSON file into Arrow |
| `ExcelFile` | Load Excel file into Arrow |
| `RESTAPISource` | Fetch data from REST API |

## Data Transform Nodes

| Node | Purpose |
|------|---------|
| `FilterRows` | Filter rows by SQL WHERE condition |
| `SelectColumns` | Select specific columns |
| `JoinTables` | Join datasets (inner/left/right/outer) |
| `GroupByAggregate` | Group by columns with aggregations |
| `SortRows` | Sort rows (asc/desc) |
| `FillMissingValues` | Handle missing (mean/median/mode/constant) |
| `RemoveDuplicateRows` | Remove duplicate rows |
| `PivotTable` | Pivot wide to long or long to wide |
| `UnionTables` | Stack datasets (UNION ALL) |
| `RenameColumns` | Rename columns |

## Analytics Nodes

| Node | Purpose |
|------|---------|
| `DescribeStats` | Statistical summary (count, mean, std) |
| `VisualizeData` | Create plots (scatter, bar, line, histogram) |
| `SampleRows` | Sample random rows |
| `CorrelationMatrix` | Compute correlation matrix |
| `ValueCounts` | Count unique values per column |
| `CrossTabulation` | Contingency table |

## Export Nodes

| Node | Purpose |
|------|---------|
| `ExportCSV` | Export to CSV |
| `ExportParquet` | Export to Parquet |
| `ExportSQL` | Write to SQL database |
| `ExportJSON` | Export to JSON |

## Backend Components

| Component | Purpose |
|-----------|---------|
| `PipelineExecutor` | DuckDB/Arrow backend for executing data pipelines |
| `Dataset` pin type | Connects data between nodes |

## Key Files

| File | Purpose |
|------|---------|
| `src/gui/node_editor.h` | NodeType enum with all data nodes |
| `src/gui/node_editor.cpp` | Node rendering and execution |
| `src/core/pipeline_executor.h/cpp` | DuckDB pipeline execution |
| `src/core/duckdb_connector.h/cpp` | DuckDB database connection |
| `src/core/arrow_dataset.h/cpp` | Arrow table handling |
