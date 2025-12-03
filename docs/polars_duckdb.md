# Polars and DuckDB Integration Architecture

This document describes the architecture for integrating DuckDB and Polars into CyxWiz for high-performance data processing.

## Overview

CyxWiz uses a dual-library approach for data processing:
- **DuckDB**: Embedded OLAP database for SQL queries and file I/O (C++ backend)
- **Polars**: High-performance DataFrame library for transformations (Python scripting)

This combination provides the best of both worlds: SQL's expressiveness for complex queries and Polars' speed for data transformations.

## Technology Comparison

| Feature | DuckDB | Polars |
|---------|--------|--------|
| **Type** | Embedded OLAP database | DataFrame library |
| **Core Language** | C++ | Rust |
| **Query Interface** | SQL + DataFrame API | DataFrame API + SQL |
| **vcpkg Available** | Yes (v1.2.2) | No (Rust-only) |
| **C++ API** | Stable C API | No official bindings |
| **Python Bindings** | Good | Excellent |
| **Larger-than-Memory** | Native spillover to disk | Limited (lazy evaluation) |
| **File Formats** | Parquet, CSV, JSON, Arrow | Parquet, CSV, JSON, Arrow, IPC |
| **Arrow Integration** | Zero-copy (limited in C++) | Zero-copy (Python) |

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CyxWiz Data Pipeline                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │              Data Sources (Parquet, CSV, JSON)                │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                  cyxwiz-backend (C++)                         │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │                   DuckDB C API                          │ │ │
│  │  │  • Direct Parquet/CSV file queries                      │ │ │
│  │  │  • SQL-based preprocessing and filtering                │ │ │
│  │  │  • Larger-than-memory dataset support                   │ │ │
│  │  │  • Batch data loading for training                      │ │ │
│  │  │  • Convert query results → CyxWiz Tensor                │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                    Apache Arrow Interface                           │
│                              │                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │               cyxwiz-engine (Python Scripting)                │ │
│  │  ┌──────────────────────┐  ┌──────────────────────────────┐ │ │
│  │  │   Polars (pip)       │  │   DuckDB Python (pip)        │ │ │
│  │  │  • DataFrame API     │  │  • SQL queries in scripts    │ │ │
│  │  │  • Feature engineer. │  │  • Query Polars DataFrames   │ │ │
│  │  │  • Lazy evaluation   │  │  • Hybrid workflows          │ │ │
│  │  │  • Fast transforms   │  │  • Ad-hoc analysis           │ │ │
│  │  └──────────────────────┘  └──────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                  CyxWiz Tensor (ArrayFire)                    │ │
│  │  • GPU/CPU compute for ML training                            │ │
│  │  • Model forward/backward passes                              │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

#### DuckDB (C++ Backend)

**Primary Role**: Data ingestion and SQL-based preprocessing

**Responsibilities**:
1. Load datasets from Parquet, CSV, JSON files
2. Execute SQL queries for filtering, aggregation, joins
3. Handle larger-than-memory datasets with automatic spillover
4. Convert query results to CyxWiz Tensors for training
5. Provide batch iterators for streaming large datasets

**Why DuckDB in Backend**:
- Available in vcpkg with stable C API
- Native larger-than-memory support
- Direct file queries without full load
- Integrates cleanly with CMake build system

#### Polars (Python Scripting)

**Primary Role**: Feature engineering and data transformations

**Responsibilities**:
1. DataFrame-style transformations in Python scripts
2. Feature engineering pipelines with lazy evaluation
3. Fast in-memory operations (5x faster than Pandas)
4. Interop with DuckDB for hybrid workflows
5. Export to NumPy/CyxWiz Tensors

**Why Polars in Engine**:
- No official C++ bindings (Rust-native)
- Excellent Python API via pybind11
- Lazy evaluation optimizes before execution
- Zero-copy Arrow integration with DuckDB

## Integration Details

### Build System Integration

#### vcpkg.json

```json
{
  "name": "cyxwiz",
  "dependencies": [
    "duckdb",
    // ... other dependencies
  ]
}
```

#### cyxwiz-backend/CMakeLists.txt

```cmake
# Find DuckDB
find_package(DuckDB CONFIG)
if(DuckDB_FOUND)
    message(STATUS "DuckDB found - Data loading enabled")
    set(CYXWIZ_HAS_DUCKDB ON)
else()
    message(WARNING "DuckDB not found - Using fallback data loading")
    set(CYXWIZ_HAS_DUCKDB OFF)
endif()

# Link DuckDB if found
if(DuckDB_FOUND)
    find_package(Threads REQUIRED)
    target_link_libraries(cyxwiz-backend PRIVATE duckdb Threads::Threads)
    target_compile_definitions(cyxwiz-backend PRIVATE CYXWIZ_HAS_DUCKDB)
endif()
```

#### Python Requirements

```txt
# cyxwiz-engine/python/requirements.txt
polars[pyarrow]>=1.3.0
duckdb>=1.2.0
pyarrow>=15.0.0
```

### C++ API Design

#### DataLoader Class

```cpp
// cyxwiz-backend/include/cyxwiz/data_loader.h
#pragma once

#ifdef CYXWIZ_HAS_DUCKDB
#include <duckdb.h>
#endif

#include <memory>
#include <vector>
#include <string>
#include <optional>
#include "cyxwiz/tensor.h"

namespace cyxwiz {

struct DataLoaderConfig {
    size_t batch_size = 1024;
    size_t memory_limit_mb = 4096;  // DuckDB memory limit
    int num_threads = 0;            // 0 = auto
};

class DataLoader {
public:
    DataLoader(const DataLoaderConfig& config = {});
    ~DataLoader();

    // File Loading
    Tensor LoadParquet(const std::string& path,
                       const std::vector<std::string>& columns = {});
    Tensor LoadCSV(const std::string& path,
                   const std::vector<std::string>& columns = {},
                   char delimiter = ',',
                   bool header = true);

    // SQL Queries
    Tensor Query(const std::string& sql);

    // Schema Inspection
    std::vector<std::string> GetColumns(const std::string& path);
    size_t GetRowCount(const std::string& path);

    // Batch Iterator for Large Datasets
    class BatchIterator {
    public:
        bool HasNext() const;
        Tensor Next();
        void Reset();
        size_t TotalBatches() const;
        size_t CurrentBatch() const;
    private:
        friend class DataLoader;
        // Implementation details
    };

    BatchIterator CreateBatchIterator(const std::string& sql, size_t batch_size);

    // Direct File Operations
    void ConvertCSVToParquet(const std::string& csv_path,
                             const std::string& parquet_path,
                             const std::string& compression = "zstd");

private:
#ifdef CYXWIZ_HAS_DUCKDB
    duckdb_database db_;
    duckdb_connection con_;
#endif
    DataLoaderConfig config_;

    Tensor ResultToTensor(void* result);
    void InitializeDatabase();
};

} // namespace cyxwiz
```

#### Usage Example (C++)

```cpp
#include <cyxwiz/data_loader.h>
#include <cyxwiz/tensor.h>

void LoadTrainingData() {
    cyxwiz::DataLoaderConfig config;
    config.memory_limit_mb = 8192;  // 8GB
    config.batch_size = 2048;

    cyxwiz::DataLoader loader(config);

    // Load entire dataset (fits in memory)
    auto data = loader.LoadParquet("data/train.parquet",
                                   {"feature1", "feature2", "label"});

    // SQL query with filtering
    auto filtered = loader.Query(R"(
        SELECT feature1, feature2, label
        FROM 'data/train.parquet'
        WHERE label IS NOT NULL
          AND timestamp >= '2024-01-01'
    )");

    // Batch loading for large datasets
    auto iterator = loader.CreateBatchIterator(
        "SELECT * FROM 'huge_dataset.parquet'",
        config.batch_size
    );

    while (iterator.HasNext()) {
        auto batch = iterator.Next();
        // Process batch...
    }
}
```

### Python API Design

#### Polars Integration

```python
# Example: Feature Engineering Pipeline
import polars as pl
import pycyxwiz

def create_features(input_path: str) -> pycyxwiz.Tensor:
    """Build feature engineering pipeline with Polars lazy evaluation."""

    # Lazy scan (doesn't load data yet)
    lazy_df = pl.scan_parquet(input_path)

    # Chain transformations (lazy - not executed)
    features = (
        lazy_df
        # Filter
        .filter(pl.col('timestamp') >= '2024-01-01')
        .filter(pl.col('label').is_not_null())

        # Temporal features
        .with_columns([
            pl.col('timestamp').dt.hour().alias('hour'),
            pl.col('timestamp').dt.day_of_week().alias('day_of_week'),
            pl.col('timestamp').dt.month().alias('month')
        ])

        # Numerical features
        .with_columns([
            (pl.col('price') * pl.col('quantity')).alias('total_value'),
            ((pl.col('value') - pl.col('value').mean()) /
             pl.col('value').std()).alias('value_normalized')
        ])

        # Select final columns
        .select([
            'hour', 'day_of_week', 'month',
            'total_value', 'value_normalized', 'label'
        ])
    )

    # Execute and collect
    df = features.collect()

    # Convert to CyxWiz Tensor
    numpy_array = df.to_numpy()
    return pycyxwiz.Tensor.from_numpy(numpy_array)
```

#### DuckDB + Polars Hybrid

```python
# Example: Hybrid Workflow
import duckdb
import polars as pl
import pycyxwiz

def hybrid_preprocessing(data_path: str) -> pycyxwiz.Tensor:
    """Combine DuckDB SQL with Polars transformations."""

    # Step 1: DuckDB for complex SQL (larger-than-memory safe)
    aggregated = duckdb.sql(f"""
        WITH user_stats AS (
            SELECT
                user_id,
                COUNT(*) as num_transactions,
                AVG(amount) as avg_amount,
                STDDEV(amount) as std_amount,
                MAX(timestamp) as last_seen
            FROM '{data_path}'
            WHERE timestamp >= CURRENT_DATE - INTERVAL 90 DAY
            GROUP BY user_id
        )
        SELECT * FROM user_stats
        WHERE num_transactions >= 5
    """).pl()  # Convert to Polars DataFrame

    # Step 2: Polars for feature engineering
    features = (
        aggregated
        .with_columns([
            # Coefficient of variation
            (pl.col('std_amount') / pl.col('avg_amount')).alias('cv_amount'),
            # Log transform
            pl.col('num_transactions').log1p().alias('log_transactions')
        ])
        .select(['user_id', 'log_transactions', 'avg_amount', 'cv_amount'])
    )

    # Step 3: Query Polars with DuckDB (bidirectional!)
    final = duckdb.sql("""
        SELECT
            CASE
                WHEN cv_amount < 0.5 THEN 'consistent'
                ELSE 'variable'
            END as spending_pattern,
            AVG(avg_amount) as segment_avg
        FROM features
        GROUP BY spending_pattern
    """).pl()

    print(final)

    # Step 4: Convert to Tensor
    feature_matrix = features.drop('user_id').to_numpy()
    return pycyxwiz.Tensor.from_numpy(feature_matrix)
```

## Data Flow

### Training Data Pipeline

```
┌──────────────────┐
│   Raw Data       │
│  (Parquet/CSV)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│                    DuckDB (C++)                          │
│  SELECT features FROM 'data.parquet' WHERE valid = true  │
└────────┬─────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│                  Polars (Python)                         │
│  df.with_columns([...]).filter(...).select([...])        │
└────────┬─────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│              CyxWiz Tensor (ArrayFire)                   │
│  GPU memory for training                                 │
└────────┬─────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│                   ML Training                            │
│  Forward → Loss → Backward → Optimizer                   │
└──────────────────────────────────────────────────────────┘
```

### Batch Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Large Dataset (50GB+)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DuckDB BatchIterator                         │
│  • Streams data in configurable batch sizes                     │
│  • Automatic memory management with spillover                   │
│  • Projection/filter pushdown to minimize I/O                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌─────────┐    ┌─────────┐    ┌─────────┐
        │ Batch 1 │    │ Batch 2 │    │ Batch N │
        │ (2048)  │    │ (2048)  │    │ (2048)  │
        └────┬────┘    └────┬────┘    └────┬────┘
             │              │              │
             ▼              ▼              ▼
        ┌─────────┐    ┌─────────┐    ┌─────────┐
        │ Tensor  │    │ Tensor  │    │ Tensor  │
        │  (GPU)  │    │  (GPU)  │    │  (GPU)  │
        └────┬────┘    └────┬────┘    └────┬────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ▼
                    Training Loop
```

## Use Cases

### 1. Loading Training Data

**Scenario**: Load a 10GB Parquet dataset, filter invalid rows, select features

```cpp
// C++ Backend
cyxwiz::DataLoader loader;

auto training_data = loader.Query(R"(
    SELECT
        feature_1, feature_2, feature_3,
        feature_4, feature_5, label
    FROM 'data/train.parquet'
    WHERE label IS NOT NULL
      AND feature_1 BETWEEN -10 AND 10
)");

// Split features and labels
auto features = training_data.Slice({0, 0}, {-1, 5});
auto labels = training_data.Slice({0, 5}, {-1, 6});
```

### 2. Feature Engineering in Python Script

**Scenario**: User writes preprocessing script in Engine console

```python
# In CyxWiz Python Console
import polars as pl

# Load and transform
df = pl.scan_parquet('project/data/raw.parquet')

processed = (
    df
    .with_columns([
        pl.col('price').log1p().alias('log_price'),
        (pl.col('high') - pl.col('low')).alias('range'),
        pl.col('volume').rolling_mean(window_size=7).alias('vol_ma7')
    ])
    .drop_nulls()
    .collect()
)

# Save processed data
processed.write_parquet('project/data/processed.parquet')
print(f"Saved {len(processed)} rows")
```

### 3. Dataset Exploration

**Scenario**: User wants to explore dataset before training

```python
import duckdb

# Quick statistics
stats = duckdb.sql("""
    SELECT
        COUNT(*) as total_rows,
        COUNT(DISTINCT user_id) as unique_users,
        AVG(value) as avg_value,
        STDDEV(value) as std_value,
        MIN(timestamp) as min_date,
        MAX(timestamp) as max_date
    FROM 'data/transactions.parquet'
""").fetchone()

print(f"Total rows: {stats[0]:,}")
print(f"Unique users: {stats[1]:,}")
print(f"Value: {stats[2]:.2f} +/- {stats[3]:.2f}")
print(f"Date range: {stats[4]} to {stats[5]}")
```

### 4. Converting CSV to Parquet

**Scenario**: User has CSV files, wants to convert for better performance

```cpp
// C++ Backend
cyxwiz::DataLoader loader;

// Convert with compression
loader.ConvertCSVToParquet(
    "data/raw_data.csv",
    "data/raw_data.parquet",
    "zstd"  // Compression codec
);

// Now queries are much faster
auto data = loader.LoadParquet("data/raw_data.parquet");
```

## Performance Considerations

### DuckDB Optimizations

1. **Memory Limit**: Configure based on available RAM
   ```cpp
   config.memory_limit_mb = 8192;  // 8GB
   ```

2. **Thread Count**: Let DuckDB auto-detect or set explicitly
   ```cpp
   config.num_threads = 8;
   ```

3. **Projection Pushdown**: Only select needed columns
   ```sql
   -- Good: Only reads 3 columns from disk
   SELECT col1, col2, col3 FROM 'huge.parquet'

   -- Bad: Reads all columns
   SELECT * FROM 'huge.parquet'
   ```

4. **Filter Pushdown**: Filter early in query
   ```sql
   -- Good: Uses Parquet zone maps to skip row groups
   SELECT * FROM 'data.parquet' WHERE date >= '2024-01-01'
   ```

### Polars Optimizations

1. **Lazy Evaluation**: Always use `scan_*` instead of `read_*`
   ```python
   # Good: Lazy
   df = pl.scan_parquet('data.parquet')

   # Avoid: Eager (loads everything)
   df = pl.read_parquet('data.parquet')
   ```

2. **Streaming Mode**: For very large datasets
   ```python
   result = lazy_df.collect(streaming=True)
   ```

3. **Predicate Pushdown**: Filter before transforms
   ```python
   # Good: Filter pushed to scan
   (pl.scan_parquet('data.parquet')
      .filter(pl.col('value') > 0)
      .with_columns([...]))

   # Less optimal: Filter after transforms
   (pl.scan_parquet('data.parquet')
      .with_columns([...])
      .filter(pl.col('value') > 0))
   ```

### Memory Management

| Dataset Size | Recommended Approach |
|--------------|---------------------|
| < 1GB | Load entirely with DuckDB or Polars |
| 1-10GB | Use lazy evaluation, filter early |
| 10-100GB | Use DuckDB batch iterator |
| > 100GB | Use DuckDB with disk spillover |

## Error Handling

### DuckDB Errors

```cpp
try {
    auto result = loader.Query("SELECT * FROM 'nonexistent.parquet'");
} catch (const std::runtime_error& e) {
    spdlog::error("Query failed: {}", e.what());
    // Handle error...
}
```

### Polars Errors

```python
import polars as pl
from polars.exceptions import ComputeError, SchemaError

try:
    df = pl.scan_parquet('data.parquet').collect()
except FileNotFoundError:
    print("File not found")
except SchemaError as e:
    print(f"Schema mismatch: {e}")
except ComputeError as e:
    print(f"Computation error: {e}")
```

## Future Extensions

### Phase 2: Arrow Integration

- Implement Arrow C Data Interface for zero-copy between DuckDB and Polars
- Direct Arrow RecordBatch → ArrayFire array conversion
- Streaming Arrow data through the pipeline

### Phase 3: GUI Integration

- Dataset Panel shows DuckDB query results in ImGui tables
- Query builder UI for non-programmers
- Data preview with statistics and visualizations

### Phase 4: Distributed Processing

- DuckDB's upcoming distributed features
- Polars' streaming capabilities for cluster deployments
- Integration with CyxWiz's P2P network for distributed data loading

## References

- [DuckDB Documentation](https://duckdb.org/docs/)
- [Polars User Guide](https://docs.pola.rs/)
- [Apache Arrow](https://arrow.apache.org/)
- [DuckDB C API](https://duckdb.org/docs/api/c/overview)
- [Polars Python API](https://docs.pola.rs/api/python/stable/reference/)
