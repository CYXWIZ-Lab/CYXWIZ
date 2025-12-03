# Data Processing & Versioning with DuckDB and Polars

This document describes the architecture for persistent dataset versioning using DuckDB as the local database, integrated with our existing data processing pipeline.

## Overview

CyxWiz currently has in-memory dataset versioning (`version_history_` map in `DataRegistry`). This design introduces persistent versioning using DuckDB as a local project database, enabling:

- **Persistent Version History**: Versions survive across sessions
- **Data Lineage Tracking**: Know exactly what transformations created each version
- **Rollback Capability**: Restore any previous version of a dataset
- **Preprocessing Pipelines**: Chain transformations with automatic versioning
- **Efficient Storage**: Delta-based versioning using Parquet snapshots

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CyxWiz Project Directory Structure                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MyProject/                                                                  │
│  ├── MyProject.cyxwiz           # Project config                            │
│  ├── scripts/                   # Python scripts                            │
│  ├── models/                    # Saved models                              │
│  ├── datasets/                  # Raw data files                            │
│  │   ├── raw/                   # Original imported data                    │
│  │   └── processed/             # Transformed datasets                      │
│  ├── checkpoints/               # Training checkpoints                      │
│  └── .cyxwiz/                   # CyxWiz internal data (NEW)                │
│      ├── data.duckdb            # DuckDB database for versioning            │
│      ├── versions/              # Parquet snapshots for each version        │
│      │   ├── mnist_v1.parquet                                               │
│      │   ├── mnist_v2.parquet                                               │
│      │   └── ...                                                            │
│      └── cache/                 # Temporary/cached data                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Database Schema

```sql
-- Dataset metadata and versioning stored in DuckDB
-- Location: {project_root}/.cyxwiz/data.duckdb

-- Core dataset registry
CREATE TABLE datasets (
    id              INTEGER PRIMARY KEY,
    name            VARCHAR NOT NULL UNIQUE,
    source_path     VARCHAR,                    -- Original file path
    source_type     VARCHAR,                    -- CSV, Parquet, MNIST, etc.
    created_at      TIMESTAMP DEFAULT now(),
    updated_at      TIMESTAMP DEFAULT now(),
    is_active       BOOLEAN DEFAULT TRUE,

    -- Schema info
    num_samples     BIGINT,
    num_features    INTEGER,
    num_classes     INTEGER,
    shape           VARCHAR,                    -- JSON array: [28, 28, 1]
    dtype           VARCHAR DEFAULT 'float32',

    -- Split configuration
    train_ratio     REAL DEFAULT 0.8,
    val_ratio       REAL DEFAULT 0.1,
    test_ratio      REAL DEFAULT 0.1,
    split_seed      INTEGER DEFAULT 42,
    stratified      BOOLEAN DEFAULT TRUE
);

-- Version history for each dataset
CREATE TABLE dataset_versions (
    id              INTEGER PRIMARY KEY,
    dataset_id      INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    version_num     INTEGER NOT NULL,           -- v1, v2, v3...
    version_tag     VARCHAR,                    -- Optional: "baseline", "normalized"

    created_at      TIMESTAMP DEFAULT now(),
    created_by      VARCHAR DEFAULT 'user',     -- user, pipeline, transform
    description     VARCHAR,

    -- Data location
    parquet_path    VARCHAR NOT NULL,           -- .cyxwiz/versions/{name}_v{num}.parquet

    -- Stats at this version
    num_samples     BIGINT,
    num_features    INTEGER,
    checksum        VARCHAR,                    -- SHA256 of data
    file_size       BIGINT,                     -- Bytes

    -- Parent version (for lineage)
    parent_version  INTEGER REFERENCES dataset_versions(id),

    UNIQUE(dataset_id, version_num)
);

-- Transformation history (data lineage)
CREATE TABLE transforms (
    id              INTEGER PRIMARY KEY,
    dataset_id      INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    input_version   INTEGER REFERENCES dataset_versions(id),
    output_version  INTEGER REFERENCES dataset_versions(id),

    transform_type  VARCHAR NOT NULL,           -- normalize, filter, augment, etc.
    transform_name  VARCHAR,                    -- User-friendly name
    parameters      VARCHAR,                    -- JSON: {"method": "zscore", "columns": [...]}

    created_at      TIMESTAMP DEFAULT now(),
    duration_ms     INTEGER,                    -- How long it took

    -- For reproducibility
    script_path     VARCHAR,                    -- Python script that ran this
    script_hash     VARCHAR                     -- Hash of script content
);

-- Preprocessing pipelines (reusable transform chains)
CREATE TABLE pipelines (
    id              INTEGER PRIMARY KEY,
    name            VARCHAR NOT NULL UNIQUE,
    description     VARCHAR,
    created_at      TIMESTAMP DEFAULT now(),
    updated_at      TIMESTAMP DEFAULT now()
);

-- Pipeline steps (ordered transforms)
CREATE TABLE pipeline_steps (
    id              INTEGER PRIMARY KEY,
    pipeline_id     INTEGER REFERENCES pipelines(id) ON DELETE CASCADE,
    step_order      INTEGER NOT NULL,
    transform_type  VARCHAR NOT NULL,
    parameters      VARCHAR,                    -- JSON

    UNIQUE(pipeline_id, step_order)
);

-- Indices for common queries
CREATE INDEX idx_versions_dataset ON dataset_versions(dataset_id);
CREATE INDEX idx_versions_created ON dataset_versions(created_at);
CREATE INDEX idx_transforms_dataset ON transforms(dataset_id);
CREATE INDEX idx_transforms_type ON transforms(transform_type);
```

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Data Processing Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        DataVersionManager (C++)                        │  │
│  │  Singleton managing DuckDB connection and version operations          │  │
│  │                                                                        │  │
│  │  • OpenDatabase(project_path)     • CreateVersion(name, data)         │  │
│  │  • CloseDatabase()                • GetVersion(name, version_num)      │  │
│  │  • RegisterDataset(info)          • ListVersions(name)                 │  │
│  │  • UnregisterDataset(name)        • Rollback(name, version_num)        │  │
│  │  • GetDatasetInfo(name)           • CompareVersions(v1, v2)            │  │
│  │                                                                        │  │
│  └─────────────────────────────────────┬─────────────────────────────────┘  │
│                                        │                                     │
│                    ┌───────────────────┴───────────────────┐                │
│                    │                                       │                │
│                    ▼                                       ▼                │
│  ┌─────────────────────────────────┐  ┌──────────────────────────────────┐ │
│  │     DuckDB (data.duckdb)        │  │    Parquet Storage (versions/)   │ │
│  │                                  │  │                                  │ │
│  │  • Metadata tables               │  │  • Actual data snapshots         │ │
│  │  • Version tracking              │  │  • Columnar, compressed          │ │
│  │  • Transform history             │  │  • Fast random access            │ │
│  │  • SQL queries on versions       │  │  • Schema evolution support      │ │
│  │                                  │  │                                  │ │
│  └─────────────────────────────────┘  └──────────────────────────────────┘ │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     DataRegistry Integration                           │  │
│  │                                                                        │  │
│  │  DataRegistry ◄───► DataVersionManager                                │  │
│  │                                                                        │  │
│  │  • LoadDataset() triggers version tracking                            │  │
│  │  • SaveVersion() persists to DuckDB + Parquet                         │  │
│  │  • GetVersionHistory() queries DuckDB                                 │  │
│  │  • Import/Export uses version storage                                 │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    TransformEngine (Python + C++)                      │  │
│  │                                                                        │  │
│  │  C++ Core:                           Python (Polars):                  │  │
│  │  • Normalize                         • Complex transforms              │  │
│  │  • OneHotEncode                      • Feature engineering             │  │
│  │  • StandardScale                     • Custom scripts                  │  │
│  │  • MinMaxScale                       • Lazy evaluation                 │  │
│  │  • Filter                            • SQL queries via DuckDB          │  │
│  │                                                                        │  │
│  │  Both automatically version outputs!                                   │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## C++ API Design

### DataVersionManager Class

```cpp
// cyxwiz-engine/src/core/data_version_manager.h
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <functional>

#ifdef CYXWIZ_HAS_DUCKDB
#include <duckdb.h>
#endif

namespace cyxwiz {

// Forward declarations
class Dataset;
struct DatasetInfo;
struct SplitConfig;

/**
 * Version metadata stored in database
 */
struct VersionInfo {
    int64_t id;
    int64_t dataset_id;
    int version_num;
    std::string version_tag;

    std::string created_at;
    std::string created_by;
    std::string description;

    std::string parquet_path;
    size_t num_samples;
    size_t num_features;
    std::string checksum;
    size_t file_size;

    std::optional<int64_t> parent_version;
};

/**
 * Transform record for lineage tracking
 */
struct TransformRecord {
    int64_t id;
    int64_t dataset_id;
    int64_t input_version_id;
    int64_t output_version_id;

    std::string transform_type;
    std::string transform_name;
    std::string parameters_json;

    std::string created_at;
    int duration_ms;

    std::string script_path;
    std::string script_hash;
};

/**
 * Pipeline definition
 */
struct PipelineInfo {
    int64_t id;
    std::string name;
    std::string description;
    std::string created_at;

    struct Step {
        int step_order;
        std::string transform_type;
        std::string parameters_json;
    };
    std::vector<Step> steps;
};

/**
 * Version comparison result
 */
struct VersionDiff {
    int64_t version_a;
    int64_t version_b;

    int64_t samples_added;
    int64_t samples_removed;
    std::vector<std::string> columns_added;
    std::vector<std::string> columns_removed;
    std::vector<std::string> columns_modified;

    // Statistics diff
    struct StatDiff {
        std::string column;
        double mean_diff;
        double std_diff;
        double min_diff;
        double max_diff;
    };
    std::vector<StatDiff> stat_diffs;
};

/**
 * DataVersionManager - Persistent versioning with DuckDB
 *
 * Singleton managing dataset versioning for the active project.
 * Uses DuckDB for metadata and Parquet for data snapshots.
 */
class DataVersionManager {
public:
    // Singleton access
    static DataVersionManager& Instance();

    // Prevent copying
    DataVersionManager(const DataVersionManager&) = delete;
    DataVersionManager& operator=(const DataVersionManager&) = delete;

    // Database lifecycle
    bool OpenDatabase(const std::string& project_root);
    void CloseDatabase();
    bool IsOpen() const { return db_open_; }
    std::string GetDatabasePath() const { return db_path_; }

    // Dataset registration
    int64_t RegisterDataset(const DatasetInfo& info, const SplitConfig& split);
    bool UnregisterDataset(const std::string& name);
    bool UpdateDataset(const std::string& name, const DatasetInfo& info);
    std::optional<DatasetInfo> GetDatasetInfo(const std::string& name) const;
    std::vector<std::string> ListDatasets() const;

    // Version management
    int64_t CreateVersion(const std::string& dataset_name,
                          const std::shared_ptr<Dataset>& data,
                          const std::string& description = "",
                          const std::string& tag = "");

    std::shared_ptr<Dataset> GetVersion(const std::string& dataset_name,
                                         int version_num) const;

    std::shared_ptr<Dataset> GetLatestVersion(const std::string& dataset_name) const;

    std::vector<VersionInfo> ListVersions(const std::string& dataset_name) const;

    std::optional<VersionInfo> GetVersionInfo(const std::string& dataset_name,
                                               int version_num) const;

    bool DeleteVersion(const std::string& dataset_name, int version_num);

    // Rollback to a previous version
    bool Rollback(const std::string& dataset_name, int version_num);

    // Version comparison
    VersionDiff CompareVersions(const std::string& dataset_name,
                                 int version_a, int version_b) const;

    // Transform tracking
    int64_t RecordTransform(const std::string& dataset_name,
                            int input_version,
                            int output_version,
                            const std::string& transform_type,
                            const std::string& parameters_json,
                            int duration_ms = 0,
                            const std::string& script_path = "");

    std::vector<TransformRecord> GetTransformHistory(const std::string& dataset_name) const;

    // Get lineage (transforms that led to a version)
    std::vector<TransformRecord> GetLineage(const std::string& dataset_name,
                                             int version_num) const;

    // Pipeline management
    int64_t CreatePipeline(const std::string& name,
                           const std::string& description,
                           const std::vector<PipelineInfo::Step>& steps);

    std::optional<PipelineInfo> GetPipeline(const std::string& name) const;
    std::vector<PipelineInfo> ListPipelines() const;
    bool DeletePipeline(const std::string& name);

    // Execute pipeline on dataset (creates new version)
    int64_t ExecutePipeline(const std::string& pipeline_name,
                            const std::string& dataset_name,
                            int input_version = -1);  // -1 = latest

    // SQL queries on versioned data
    std::string QueryVersion(const std::string& sql,
                              const std::string& dataset_name,
                              int version_num = -1) const;  // Returns JSON

    // Callbacks
    using VersionCreatedCallback = std::function<void(const std::string& name, int version)>;
    void SetOnVersionCreated(VersionCreatedCallback cb) { on_version_created_ = std::move(cb); }

    // Statistics
    size_t GetTotalVersionCount() const;
    size_t GetTotalStorageUsed() const;  // Bytes

    // Cleanup old versions
    void PruneVersions(const std::string& dataset_name, int keep_count = 10);
    void PruneAllOldVersions(int keep_count = 10);

private:
    DataVersionManager() = default;
    ~DataVersionManager();

    // Internal helpers
    bool InitializeSchema();
    std::string GenerateParquetPath(const std::string& name, int version) const;
    std::string ComputeChecksum(const std::string& parquet_path) const;
    bool WriteParquet(const std::shared_ptr<Dataset>& data,
                      const std::string& path) const;
    std::shared_ptr<Dataset> ReadParquet(const std::string& path) const;

    // State
    bool db_open_ = false;
    std::string db_path_;
    std::string versions_dir_;

#ifdef CYXWIZ_HAS_DUCKDB
    duckdb_database db_;
    duckdb_connection con_;
#endif

    // Callbacks
    VersionCreatedCallback on_version_created_;
};

} // namespace cyxwiz
```

### Integration with DataRegistry

```cpp
// Updates to data_registry.h

class DataRegistry {
public:
    // ... existing methods ...

    // Enhanced versioning (now persistent)
    bool SaveVersion(const std::string& name,
                     const std::string& description = "",
                     const std::string& tag = "");

    std::vector<DatasetVersion> GetVersionHistory(const std::string& name) const;

    // New: Load a specific version
    DatasetHandle LoadVersion(const std::string& name, int version_num);

    // New: Rollback to previous version
    bool RollbackToVersion(const std::string& name, int version_num);

    // New: Apply transform and auto-version
    DatasetHandle ApplyTransform(const std::string& name,
                                  const std::string& transform_type,
                                  const std::string& params_json);

private:
    // Connects to DataVersionManager for persistence
    void SyncWithVersionManager();
};
```

## Python API Design

### Polars + DuckDB + Versioning

```python
# cyxwiz-engine/python/cyxwiz_data.py
"""
CyxWiz Data Processing Module

Provides high-level data processing with automatic versioning.
Uses Polars for transformations and DuckDB for persistence.
"""

import polars as pl
import duckdb
from pathlib import Path
from typing import Optional, List, Dict, Any
import hashlib
import json
from datetime import datetime

class CyxWizData:
    """High-level data processing with automatic versioning."""

    def __init__(self, project_path: str):
        """Initialize with project path."""
        self.project_path = Path(project_path)
        self.cyxwiz_dir = self.project_path / ".cyxwiz"
        self.db_path = self.cyxwiz_dir / "data.duckdb"
        self.versions_dir = self.cyxwiz_dir / "versions"

        # Ensure directories exist
        self.cyxwiz_dir.mkdir(exist_ok=True)
        self.versions_dir.mkdir(exist_ok=True)

        # Connect to DuckDB
        self.conn = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema if not exists."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY,
                name VARCHAR UNIQUE,
                source_path VARCHAR,
                source_type VARCHAR,
                created_at TIMESTAMP DEFAULT now(),
                num_samples BIGINT,
                shape VARCHAR
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS dataset_versions (
                id INTEGER PRIMARY KEY,
                dataset_id INTEGER REFERENCES datasets(id),
                version_num INTEGER,
                parquet_path VARCHAR,
                created_at TIMESTAMP DEFAULT now(),
                description VARCHAR,
                num_samples BIGINT,
                checksum VARCHAR,
                parent_version INTEGER,
                UNIQUE(dataset_id, version_num)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS transforms (
                id INTEGER PRIMARY KEY,
                dataset_id INTEGER REFERENCES datasets(id),
                input_version INTEGER,
                output_version INTEGER,
                transform_type VARCHAR,
                parameters VARCHAR,
                created_at TIMESTAMP DEFAULT now()
            )
        """)

    def load(self, name: str, version: int = -1) -> pl.LazyFrame:
        """
        Load a dataset by name, optionally at a specific version.

        Args:
            name: Dataset name
            version: Version number (-1 for latest)

        Returns:
            Polars LazyFrame for efficient processing
        """
        if version == -1:
            # Get latest version
            result = self.conn.execute("""
                SELECT v.parquet_path
                FROM dataset_versions v
                JOIN datasets d ON v.dataset_id = d.id
                WHERE d.name = ?
                ORDER BY v.version_num DESC
                LIMIT 1
            """, [name]).fetchone()
        else:
            result = self.conn.execute("""
                SELECT v.parquet_path
                FROM dataset_versions v
                JOIN datasets d ON v.dataset_id = d.id
                WHERE d.name = ? AND v.version_num = ?
            """, [name, version]).fetchone()

        if not result:
            raise ValueError(f"Dataset '{name}' version {version} not found")

        return pl.scan_parquet(result[0])

    def register(self, name: str, path: str) -> int:
        """
        Register a new dataset and create initial version.

        Args:
            name: Unique dataset name
            path: Path to data file (CSV, Parquet, etc.)

        Returns:
            Version number (1)
        """
        path = Path(path)

        # Detect type and load
        if path.suffix == '.csv':
            df = pl.scan_csv(str(path))
            source_type = 'CSV'
        elif path.suffix in ['.parquet', '.pq']:
            df = pl.scan_parquet(str(path))
            source_type = 'Parquet'
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        # Collect to get stats
        collected = df.collect()
        num_samples = len(collected)
        shape = json.dumps(list(collected.shape))

        # Register dataset
        self.conn.execute("""
            INSERT INTO datasets (name, source_path, source_type, num_samples, shape)
            VALUES (?, ?, ?, ?, ?)
        """, [name, str(path), source_type, num_samples, shape])

        dataset_id = self.conn.execute(
            "SELECT id FROM datasets WHERE name = ?", [name]
        ).fetchone()[0]

        # Create version 1
        parquet_path = str(self.versions_dir / f"{name}_v1.parquet")
        collected.write_parquet(parquet_path)

        checksum = self._compute_checksum(parquet_path)

        self.conn.execute("""
            INSERT INTO dataset_versions
            (dataset_id, version_num, parquet_path, description, num_samples, checksum)
            VALUES (?, 1, ?, 'Initial import', ?, ?)
        """, [dataset_id, parquet_path, num_samples, checksum])

        return 1

    def transform(self,
                  name: str,
                  transform_fn,
                  description: str = "",
                  input_version: int = -1) -> int:
        """
        Apply a transformation and create a new version.

        Args:
            name: Dataset name
            transform_fn: Function that takes LazyFrame and returns LazyFrame
            description: Description of the transform
            input_version: Version to transform (-1 for latest)

        Returns:
            New version number
        """
        # Load input version
        input_df = self.load(name, input_version)

        # Get actual input version number
        if input_version == -1:
            input_version = self.get_latest_version(name)

        # Apply transform
        output_df = transform_fn(input_df)

        # Collect and save
        collected = output_df.collect()

        # Get next version number
        next_version = input_version + 1

        # Save parquet
        parquet_path = str(self.versions_dir / f"{name}_v{next_version}.parquet")
        collected.write_parquet(parquet_path)

        # Get dataset ID
        dataset_id = self.conn.execute(
            "SELECT id FROM datasets WHERE name = ?", [name]
        ).fetchone()[0]

        # Get input version ID
        input_version_id = self.conn.execute("""
            SELECT id FROM dataset_versions
            WHERE dataset_id = ? AND version_num = ?
        """, [dataset_id, input_version]).fetchone()[0]

        # Record new version
        checksum = self._compute_checksum(parquet_path)
        self.conn.execute("""
            INSERT INTO dataset_versions
            (dataset_id, version_num, parquet_path, description, num_samples, checksum, parent_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [dataset_id, next_version, parquet_path, description,
              len(collected), checksum, input_version_id])

        # Record transform
        output_version_id = self.conn.execute("""
            SELECT id FROM dataset_versions
            WHERE dataset_id = ? AND version_num = ?
        """, [dataset_id, next_version]).fetchone()[0]

        self.conn.execute("""
            INSERT INTO transforms
            (dataset_id, input_version, output_version, transform_type, parameters)
            VALUES (?, ?, ?, 'python_function', ?)
        """, [dataset_id, input_version_id, output_version_id,
              json.dumps({'description': description})])

        return next_version

    def get_latest_version(self, name: str) -> int:
        """Get the latest version number for a dataset."""
        result = self.conn.execute("""
            SELECT MAX(v.version_num)
            FROM dataset_versions v
            JOIN datasets d ON v.dataset_id = d.id
            WHERE d.name = ?
        """, [name]).fetchone()
        return result[0] if result[0] else 0

    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """List all versions of a dataset."""
        results = self.conn.execute("""
            SELECT v.version_num, v.created_at, v.description,
                   v.num_samples, v.checksum
            FROM dataset_versions v
            JOIN datasets d ON v.dataset_id = d.id
            WHERE d.name = ?
            ORDER BY v.version_num DESC
        """, [name]).fetchall()

        return [
            {
                'version': r[0],
                'created_at': str(r[1]),
                'description': r[2],
                'num_samples': r[3],
                'checksum': r[4][:8] + '...'  # Truncate for display
            }
            for r in results
        ]

    def get_lineage(self, name: str, version: int = -1) -> List[Dict[str, Any]]:
        """Get the transformation lineage for a version."""
        if version == -1:
            version = self.get_latest_version(name)

        # Recursive CTE to get lineage
        results = self.conn.execute("""
            WITH RECURSIVE lineage AS (
                SELECT v.id, v.version_num, v.parent_version, v.description
                FROM dataset_versions v
                JOIN datasets d ON v.dataset_id = d.id
                WHERE d.name = ? AND v.version_num = ?

                UNION ALL

                SELECT v.id, v.version_num, v.parent_version, v.description
                FROM dataset_versions v
                JOIN lineage l ON v.id = l.parent_version
            )
            SELECT version_num, description FROM lineage ORDER BY version_num
        """, [name, version]).fetchall()

        return [{'version': r[0], 'description': r[1]} for r in results]

    def query(self, sql: str) -> pl.DataFrame:
        """
        Execute SQL query on versioned data.

        Supports special syntax:
        - FROM dataset:version (e.g., FROM mnist:3)
        - FROM dataset (uses latest version)
        """
        import re

        # Replace dataset:version references with actual paths
        def replace_dataset_ref(match):
            full_match = match.group(1)
            if ':' in full_match:
                name, version = full_match.split(':')
                version = int(version)
            else:
                name = full_match
                version = -1

            df = self.load(name, version).collect()
            # Register as temp view
            self.conn.register(f"_temp_{name}", df.to_arrow())
            return f"_temp_{name}"

        # Pattern: dataset_name or dataset_name:version_num
        processed_sql = re.sub(
            r"FROM\s+([a-zA-Z_][a-zA-Z0-9_]*(?::\d+)?)",
            lambda m: f"FROM {replace_dataset_ref(m)}",
            sql,
            flags=re.IGNORECASE
        )

        result = self.conn.execute(processed_sql).pl()
        return result

    def rollback(self, name: str, version: int) -> bool:
        """
        Rollback to a previous version (creates new version pointing to old data).
        """
        # Get the version info
        version_info = self.conn.execute("""
            SELECT v.parquet_path, v.num_samples, v.checksum
            FROM dataset_versions v
            JOIN datasets d ON v.dataset_id = d.id
            WHERE d.name = ? AND v.version_num = ?
        """, [name, version]).fetchone()

        if not version_info:
            return False

        # Create new version with same data
        latest = self.get_latest_version(name)
        new_version = latest + 1

        # Copy parquet file
        import shutil
        old_path = version_info[0]
        new_path = str(self.versions_dir / f"{name}_v{new_version}.parquet")
        shutil.copy(old_path, new_path)

        # Get dataset and version IDs
        dataset_id = self.conn.execute(
            "SELECT id FROM datasets WHERE name = ?", [name]
        ).fetchone()[0]

        rollback_from_id = self.conn.execute("""
            SELECT id FROM dataset_versions
            WHERE dataset_id = ? AND version_num = ?
        """, [dataset_id, version]).fetchone()[0]

        # Record new version
        self.conn.execute("""
            INSERT INTO dataset_versions
            (dataset_id, version_num, parquet_path, description, num_samples, checksum, parent_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [dataset_id, new_version, new_path,
              f'Rollback to v{version}', version_info[1], version_info[2], rollback_from_id])

        return True

    def _compute_checksum(self, path: str) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def close(self):
        """Close database connection."""
        self.conn.close()


# Convenience functions for use in CyxWiz Python Console
_data_manager: Optional[CyxWizData] = None

def init_data(project_path: str):
    """Initialize data manager for a project."""
    global _data_manager
    _data_manager = CyxWizData(project_path)

def load(name: str, version: int = -1) -> pl.LazyFrame:
    """Load a versioned dataset."""
    return _data_manager.load(name, version)

def save_version(name: str, df: pl.LazyFrame, description: str = "") -> int:
    """Save a new version of a dataset."""
    return _data_manager.transform(
        name,
        lambda _: df,
        description
    )

def versions(name: str):
    """List versions of a dataset."""
    return _data_manager.list_versions(name)

def lineage(name: str, version: int = -1):
    """Get lineage of a dataset version."""
    return _data_manager.get_lineage(name, version)

def query(sql: str) -> pl.DataFrame:
    """Execute SQL on versioned datasets."""
    return _data_manager.query(sql)
```

### Usage Examples

```python
# In CyxWiz Python Console

# Initialize for current project
import cyxwiz_data as data
data.init_data("path/to/MyProject")

# Register a new dataset
data._data_manager.register("mnist", "datasets/raw/mnist.csv")

# Load latest version
df = data.load("mnist")

# Apply transformations with automatic versioning
def normalize_features(df):
    return df.with_columns([
        ((pl.col("pixel_*") - pl.col("pixel_*").mean()) /
         pl.col("pixel_*").std()).name.keep()
    ])

v2 = data._data_manager.transform(
    "mnist",
    normalize_features,
    "Normalized pixel values"
)
print(f"Created version {v2}")

# Load specific version
df_v1 = data.load("mnist", version=1)
df_v2 = data.load("mnist", version=2)

# View version history
for v in data.versions("mnist"):
    print(f"v{v['version']}: {v['description']} ({v['num_samples']} samples)")

# View lineage
print("Lineage for v2:")
for step in data.lineage("mnist", 2):
    print(f"  v{step['version']}: {step['description']}")

# SQL queries on versioned data
result = data.query("""
    SELECT label, COUNT(*) as count
    FROM mnist:2
    GROUP BY label
    ORDER BY count DESC
""")
print(result)

# Rollback to previous version
data._data_manager.rollback("mnist", 1)
print(f"Rolled back to v1, new version: {data._data_manager.get_latest_version('mnist')}")
```

## Data Flow

### Transform Pipeline with Versioning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Data Preprocessing with Versioning                        │
└─────────────────────────────────────────────────────────────────────────────┘

  User loads raw data
         │
         ▼
┌─────────────────────┐
│  datasets/raw.csv   │
│  (Original file)    │
└─────────┬───────────┘
          │
          │ register("mydata", "raw.csv")
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Version 1: Initial Import                                                   │
│  ├── .cyxwiz/versions/mydata_v1.parquet                                     │
│  └── DuckDB: dataset_versions(id=1, version_num=1, description="Initial")   │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          │ transform(normalize_columns, "Normalize features")
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Version 2: Normalized                                                       │
│  ├── .cyxwiz/versions/mydata_v2.parquet                                     │
│  ├── DuckDB: dataset_versions(id=2, version_num=2, parent=1)                │
│  └── DuckDB: transforms(input=1, output=2, type="normalize")                │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          │ transform(encode_labels, "One-hot encode labels")
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Version 3: Encoded                                                          │
│  ├── .cyxwiz/versions/mydata_v3.parquet                                     │
│  ├── DuckDB: dataset_versions(id=3, version_num=3, parent=2)                │
│  └── DuckDB: transforms(input=2, output=3, type="encode")                   │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          │ transform(augment_data, "Add data augmentation")
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Version 4: Augmented (CURRENT)                                              │
│  ├── .cyxwiz/versions/mydata_v4.parquet                                     │
│  ├── DuckDB: dataset_versions(id=4, version_num=4, parent=3)                │
│  └── DuckDB: transforms(input=3, output=4, type="augment")                  │
└─────────────────────────────────────────────────────────────────────────────┘

  LINEAGE QUERY: "How was v4 created?"

  v1 (Initial import)
    └─► v2 (Normalize features)
         └─► v3 (One-hot encode labels)
              └─► v4 (Add data augmentation)
```

### GUI Integration Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Dataset Panel (ImGui)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Dataset: mnist                    [v3 ▾]  [Versions] [Lineage]             │
│  ─────────────────────────────────────────────────────────────              │
│  Samples: 60,000  │  Features: 784  │  Classes: 10                          │
│                                                                              │
│  ┌─ Version History ──────────────────────────────────────────────────────┐ │
│  │ v3  2024-12-01 14:30  "Normalized and encoded"        [Load] [Compare] │ │
│  │ v2  2024-12-01 14:25  "Normalized pixel values"       [Load] [Compare] │ │
│  │ v1  2024-12-01 14:20  "Initial import from CSV"       [Load] [Compare] │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─ Lineage (v3) ─────────────────────────────────────────────────────────┐ │
│  │ v1 ──[import]──► v2 ──[normalize]──► v3 ──[encode]──►                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  [Save New Version]  [Rollback]  [Export Parquet]  [Compare Versions]       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Core Infrastructure

1. **Add DuckDB to vcpkg.json**
   ```json
   {
     "dependencies": [
       "duckdb",
       ...
     ]
   }
   ```

2. **Create DataVersionManager class**
   - DuckDB connection management
   - Schema initialization
   - Basic CRUD operations

3. **Create .cyxwiz directory structure**
   - Hook into ProjectManager::CreateProject()
   - Auto-create data.duckdb and versions/

### Phase 2: Version Storage

1. **Parquet read/write integration**
   - Use DuckDB's built-in Parquet support
   - Checksum computation

2. **Version creation and retrieval**
   - CreateVersion() with automatic numbering
   - GetVersion() with Parquet loading

3. **DataRegistry integration**
   - Hook SaveVersion() to DataVersionManager
   - Hook GetVersionHistory() to query DuckDB

### Phase 3: Transform Tracking

1. **Transform recording**
   - Capture transform type and parameters
   - Link input/output versions

2. **Lineage queries**
   - Recursive CTE for full lineage
   - Parent version tracking

3. **Python API**
   - CyxWizData class implementation
   - Console convenience functions

### Phase 4: GUI Integration

1. **Version History Panel**
   - List versions with metadata
   - Load/compare buttons

2. **Lineage Visualization**
   - Simple text-based lineage display
   - (Future: ImNodes graph)

3. **Transform Preview**
   - Show stats diff between versions
   - Preview data samples

### Phase 5: Advanced Features

1. **Pipelines**
   - Save reusable transform chains
   - One-click pipeline execution

2. **Version pruning**
   - Auto-delete old versions
   - Keep configurable history

3. **Export/Import**
   - Export version to standalone Parquet
   - Import external Parquet as new version

## Configuration

### Project Config Extension

```json
// MyProject.cyxwiz
{
  "name": "MyProject",
  "version": "1.0.0",
  "data_versioning": {
    "enabled": true,
    "auto_version_on_transform": true,
    "max_versions_per_dataset": 50,
    "prune_old_versions": true,
    "storage_limit_mb": 10240
  }
}
```

## References

- [DuckDB Documentation](https://duckdb.org/docs/)
- [DuckDB C API](https://duckdb.org/docs/api/c/overview)
- [Polars User Guide](https://docs.pola.rs/)
- [Apache Parquet](https://parquet.apache.org/)
- [Data Version Control (DVC)](https://dvc.org/) - Inspiration for design
