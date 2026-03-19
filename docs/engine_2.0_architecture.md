# CyxWiz Engine 2.0 — Data Studio Integration Architecture

**Document Version:** 1.0
**Author:** CyxWiz Architecture Team
**Date:** 2026-03-19
**Status:** Design Phase

---

## Executive Summary

CyxWiz Engine 2.0 introduces **Data Studio**, a first-class visual data preparation system integrated directly into the engine. Inspired by KNIME Analytics Platform, Data Studio provides a complete data workflow pipeline builder that transforms raw data into ML-ready tensors and hands them directly to the Node Editor for training — all within a single application.

**Key Goals:**
1. **Zero-Friction Workflow:** From raw CSV → clean tensors → trained model without leaving the IDE
2. **Visual Pipeline Builder:** Node-based data transformation (separate from ML Node Editor)
3. **DuckDB Integration:** SQL query editor for exploratory analysis
4. **6-Phase Data Lifecycle:** Data Access → Transform → Analyze → Annotate → Visualize → Deploy
5. **Seamless Handoff:** Pipeline outputs connect directly to Node Editor's `DataInput` nodes
6. **Backward Compatibility:** Existing projects, saved graphs, and workflows continue to work

---

## 1. Technology Foundation: Apache Arrow + DuckDB

### 1.1 Why Arrow + DuckDB?

**Critical Decision:** Data Studio uses **Apache Arrow** as the native in-memory data format with **DuckDB** for SQL transformations. This provides zero-copy data interchange and best-in-class performance.

**Key Benefits:**
1. **Zero-Copy Integration:** DuckDB and Arrow share memory buffers without copying ([DuckDB Blog](https://duckdb.org/2021/12/03/duck-arrow))
2. **Larger-Than-Memory:** Stream datasets bigger than RAM through Arrow's chunked format
3. **Advanced Optimization:** DuckDB pushes filters/projections into Arrow scans (partition elimination, column pruning)
4. **Industry Standard:** Arrow is the de-facto standard for columnar data (Parquet, Feather, IPC)
5. **Fast SQL Queries:** DuckDB on Arrow is extremely fast (vectorized execution, parallel processing)

### 1.2 Data Format Architecture

```
Raw Data (CSV/Parquet/Feather)
  ↓
DuckDB Scan (zero-copy read)
  ↓
Apache Arrow Table (columnar in-memory)
  ↓
Data Studio Pipeline (transformations on Arrow)
  ↓
Arrow Table (cleaned/transformed)
  ↓
Convert to ArrayFire Tensor (only at final step for ML training)
  ↓
ML Node Editor (tensor operations)
```

**Rationale:**
- Keep data in Arrow format as long as possible (efficient columnar operations)
- Only convert to tensors when entering ML Node Editor
- DuckDB provides SQL interface for exploratory analysis
- Zero-copy handoff between Arrow and DuckDB

### 1.3 Arrow Integration Points

**In DataRegistry:**
```cpp
class Dataset {
    std::shared_ptr<arrow::Table> arrow_table_;  // Arrow format (primary)
    af::array tensor_;                            // ArrayFire tensor (lazy conversion)

    // Convert Arrow → Tensor only when needed
    af::array GetTensor() {
        if (!tensor_.isempty()) return tensor_;
        tensor_ = ArrowToTensor(arrow_table_);
        return tensor_;
    }
};
```

**In Data Studio Pipeline:**
```cpp
class DataStudioNode {
    // Execute node on Arrow table
    virtual arrow::Result<std::shared_ptr<arrow::Table>>
    Execute(std::shared_ptr<arrow::Table> input) = 0;
};

// Example: Filter node
class FilterRowsNode : public DataStudioNode {
    arrow::Result<std::shared_ptr<arrow::Table>> Execute(
        std::shared_ptr<arrow::Table> input) override {

        // Use DuckDB for filtering (zero-copy)
        auto result = duckdb_conn_->Query(
            "SELECT * FROM input_table WHERE " + filter_expression_);
        return result->FetchArrowTable();  // Zero-copy!
    }
};
```

**DuckDB Integration:**
```cpp
class DataStudioPipeline {
    duckdb::DuckDB db_;                    // In-memory database
    duckdb::Connection conn_;              // Connection for queries

    void RegisterDataset(const std::string& name,
                        std::shared_ptr<arrow::Table> table) {
        // Register Arrow table as DuckDB table (zero-copy view)
        conn_.Query("CREATE VIEW " + name + " AS SELECT * FROM table");
    }

    arrow::Table QuerySQL(const std::string& sql) {
        auto result = conn_.Query(sql);
        return result->FetchArrowTable();  // Zero-copy result
    }
};
```

### 1.4 Memory Efficiency Benefits

**Without Arrow (Current System):**
```
CSV File → std::vector<float> → af::array → Copy to DataRegistry
Memory: 3x dataset size (file buffer + vector + tensor)
```

**With Arrow (New System):**
```
CSV File → Arrow Table (mmap or zero-copy read) → View in DuckDB
Memory: 1x dataset size (single Arrow buffer, shared across all views)
Tensor: Only created when entering ML Node Editor
```

**Streaming Support:**
```cpp
// Handle 100M+ row datasets with streaming
auto reader = arrow::ipc::RecordBatchFileReader::Open(file);
for (int i = 0; i < reader->num_record_batches(); ++i) {
    auto batch = reader->ReadRecordBatch(i);
    ProcessBatch(batch);  // Process in chunks, never load full dataset
}
```

---

## 2. Current Engine Architecture Analysis

### 2.1 Existing Component Structure

```
CyxWizApp (application.cpp/h)
  └─ MainWindow (main_window.cpp/h)
      ├─ NodeEditor (node_editor.cpp/h)              — ML pipeline builder (ImNodes)
      ├─ Console (console.cpp/h)                     — Python REPL
      ├─ Viewport (viewport.cpp/h)                   — Training visualization
      ├─ Properties (properties.cpp/h)               — Node properties
      ├─ DatasetPanel (dataset_panel.cpp/h)         — Dataset manager
      ├─ ScriptEditorPanel (script_editor.cpp/h)    — Python script editor
      ├─ TableViewerPanel (table_viewer.cpp/h)      — Data table viewer
      ├─ AssetBrowserPanel (asset_browser.cpp/h)    — File tree
      └─ ~80 other panels (analysis, clustering, signal processing, etc.)

Core Systems:
  - DataRegistry (data_registry.cpp/h)              — Dataset storage, LRU cache, memory management
  - GraphExecutor (graph_executor.cpp/h)            — Runtime graph evaluation (MuJoCo/RL nodes)
  - GraphCompiler (graph_compiler.cpp/h)            — Node graph → executable code
  - AsyncTaskManager (async_task_manager.cpp/h)    — Background tasks with progress
  - AnnotationManager (annotation_manager.cpp/h)   — Segmentation annotations
  - ScriptingEngine (scripting_engine.cpp/h)       — Embedded Python (pybind11)
```

### 1.2 Key Architectural Patterns

**Pattern 1: Panel-Based UI**
- Each feature is a panel class (`*Panel`)
- Panels register with `MainWindow`'s sidebar for visibility toggle
- Panels are `std::unique_ptr` members of `MainWindow`
- All panels render in dockable ImGui windows

**Pattern 2: Centralized Data Registry**
- `DataRegistry` is a singleton managing all loaded datasets
- Datasets are identified by string name (unique key)
- Memory management via LRU eviction when exceeding limit
- Thread-safe access with `std::mutex`

**Pattern 3: ImNodes for Visual Editing**
- Current `NodeEditor` uses ImNodes for ML pipeline graphs
- Nodes have typed input/output pins (`PinType::Tensor`, `PinType::Loss`, etc.)
- Links connect pins for data flow
- Graph is serializable to JSON

**Pattern 4: Async Task Execution**
- Background tasks managed by `AsyncTaskManager`
- Progress callbacks update UI (progress bars, notifications)
- Used for dataset loading, analytics computation, etc.

**Pattern 5: Plugin System (Existing)**
- Plugins can register custom node types (e.g., MuJoCo nodes)
- Nodes dynamically injected into context menus
- Plugin DLLs loaded via `PluginManager`

### 1.3 Critical Constraints

**Constraint 1: ImNodes Context Limitation**
- ImNodes uses a single `ImNodesEditorContext` per editor
- Cannot reuse same context for multiple editors
- **Solution:** Create separate `ImNodesEditorContext` for Data Studio pipeline canvas

**Constraint 2: Node ID Collision**
- Node IDs must be globally unique integers
- Current system: `next_node_id_`, `next_pin_id_`, `next_link_id_` in `NodeEditor`
- **Solution:** Separate ID counters for Data Studio (`data_studio_next_node_id_`, etc.)

**Constraint 3: Memory Pressure**
- `DataRegistry` already manages dataset memory with LRU eviction
- **Solution:** Ensure Data Studio pipeline execution respects memory limits

**Constraint 4: Single-Threaded ImGui**
- All ImGui calls must be on main thread
- Background tasks must use callbacks to update UI
- **Solution:** Data Studio pipeline execution via `AsyncTaskManager`, results pushed to main thread

---

## 2. Data Studio Architecture Design

### 2.1 High-Level System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     CyxWiz Engine 2.0                            │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                   MainWindow                               │  │
│  │  ┌──────────────────┐        ┌──────────────────────────┐  │  │
│  │  │  Asset Browser   │────────▶ Data Studio Panel        │  │  │
│  │  │  - Import Data   │        │  ┌─────────────────────┐  │  │  │
│  │  └──────────────────┘        │  │ Pipeline Canvas     │  │  │  │
│  │                              │  │ (ImNodes Context 2) │  │  │  │
│  │  ┌──────────────────┐        │  └─────────────────────┘  │  │  │
│  │  │  DataRegistry    │◀───────┤  Tab Bar:                │  │  │
│  │  │  - Raw Data      │        │  [Pipeline][Analysis]    │  │  │
│  │  │  - Cleaned Data  │        │  [Visualization][Query]  │  │  │
│  │  └──────────────────┘        └──────────────────────────┘  │  │
│  │           ▲                              │                  │  │
│  │           │ Dataset Handoff              │                  │  │
│  │           ▼                              ▼                  │  │
│  │  ┌──────────────────┐        ┌──────────────────────────┐  │  │
│  │  │  Node Editor     │        │  DuckDB Query Engine     │  │  │
│  │  │  (ImNodes Ctx 1) │        │  - In-memory tables      │  │  │
│  │  │  - DataInput     │        │  - SQL queries           │  │  │
│  │  │  - ML Layers     │        └──────────────────────────┘  │  │
│  │  └──────────────────┘                                       │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Breakdown

#### 2.2.1 DataStudioPanel (Primary Container)

**File:** `cyxwiz-engine/src/gui/panels/data_studio_panel.h/cpp`

**Responsibilities:**
- Top-level container for all Data Studio features
- Manages 4-tab workspace: Pipeline, Analysis, Visualization, Query
- Owns `DataStudioPipelineCanvas`, `DataStudioAnalyzer`, etc.

**Class Structure:**
```cpp
namespace cyxwiz {

class DataStudioPanel {
public:
    DataStudioPanel();
    ~DataStudioPanel();

    void Render();
    bool* GetVisiblePtr() { return &show_window_; }

    // Triggered when user imports data via Asset Browser or drag-drop
    void OnDataImported(const std::string& path, const std::string& name);

    // Get current pipeline output for handoff to Node Editor
    std::optional<std::string> GetPipelineOutput() const;

private:
    void RenderTabBar();
    void RenderPipelineTab();
    void RenderAnalysisTab();
    void RenderVisualizationTab();
    void RenderQueryTab();

    bool show_window_ = false;
    enum class Tab { Pipeline, Analysis, Visualization, Query };
    Tab active_tab_ = Tab::Pipeline;

    // Sub-components
    std::unique_ptr<DataStudioPipelineCanvas> pipeline_canvas_;
    std::unique_ptr<DataStudioAnalyzer> analyzer_;
    std::unique_ptr<DataStudioVisualizer> visualizer_;
    std::unique_ptr<DataStudioQueryEditor> query_editor_;

    // Shared state
    std::string current_dataset_;  // Currently active dataset in pipeline
};

} // namespace cyxwiz
```

---

#### 2.2.2 DataStudioPipelineCanvas (Visual Pipeline Builder)

**File:** `cyxwiz-engine/src/gui/data_studio/pipeline_canvas.h/cpp`

**Responsibilities:**
- Visual node editor for data pipelines (SEPARATE from ML Node Editor)
- Uses its own `ImNodesEditorContext` to avoid conflicts
- Node categories: Input, Tabular, Text, Time-Series, Feature Eng., Analyze, Output
- Pipeline execution engine

**Class Structure:**
```cpp
namespace cyxwiz {

// Data Studio Node Types (distinct from NodeType in node_editor.h)
enum class DataNodeType {
    // ===== Input Nodes =====
    FileInput,          // Load CSV, JSON, HDF5, etc.
    CloudInput,         // Load from CyxCloud S3
    SQLInput,           // Query from SQL database
    APIInput,           // REST API data source

    // ===== Tabular Nodes =====
    RemoveDuplicates,   // Drop duplicate rows
    FillMissing,        // Impute nulls (mean, median, mode, forward-fill, etc.)
    FilterRows,         // Filter by condition (IQR, hard bounds, SQL WHERE)
    TypeCast,           // Cast column data types
    SelectColumns,      // Column selection/projection
    RenameColumns,      // Rename columns
    SortRows,           // Sort by column
    DropColumns,        // Remove columns
    MergeDatasets,      // Join/concat multiple datasets

    // ===== Text Nodes =====
    TextClean,          // Lowercase, trim, remove punctuation
    TextTokenize,       // Tokenization (word, sentence, BPE)
    TextNormalize,      // Stemming, lemmatization
    TextVectorize,      // TF-IDF, Count vectorizer

    // ===== Time-Series Nodes =====
    TSWindow,           // Sliding window
    TSFeatures,         // Lag, rolling stats, diff
    TSSplit,            // Chronological train/val/test split
    TSResample,         // Upsample/downsample time-series

    // ===== Feature Engineering Nodes =====
    StandardScale,      // (x - mean) / std
    MinMaxScale,        // (x - min) / (max - min)
    RobustScale,        // Median-based scaling
    OneHotEncode,       // Categorical → one-hot
    LabelEncode,        // Categorical → int
    BinColumn,          // Discretization (equal-width, quantile)
    PolynomialFeatures, // x² x³ interactions
    PCA,                // Dimensionality reduction
    TruncatedSVD,       // LSA for sparse data

    // ===== Analyze Nodes =====
    DescriptiveStats,   // Compute mean, std, quantiles
    Correlation,        // Correlation matrix
    MissingValueReport, // Null analysis
    OutlierDetection,   // IQR, Z-score, Isolation Forest
    TrainValSplit,      // Random train/val/test split

    // ===== Output Nodes =====
    SaveDataset,        // Save to DataRegistry with version tag
    ExportFile,         // Export to CSV, Parquet, HDF5, JSON
    DeployToNodeEditor  // Send dataset to ML Node Editor's DataInput
};

struct DataPipelineNode {
    int id;
    DataNodeType type;
    std::string name;
    std::vector<NodePin> inputs;
    std::vector<NodePin> outputs;
    std::map<std::string, std::string> parameters;  // Node config

    // Execution state
    bool executed = false;
    bool has_error = false;
    std::string error_message;
};

struct DataPipelineLink {
    int id;
    int from_node;
    int from_pin;
    int to_node;
    int to_pin;
};

class DataStudioPipelineCanvas {
public:
    DataStudioPipelineCanvas();
    ~DataStudioPipelineCanvas();

    void Render();

    // Execute the entire pipeline (async)
    void RunPipeline();
    void StopPipeline();
    bool IsPipelineRunning() const { return is_running_; }

    // Get pipeline output dataset (for handoff to Node Editor)
    std::optional<std::string> GetOutputDatasetName() const;

    // Save/load pipeline to/from JSON
    bool SavePipeline(const std::string& filepath);
    bool LoadPipeline(const std::string& filepath);

    // Clear all nodes
    void ClearPipeline();

private:
    void ShowToolbar();
    void RenderNodes();
    void HandleInteractions();
    void ShowContextMenu();

    // Node management
    void AddNode(DataNodeType type, const std::string& name);
    void DeleteNode(int node_id);

    // Execution
    void ExecutePipelineAsync();
    bool ValidatePipeline(std::string& error);
    std::vector<int> TopologicalSort();
    bool ExecuteNode(DataPipelineNode& node);

    // Node execution implementations
    bool ExecuteFileInput(DataPipelineNode& node);
    bool ExecuteRemoveDuplicates(DataPipelineNode& node);
    bool ExecuteFillMissing(DataPipelineNode& node);
    bool ExecuteFilterRows(DataPipelineNode& node);
    bool ExecuteStandardScale(DataPipelineNode& node);
    // ... (one per DataNodeType)

    // Data flow tracking
    std::string GetInputDatasetName(int node_id) const;
    void SetOutputDatasetName(int node_id, const std::string& name);

    // Separate ImNodes context (CRITICAL: not shared with ML Node Editor)
    ImNodesEditorContext* editor_context_;

    // Pipeline state
    std::vector<DataPipelineNode> nodes_;
    std::vector<DataPipelineLink> links_;
    int next_node_id_ = 1000000;  // Start at 1M to avoid collision with ML nodes
    int next_pin_id_ = 1000000;
    int next_link_id_ = 1000000;

    // Execution state
    std::atomic<bool> is_running_{false};
    std::thread pipeline_thread_;
    std::map<int, std::string> node_dataset_map_;  // node_id -> intermediate dataset name

    // Output tracking
    std::string output_dataset_name_;  // Dataset produced by final node

    bool show_context_menu_ = false;
    ImVec2 context_menu_pos_;
};

} // namespace cyxwiz
```

**Key Design Decisions:**

1. **Separate Namespace for Data Nodes:**
   - `DataNodeType` enum distinct from `NodeType` in `node_editor.h`
   - Prevents confusion between ML layers and data operations
   - Different node color schemes for visual differentiation

2. **Dedicated ImNodes Context:**
   - `editor_context_` created via `ImNodes::EditorContextCreate()` in constructor
   - Set active with `ImNodes::EditorContextSet(editor_context_)` in `Render()`
   - **Never shared** with `NodeEditor` — this prevents rendering conflicts

3. **Intermediate Dataset Storage:**
   - Each node's output is stored in `DataRegistry` with a unique name
   - Naming convention: `ds_pipeline_{node_id}` (e.g., `ds_pipeline_1000042`)
   - Allows inspection of intermediate results
   - Final output tagged with user-friendly version name

4. **Async Execution:**
   - `RunPipeline()` spawns background thread via `AsyncTaskManager`
   - Progress callbacks update UI (progress bar in toolbar)
   - Main thread polls for completion, updates node status colors

---

#### 2.2.3 DataStudioQueryEditor (DuckDB Integration)

**File:** `cyxwiz-engine/src/gui/data_studio/query_editor.h/cpp`

**Dependencies:**
- DuckDB C++ library (embedded)
- `ImGui::InputTextMultiline` for SQL editor
- `TableViewerPanel` for rendering results

**Responsibilities:**
- SQL query editor with syntax highlighting (optional via `TextEditor` widget)
- Execute queries against datasets in `DataRegistry`
- Display results in table format
- Save query results as new datasets

**Class Structure:**
```cpp
namespace cyxwiz {

class DataStudioQueryEditor {
public:
    DataStudioQueryEditor();
    ~DataStudioQueryEditor();

    void Render();

    // Register dataset with DuckDB (creates in-memory table)
    void RegisterDataset(const std::string& name);

    // Execute query and display results
    void ExecuteQuery();

private:
    void RenderQueryInput();
    void RenderResults();
    void RenderErrorMessage();

    // DuckDB instance (in-memory database)
    std::unique_ptr<duckdb::DuckDB> db_;
    std::unique_ptr<duckdb::Connection> conn_;

    // UI state
    char query_buffer_[8192] = "";
    bool has_results_ = false;
    bool has_error_ = false;
    std::string error_message_;

    // Query results
    std::vector<std::string> result_columns_;
    std::vector<std::vector<std::string>> result_rows_;
    size_t result_count_ = 0;
    float query_time_ms_ = 0.0f;
};

} // namespace cyxwiz
```

**DuckDB Integration Strategy:**

1. **In-Memory Tables:**
   - When a dataset is loaded into `DataRegistry`, create corresponding DuckDB table
   - Table name = dataset name
   - Use DuckDB's `CREATE TABLE ... AS SELECT * FROM read_csv(...)` or manual `INSERT`

2. **Dataset Synchronization:**
   - `DataRegistry::LoadDataset()` callback triggers `RegisterDataset()`
   - Only register datasets explicitly opened in Data Studio (not all loaded datasets)

3. **Query Execution:**
   - User types SQL in `query_buffer_`
   - Click "Run" → execute via `conn_->Query(query_buffer_)`
   - Results converted to string table for display

4. **Save Results:**
   - Button: "Save as Dataset"
   - Query result → new dataset in `DataRegistry`
   - Accessible to pipeline canvas and ML Node Editor

**Example Query:**
```sql
SELECT city, AVG(price) as avg_price, COUNT(*) as count
FROM properties_cleaned
GROUP BY city
ORDER BY avg_price DESC
LIMIT 10;
```

---

#### 2.2.4 DataStudioAnalyzer (Statistical Analysis)

**File:** `cyxwiz-engine/src/gui/data_studio/analyzer.h/cpp`

**Responsibilities:**
- Run statistical analysis on pipeline outputs
- Displays: descriptive stats, distributions, correlation matrix, missing value report
- Reuses existing panels: `DataProfilerPanel`, `CorrelationMatrixPanel`, etc.

**Class Structure:**
```cpp
namespace cyxwiz {

class DataStudioAnalyzer {
public:
    DataStudioAnalyzer();

    void Render();

    // Set dataset to analyze
    void SetDataset(const std::string& name);

private:
    void RenderDescriptiveStats();
    void RenderDistribution();
    void RenderCorrelation();
    void RenderMissingValues();
    void RenderOutliers();

    std::string current_dataset_;

    // Reuse existing panels (composition, not duplication)
    std::unique_ptr<DataProfilerPanel> profiler_panel_;
    std::unique_ptr<CorrelationMatrixPanel> correlation_panel_;
    std::unique_ptr<MissingValuePanel> missing_value_panel_;
    std::unique_ptr<OutlierDetectionPanel> outlier_panel_;
};

} // namespace cyxwiz
```

**Design Note:**
- **No new code needed** — reuse existing analysis panels
- Analyzer is just a **wrapper/orchestrator** for existing functionality
- Benefit: Maintains consistency with existing UI

---

#### 2.2.5 DataStudioVisualizer (Chart Builder)

**File:** `cyxwiz-engine/src/gui/data_studio/visualizer.h/cpp`

**Responsibilities:**
- Create charts from pipeline data
- Chart types: Bar, Line, Scatter, Histogram, Heatmap, Box plot
- Uses ImPlot for rendering

**Class Structure:**
```cpp
namespace cyxwiz {

enum class ChartType {
    Bar, Line, Scatter, Histogram, Heatmap, BoxPlot, Violin
};

class DataStudioVisualizer {
public:
    DataStudioVisualizer();

    void Render();

    // Set dataset to visualize
    void SetDataset(const std::string& name);

private:
    void RenderChartTypeSelector();
    void RenderChartConfig();
    void RenderChart();

    // Chart rendering
    void RenderBarChart();
    void RenderLineChart();
    void RenderScatterPlot();
    void RenderHistogram();
    void RenderHeatmap();

    std::string current_dataset_;
    ChartType selected_chart_ = ChartType::Bar;

    // Chart configuration
    std::string x_column_;
    std::string y_column_;
    std::string group_by_column_;
    int bin_count_ = 20;  // For histogram
};

} // namespace cyxwiz
```

---

### 2.3 Data Flow Architecture

#### 2.3.1 End-to-End Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: Data Import                                            │
│ User Action: File → Import Data → Select "properties_raw.csv"  │
│ Result: Dataset loaded into DataRegistry as "properties_raw"   │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: Pipeline Building (Data Studio Panel — Pipeline Tab)  │
│ User Action: Drag "FileInput" node onto canvas                 │
│            → Configure: dataset = "properties_raw"              │
│            → Add "RemoveDuplicates" node                        │
│            → Add "FillMissing" node (median imputation)         │
│            → Add "FilterRows" node (IQR outlier removal)        │
│            → Add "StandardScale" node (price, sqft)             │
│            → Add "DeployToNodeEditor" node                      │
│ Result: Visual pipeline graph with 6 connected nodes           │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: Pipeline Execution                                     │
│ User Action: Click "Run Pipeline" button                       │
│ System:                                                         │
│   1. Validate graph (check for cycles, required params)        │
│   2. Topological sort (execution order)                        │
│   3. Spawn async thread via AsyncTaskManager                   │
│   4. Execute nodes sequentially:                                │
│      - FileInput:        Load "properties_raw" (already loaded) │
│      - RemoveDuplicates: Create "ds_pipeline_1000001"          │
│      - FillMissing:      Create "ds_pipeline_1000002"          │
│      - FilterRows:       Create "ds_pipeline_1000003"          │
│      - StandardScale:    Create "ds_pipeline_1000004"          │
│      - DeployToNodeEditor: Tag final dataset as "properties_v1"│
│   5. Update node status colors (green = success, red = error)  │
│ Result: Clean dataset "properties_v1" in DataRegistry          │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: Analysis (Optional — Data Studio Panel — Analysis Tab)│
│ User Action: Switch to Analysis tab                            │
│ System: Auto-run DataProfiler on "properties_v1"               │
│ Display:                                                        │
│   - Descriptive stats (mean, std, min, max per column)         │
│   - Null rate (0% after cleaning)                              │
│   - Distribution plots (histograms)                             │
│   - Correlation matrix (price vs sqft = 0.82)                  │
│ Result: User confirms data quality                             │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: SQL Query (Optional — Data Studio Panel — Query Tab)  │
│ User Action: Type SQL query:                                   │
│   SELECT city, AVG(price) FROM properties_v1 GROUP BY city     │
│ System: Execute query via DuckDB, display results              │
│ Result: User explores data interactively                       │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 6: Handoff to Node Editor (ML Training)                  │
│ User Action: Click "Deploy to Node Editor" in pipeline         │
│ System:                                                         │
│   1. DataStudioPanel calls GetPipelineOutput()                 │
│      → Returns "properties_v1"                                  │
│   2. MainWindow calls NodeEditor::UpdateDatasetNodeName()      │
│   3. Node Editor's DataInput node updates:                     │
│      - Name: "properties_v1"                                    │
│      - Shape: [74908, 31]                                       │
│      - Type: Float32                                            │
│   4. User continues building ML model in Node Editor           │
│      - Connects DataInput → Dense(128) → ReLU → Dense(1)      │
│      - Adds MSELoss, Adam optimizer                             │
│      - Clicks "Start Training"                                  │
│ Result: Model trains on cleaned data from Data Studio          │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.3.2 Dataset Naming Convention

| Stage | Dataset Name | Storage Location | Visibility |
|-------|--------------|------------------|-----------|
| Raw Import | `properties_raw` | `DataRegistry` | User-visible |
| Pipeline Intermediate | `ds_pipeline_1000001` | `DataRegistry` | Hidden (internal) |
| Pipeline Output | `properties_v1` | `DataRegistry` | User-visible |
| ML Training Input | `properties_v1` | `DataRegistry` (same) | Node Editor `DataInput` |

**Naming Rules:**
- User-imported datasets: Use filename without extension
- Pipeline intermediates: `ds_pipeline_{node_id}` (hidden from UI by prefix)
- Pipeline outputs: User-defined name with `_v{N}` suffix (version tracking)

---

### 2.4 Integration Points

#### 2.4.1 AssetBrowserPanel Integration

**Location:** `cyxwiz-engine/src/gui/panels/asset_browser.cpp`

**Modification:**
```cpp
// In AssetBrowserPanel::OnFileDoubleClick()
if (extension == ".csv" || extension == ".json" || extension == ".h5") {
    // NEW: If Data Studio panel exists, import into it
    if (auto* ds_panel = GetMainWindow()->GetDataStudioPanel()) {
        ds_panel->OnDataImported(filepath, filename_without_ext);
        ds_panel->Show();  // Activate Data Studio panel
    } else {
        // Fallback: Load into DataRegistry directly (current behavior)
        DataRegistry::Instance().LoadDataset(filepath, filename_without_ext);
    }
}
```

#### 2.4.2 MainWindow Integration

**Location:** `cyxwiz-engine/src/gui/main_window.h/cpp`

**Changes:**
```cpp
// In main_window.h
class MainWindow {
public:
    // ... existing methods ...

    // NEW: Access to Data Studio panel
    cyxwiz::DataStudioPanel* GetDataStudioPanel() { return data_studio_panel_.get(); }

private:
    // ... existing panels ...

    // NEW: Data Studio panel
    std::unique_ptr<cyxwiz::DataStudioPanel> data_studio_panel_;
};
```

```cpp
// In main_window.cpp constructor
MainWindow::MainWindow() {
    // ... existing initialization ...

    // NEW: Initialize Data Studio panel
    data_studio_panel_ = std::make_unique<cyxwiz::DataStudioPanel>();

    // Register with sidebar
    RegisterPanelsWithSidebar();
}

// In RegisterPanelsWithSidebar()
void MainWindow::RegisterPanelsWithSidebar() {
    // ... existing panels ...

    // NEW: Register Data Studio
    sidebar_registry_["Data Studio"] = data_studio_panel_->GetVisiblePtr();
}
```

#### 2.4.3 NodeEditor Integration (Dataset Handoff)

**Location:** `cyxwiz-engine/src/gui/node_editor.cpp`

**New Method:**
```cpp
// In node_editor.cpp
void NodeEditor::SetDatasetFromDataStudio(const std::string& dataset_name) {
    // Find existing DatasetInput node or create one
    MLNode* dataset_input_node = nullptr;
    for (auto& node : nodes_) {
        if (node.type == NodeType::DatasetInput) {
            dataset_input_node = &node;
            break;
        }
    }

    if (!dataset_input_node) {
        // Create DatasetInput node if it doesn't exist
        AddNode(NodeType::DatasetInput, "Data Input");
        dataset_input_node = &nodes_.back();
    }

    // Update node to reference the dataset
    UpdateDatasetNodeName(dataset_name);

    // Save undo state
    SaveUndoState();

    // Trigger shape inference for connected nodes
    if (shape_inference_) {
        shape_inference_->InferShapes(nodes_, links_);
    }
}
```

**Handoff Trigger:**
```cpp
// In DataStudioPanel::OnDeployToNodeEditor()
void DataStudioPanel::OnDeployToNodeEditor() {
    auto output_dataset = pipeline_canvas_->GetOutputDatasetName();
    if (!output_dataset.has_value()) {
        // Show error: No output dataset
        return;
    }

    // Get NodeEditor from MainWindow
    auto* main_window = GetMainWindow();  // Passed via constructor
    auto* node_editor = main_window->GetNodeEditor();

    // Handoff dataset
    node_editor->SetDatasetFromDataStudio(output_dataset.value());

    // Show Node Editor panel
    node_editor->Show();

    // Notify user
    ImGui::OpenPopup("Dataset Deployed");
}
```

---

## 3. Node Type Catalog

### 3.1 Input Nodes

| Node Type | Parameters | Description |
|-----------|------------|-------------|
| **FileInput** | `path` (string), `format` (auto/csv/json/hdf5/txt) | Load dataset from file |
| **CloudInput** | `bucket` (string), `key` (string) | Load from CyxCloud S3 |
| **SQLInput** | `connection_string`, `query` | Query SQL database |
| **APIInput** | `url`, `method` (GET/POST), `headers` | REST API data source |

### 3.2 Tabular Nodes

| Node Type | Parameters | Description |
|-----------|------------|-------------|
| **RemoveDuplicates** | `subset` (columns), `keep` (first/last/none) | Drop duplicate rows |
| **FillMissing** | `strategy` (mean/median/mode/ffill/bfill/constant), `columns`, `value` | Impute nulls |
| **FilterRows** | `method` (iqr/zscore/hard_bounds), `column`, `min`, `max` | Filter outliers/values |
| **TypeCast** | `columns`, `target_type` (int/float/str/datetime) | Cast data types |
| **SelectColumns** | `columns` (list) | Keep only specified columns |
| **DropColumns** | `columns` (list) | Remove columns |
| **RenameColumns** | `mapping` (old→new) | Rename columns |
| **SortRows** | `by` (column), `ascending` (bool) | Sort rows |
| **MergeDatasets** | `how` (inner/left/right/outer/concat), `on` (key columns) | Join/concat datasets |

### 3.3 Text Nodes

| Node Type | Parameters | Description |
|-----------|------------|-------------|
| **TextClean** | `lowercase`, `remove_punct`, `remove_numbers`, `trim` | Clean text |
| **TextTokenize** | `method` (word/sentence/bpe), `column` | Tokenize text |
| **TextNormalize** | `method` (stem/lemma), `language` | Normalize text |
| **TextVectorize** | `method` (tfidf/count), `max_features`, `ngram_range` | Convert text to vectors |

### 3.4 Time-Series Nodes

| Node Type | Parameters | Description |
|-----------|------------|-------------|
| **TSWindow** | `window_size`, `stride`, `group_by`, `sort_by` | Sliding window |
| **TSFeatures** | `features` (lag/rolling_mean/rolling_std/diff) | Extract time features |
| **TSSplit** | `train_ratio`, `val_ratio`, `test_ratio` | Chronological split |
| **TSResample** | `freq` (1s/1min/1h/1d), `method` (mean/sum/first/last) | Resample time-series |

### 3.5 Feature Engineering Nodes

| Node Type | Parameters | Description |
|-----------|------------|-------------|
| **StandardScale** | `columns`, `with_mean`, `with_std` | Z-score normalization |
| **MinMaxScale** | `columns`, `feature_range` ([0,1] or [-1,1]) | Min-max scaling |
| **RobustScale** | `columns`, `quantile_range` ([25,75]) | Median-based scaling |
| **OneHotEncode** | `columns`, `drop_first` | Categorical → one-hot |
| **LabelEncode** | `column` | Categorical → int |
| **BinColumn** | `column`, `bins` (count or edges), `strategy` (uniform/quantile) | Discretization |
| **PolynomialFeatures** | `degree`, `interaction_only`, `include_bias` | Polynomial expansion |
| **PCA** | `n_components`, `whiten` | Dimensionality reduction |
| **TruncatedSVD** | `n_components` | LSA for sparse data |

### 3.6 Analyze Nodes

| Node Type | Parameters | Description |
|-----------|------------|-------------|
| **DescriptiveStats** | `columns` | Compute mean, std, min, max, quantiles |
| **Correlation** | `method` (pearson/spearman/kendall) | Correlation matrix |
| **MissingValueReport** | - | Null counts and percentages |
| **OutlierDetection** | `method` (iqr/zscore/isolation_forest), `columns` | Detect outliers |
| **TrainValSplit** | `train_ratio`, `val_ratio`, `test_ratio`, `stratify`, `shuffle` | Random split |

### 3.7 Output Nodes

| Node Type | Parameters | Description |
|-----------|------------|-------------|
| **SaveDataset** | `name`, `version` | Save to DataRegistry |
| **ExportFile** | `path`, `format` (csv/parquet/hdf5/json) | Export to file |
| **DeployToNodeEditor** | `name` (output dataset name) | Send to ML Node Editor |

---

## 4. Technology Stack

### 4.1 New Dependencies

| Library | Version | Purpose | Integration Method |
|---------|---------|---------|-------------------|
| **DuckDB** | 1.0.0+ | In-memory SQL query engine | vcpkg, CMake `find_package(DuckDB)` |

**CMakeLists.txt Addition:**
```cmake
# Data Studio dependencies
find_package(DuckDB REQUIRED)

target_link_libraries(cyxwiz-engine PRIVATE
    # ... existing libs ...
    duckdb::duckdb
)
```

**vcpkg.json Addition:**
```json
{
  "dependencies": [
    "duckdb"
  ]
}
```

### 4.2 Existing Dependencies (Reused)

- **ImNodes** — Already used by `NodeEditor`, create separate context
- **ImPlot** — Already used for training plots, reuse for Data Studio visualizations
- **ImGui** — Base UI framework
- **nlohmann/json** — Pipeline serialization
- **spdlog** — Logging
- **ArrayFire** — Tensor operations for scaling/normalization
- **pybind11** — Python integration (for HuggingFace/Kaggle loaders)

---

## 5. File Structure

### 5.1 New Files

```
cyxwiz-engine/
  src/
    gui/
      data_studio/
        pipeline_canvas.h         — Visual pipeline builder (ImNodes)
        pipeline_canvas.cpp
        query_editor.h            — DuckDB SQL editor
        query_editor.cpp
        analyzer.h                — Statistical analysis wrapper
        analyzer.cpp
        visualizer.h              — Chart builder (ImPlot)
        visualizer.cpp

        nodes/
          node_executors.h        — Base class for node execution
          node_executors.cpp
          input_nodes.cpp         — FileInput, CloudInput, etc.
          tabular_nodes.cpp       — RemoveDuplicates, FillMissing, etc.
          text_nodes.cpp          — TextClean, TextTokenize, etc.
          timeseries_nodes.cpp    — TSWindow, TSFeatures, etc.
          feature_nodes.cpp       — StandardScale, OneHotEncode, etc.
          analyze_nodes.cpp       — DescriptiveStats, Correlation, etc.
          output_nodes.cpp        — SaveDataset, ExportFile, etc.

      panels/
        data_studio_panel.h       — Top-level Data Studio panel
        data_studio_panel.cpp

    core/
      data_studio/
        pipeline_executor.h       — Async pipeline execution engine
        pipeline_executor.cpp
        duckdb_manager.h          — DuckDB instance manager
        duckdb_manager.cpp
        node_registry.h           — Registry of available Data Studio nodes
        node_registry.cpp
```

### 5.2 Modified Files

| File | Modification |
|------|-------------|
| `main_window.h/cpp` | Add `DataStudioPanel` member, register with sidebar |
| `asset_browser.cpp` | Add Data Studio import on double-click CSV/JSON/HDF5 |
| `node_editor.h/cpp` | Add `SetDatasetFromDataStudio()` method |
| `data_registry.h/cpp` | Add `HideInternalDatasets()` filter for UI (hide `ds_pipeline_*`) |
| `CMakeLists.txt` | Add DuckDB dependency, link Data Studio sources |

---

## 6. Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

**Goals:**
- Establish project structure
- Create `DataStudioPanel` skeleton
- Separate ImNodes context for pipeline canvas
- Basic node types (FileInput, SaveDataset)
- Pipeline execution engine (no transformations yet)

**Deliverables:**
- `DataStudioPanel` renders in MainWindow
- Can add FileInput and SaveDataset nodes
- Nodes connect with links
- "Run Pipeline" button executes trivial pipeline (load → save)

**Test Case:**
```
1. Open Data Studio panel
2. Add FileInput node, configure to "properties_raw.csv"
3. Add SaveDataset node, name = "properties_copy"
4. Connect nodes
5. Click "Run Pipeline"
6. Verify "properties_copy" appears in DataRegistry
```

---

### Phase 2: Tabular Transformations (Week 3-4)

**Goals:**
- Implement 10 tabular nodes (RemoveDuplicates, FillMissing, FilterRows, etc.)
- Node execution logic with error handling
- Intermediate dataset storage
- Progress reporting via AsyncTaskManager

**Deliverables:**
- All tabular nodes functional
- Can build real data cleaning pipelines
- Visual progress bar during execution
- Node status colors (green = success, red = error)

**Test Case:**
```
Use Case 1: Data Cleaning Pipeline (from docs/Data Studio/CyxWiz_DataStudio_UseCases.html)
- 8-node pipeline: FileInput → RemoveDuplicates → FillMissing → FilterRows → TypeCast → TextClean → OneHotEncode → StandardScale → SaveDataset
- Verify output matches expected statistics
```

---

### Phase 3: Analysis & Visualization Tabs (Week 5)

**Goals:**
- Implement DataStudioAnalyzer (reuse existing panels)
- Implement DataStudioVisualizer (ImPlot charts)
- Tab switching between Pipeline/Analysis/Visualization

**Deliverables:**
- Analysis tab shows descriptive stats, correlation matrix, missing value report
- Visualization tab shows bar/line/scatter charts
- Charts update when dataset changes

**Test Case:**
```
1. Run pipeline from Phase 2
2. Switch to Analysis tab
3. Verify descriptive stats (mean, std, min, max)
4. Switch to Visualization tab
5. Create histogram of "price" column
6. Verify histogram renders correctly
```

---

### Phase 4: DuckDB Query Editor (Week 6)

**Goals:**
- Integrate DuckDB library
- Implement DataStudioQueryEditor
- Dataset → DuckDB table synchronization
- Query execution and result display

**Deliverables:**
- Query tab functional
- Can execute SQL queries on pipeline outputs
- Results display in table format
- "Save as Dataset" button works

**Test Case:**
```sql
SELECT city, AVG(price) as avg_price, COUNT(*) as count
FROM properties_v1
GROUP BY city
ORDER BY avg_price DESC
LIMIT 10;
```

---

### Phase 5: Node Editor Handoff (Week 7)

**Goals:**
- Implement `DeployToNodeEditor` node
- Handoff mechanism from Data Studio → Node Editor
- Update `DataInput` node automatically
- End-to-end workflow testing

**Deliverables:**
- "Deploy to Node Editor" button works
- Node Editor's DataInput node updates with correct dataset name and shape
- Can train ML model on Data Studio output

**Test Case:**
```
1. Build cleaning pipeline in Data Studio
2. Add DeployToNodeEditor node at end
3. Run pipeline
4. Click "Deploy to Node Editor"
5. Switch to Node Editor
6. Verify DataInput node shows correct dataset
7. Build simple model (DataInput → Dense → Output)
8. Start training
9. Verify training uses cleaned data
```

---

### Phase 6: Advanced Nodes (Week 8-9)

**Goals:**
- Implement text processing nodes
- Implement time-series nodes
- Implement feature engineering nodes (PCA, PolynomialFeatures)

**Deliverables:**
- All 50+ node types implemented
- Use Case 2 (Anomaly Detection) from docs working end-to-end

**Test Case:**
```
Use Case 2: Anomaly Detection Pipeline
- CloudInput → FillMissing → FilterRows → TSWindow → TSFeatures → StandardScale → TrainValSplit → DeployToNodeEditor
- Verify output matches expected shape [N, 60, 6] for LSTM encoder
```

---

### Phase 7: Save/Load & Polish (Week 10)

**Goals:**
- Pipeline save/load to JSON
- Error handling and validation
- UI polish (icons, tooltips, colors)
- Documentation

**Deliverables:**
- Can save pipeline as `.cyxpipe` file
- Can load pipeline from file
- Comprehensive error messages for node failures
- User guide documentation

**Test Case:**
```
1. Build complex pipeline
2. Save as "cleaning_pipeline.cyxpipe"
3. Close application
4. Reopen application
5. Load "cleaning_pipeline.cyxpipe"
6. Verify all nodes and links restored correctly
7. Run pipeline
8. Verify output matches original
```

---

### Phase 8: Performance Optimization (Week 11)

**Goals:**
- Lazy evaluation (only execute dirty nodes)
- Parallel node execution (where possible)
- Memory optimization for large datasets
- Streaming support for 100M+ row datasets

**Deliverables:**
- Pipeline execution time reduced by 50% for multi-branch graphs
- Can handle 100M row datasets with streaming
- Memory usage stays under configured limit

---

## 7. Backward Compatibility Strategy

### 7.1 Project File Format

**Current Format (.cyxproject):**
```json
{
  "version": "1.0",
  "name": "MyProject",
  "node_graph": { ... },
  "datasets": [ ... ],
  "settings": { ... }
}
```

**Enhanced Format (2.0):**
```json
{
  "version": "2.0",
  "name": "MyProject",
  "node_graph": { ... },            // ML Node Editor graph (unchanged)
  "data_studio_pipeline": { ... },  // NEW: Data Studio pipeline graph
  "datasets": [ ... ],
  "settings": { ... }
}
```

**Migration Strategy:**
- Version 1.0 projects load normally (no data_studio_pipeline key)
- Version 2.0 projects load both ML graph and Data Studio pipeline
- SaveProject() detects if Data Studio pipeline exists, writes version 2.0

### 7.2 Dataset Registry

**No Breaking Changes:**
- Existing datasets loaded into `DataRegistry` continue to work
- Data Studio adds new datasets with `ds_pipeline_*` prefix (hidden from UI)
- ML Node Editor unaware of Data Studio (only sees final output dataset)

### 7.3 Node Types

**No Collision:**
- ML nodes: `NodeType` enum in `node_editor.h`
- Data Studio nodes: `DataNodeType` enum in `pipeline_canvas.h`
- Separate namespaces prevent confusion

---

## 8. UI/UX Design

### 8.1 Panel Layout

**Data Studio Panel (Dockable Window):**

```
┌─────────────────────────────────────────────────────────────────┐
│  Data Studio                                            [X]      │
├─────────────────────────────────────────────────────────────────┤
│  [Pipeline] [Analysis] [Visualization] [Query]                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Toolbar: [Run ▶] [Stop ⏹] [Save 💾] [Load 📁] [Clear]   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Pipeline Canvas                        │   │
│  │                                                          │   │
│  │  ┌───────────┐    ┌────────────┐    ┌──────────────┐   │   │
│  │  │ FileInput │───▶│ FillMissing│───▶│ StandardScale│   │   │
│  │  └───────────┘    └────────────┘    └──────────────┘   │   │
│  │                                                          │   │
│  │                                       ┌──────────────┐   │   │
│  │                                       │ DeployToNode │   │   │
│  │                                       │    Editor    │   │   │
│  │                                       └──────────────┘   │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Status: ✅ Pipeline complete (28.4s) | 74,908 rows | 31 cols  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Node Visual Design

**Data Studio Node Style:**

```
┌─────────────────────┐
│ INPUT               │ ← Category badge (blue)
├─────────────────────┤
│  📁 File Input      │ ← Icon + name
├─────────────────────┤
│ dataset: props.csv  │ ← Key parameter
│ ✓ 80,412 rows       │ ← Status
└─────────────────────┘
   ●                    ← Output pin
```

**Color Scheme (by category):**
- Input nodes: Blue (#0e7fc2)
- Tabular nodes: Gold (#f0a500)
- Text nodes: Green (#2dc653)
- Time-Series nodes: Purple (#7c4dff)
- Feature Eng. nodes: Cyan (#00b8a9)
- Analyze nodes: Teal (#00b8a9)
- Output nodes: Green (#2dc653)

### 8.3 Context Menu Structure

**Right-click on canvas:**

```
Add Node ▶
  Input ▶
    File Input
    Cloud Input
    SQL Input
    API Input
  Tabular ▶
    Remove Duplicates
    Fill Missing
    Filter Rows
    Type Cast
    Select Columns
    Drop Columns
    Rename Columns
    Sort Rows
    Merge Datasets
  Text ▶
    Text Clean
    Text Tokenize
    Text Normalize
    Text Vectorize
  Time-Series ▶
    TS Window
    TS Features
    TS Split
    TS Resample
  Feature Engineering ▶
    Standard Scale
    Min-Max Scale
    Robust Scale
    One-Hot Encode
    Label Encode
    Bin Column
    Polynomial Features
    PCA
    Truncated SVD
  Analyze ▶
    Descriptive Stats
    Correlation
    Missing Value Report
    Outlier Detection
    Train/Val Split
  Output ▶
    Save Dataset
    Export File
    Deploy to Node Editor
─────────────────
Paste
Select All
Clear Pipeline
```

---

## 9. Error Handling & Validation

### 9.1 Pipeline Validation Rules

**Pre-Execution Checks:**
1. **Graph Structure:**
   - Must have at least one Input node
   - Must have at least one Output node
   - No cycles allowed (topological sort must succeed)
   - All nodes must be reachable from an Input node

2. **Node Configuration:**
   - All required parameters filled
   - File paths exist (for FileInput)
   - Column names exist in upstream dataset (for column-specific operations)

3. **Type Safety:**
   - Input pin types match output pin types
   - Tabular → Tabular (dataset flow)
   - Scalar → Scalar (signal flow)

**Runtime Errors:**
- Node execution failure → Mark node red, stop pipeline, display error popup
- Dataset not found → Show error, suggest available datasets
- Memory limit exceeded → Auto-evict intermediate datasets, retry

### 9.2 User Feedback

**Visual Indicators:**
- ⚪ Gray node: Not executed yet
- 🔵 Blue node: Currently executing (animated pulse)
- 🟢 Green node: Execution successful
- 🔴 Red node: Execution failed
- 🟡 Yellow node: Warning (e.g., high null rate after FillMissing)

**Error Messages:**
```
❌ Node "Fill Missing" failed:
   Column "age" not found in dataset "properties_raw"

   Suggestions:
   - Check column name spelling
   - Verify upstream node output
   - View dataset schema in Analysis tab

   [View Details] [Skip Node] [Stop Pipeline]
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

**Target Coverage:** 80%+

**Test Files:**
```
cyxwiz-engine/tests/
  data_studio/
    test_pipeline_canvas.cpp       — Node graph operations
    test_node_executors.cpp        — Individual node execution
    test_pipeline_executor.cpp     — Full pipeline execution
    test_duckdb_manager.cpp        — SQL query execution
    test_dataset_handoff.cpp       — Data Studio → Node Editor
```

**Example Test:**
```cpp
TEST_CASE("DataStudioPipelineCanvas - RemoveDuplicates node", "[data_studio]") {
    DataStudioPipelineCanvas canvas;

    // Create test pipeline: FileInput → RemoveDuplicates → SaveDataset
    auto input_node = canvas.AddNode(DataNodeType::FileInput, "Input");
    input_node.parameters["path"] = "test_data/duplicates.csv";

    auto dedup_node = canvas.AddNode(DataNodeType::RemoveDuplicates, "Dedup");
    dedup_node.parameters["subset"] = "id";
    dedup_node.parameters["keep"] = "first";

    auto output_node = canvas.AddNode(DataNodeType::SaveDataset, "Output");
    output_node.parameters["name"] = "cleaned";

    canvas.CreateLink(input_node.outputs[0].id, dedup_node.inputs[0].id);
    canvas.CreateLink(dedup_node.outputs[0].id, output_node.inputs[0].id);

    // Execute
    canvas.RunPipeline();
    REQUIRE(canvas.WaitForCompletion());  // Blocks until done

    // Verify
    auto output_dataset = DataRegistry::Instance().GetDataset("cleaned");
    REQUIRE(output_dataset.IsValid());
    REQUIRE(output_dataset.Size() == 95);  // 100 rows → 95 after dedup
}
```

### 10.2 Integration Tests

**End-to-End Scenarios:**
1. **Data Cleaning Pipeline:**
   - Load CSV with duplicates, nulls, outliers
   - Apply RemoveDuplicates → FillMissing → FilterRows
   - Verify output statistics match expected

2. **Anomaly Detection Pipeline:**
   - Load time-series sensor data
   - Apply TSWindow → TSFeatures → StandardScale
   - Verify output shape [N, 60, 6]

3. **Data Studio → Node Editor Handoff:**
   - Build cleaning pipeline
   - Deploy to Node Editor
   - Train simple model
   - Verify training uses correct dataset

### 10.3 Performance Benchmarks

**Target Metrics:**

| Operation | Dataset Size | Target Time | Max Memory |
|-----------|-------------|-------------|------------|
| Load CSV | 1M rows | < 2s | < 500 MB |
| RemoveDuplicates | 1M rows | < 1s | < 200 MB |
| FillMissing (median) | 1M rows | < 3s | < 300 MB |
| StandardScale | 1M rows, 50 cols | < 2s | < 400 MB |
| Full Pipeline (8 nodes) | 80K rows | < 10s | < 1 GB |
| DuckDB Query (GROUP BY) | 1M rows | < 500ms | < 300 MB |

**Benchmark Script:**
```cpp
// tests/benchmarks/data_studio_benchmarks.cpp
#include <benchmark/benchmark.h>

static void BM_RemoveDuplicates(benchmark::State& state) {
    // Setup: Load 1M row dataset with 10% duplicates
    auto dataset = LoadTestDataset("1M_duplicates.csv");

    for (auto _ : state) {
        DataStudioPipelineCanvas canvas;
        // ... execute RemoveDuplicates node
        benchmark::DoNotOptimize(canvas);
    }

    state.SetItemsProcessed(1000000);
}
BENCHMARK(BM_RemoveDuplicates);
```

---

## 11. Security & Sandboxing

### 11.1 SQL Injection Prevention (DuckDB)

**Risk:** User-provided column names or values in queries
**Mitigation:**
- Use DuckDB's prepared statements for parameterized queries
- Whitelist column names before insertion into queries
- Escape special characters in user input

**Example:**
```cpp
// UNSAFE:
std::string query = "SELECT * FROM dataset WHERE city = '" + user_input + "'";

// SAFE (prepared statement):
auto stmt = conn_->Prepare("SELECT * FROM dataset WHERE city = ?");
stmt->Execute(user_input);
```

### 11.2 File System Access

**Risk:** User loads malicious file paths
**Mitigation:**
- File dialog enforces valid extensions (.csv, .json, .h5)
- Validate file exists before loading
- Sandbox file access to project directory + system temp

### 11.3 Python Script Execution (for HuggingFace/Kaggle)

**Risk:** Arbitrary code execution via Python loaders
**Mitigation:**
- Python scripts run in same interpreter as existing ScriptingEngine (already sandboxed)
- HuggingFace/Kaggle loaders use official Python libraries (no eval/exec)

---

## 12. Future Enhancements (Post-MVP)

### 12.1 Cloud Datasets (Phase 2)

**Goal:** Load datasets directly from cloud storage

**Supported Sources:**
- CyxCloud (S3-compatible)
- AWS S3
- Google Cloud Storage
- Azure Blob Storage

**Implementation:**
- Add `CloudInput` node with authentication
- Streaming support for large cloud files
- Incremental loading (download chunks on demand)

### 12.2 AutoML Pipeline Generation

**Goal:** Suggest optimal pipeline for a given dataset

**Features:**
- Analyze raw dataset (missing values, outliers, data types)
- Recommend transformation pipeline
- One-click "Auto-Clean" button

**Implementation:**
- Rule-based heuristics (if null_rate > 5% → suggest FillMissing)
- ML-based suggestion (train meta-model on 1000s of datasets)

### 12.3 Collaborative Pipelines

**Goal:** Share pipelines with team members

**Features:**
- Export pipeline as shareable `.cyxpipe` file
- Import colleague's pipeline
- Version control integration (Git)

### 12.4 Real-Time Streaming Pipelines

**Goal:** Process live data streams (IoT sensors, logs)

**Implementation:**
- Add `StreamInput` node (Kafka, MQTT, WebSocket)
- Incremental pipeline execution (process mini-batches)
- Live dashboard updates

---

## 13. Open Questions & Decisions Needed

### 13.1 DuckDB: Embedded vs Server?

**Option A: Embedded DuckDB (Recommended)**
- Pros: No external dependencies, zero-config, single-process
- Cons: Limited to single-machine, no multi-user

**Option B: External DuckDB Server**
- Pros: Multi-user, remote datasets
- Cons: Requires server setup, more complex

**Decision:** Start with Embedded (Phase 1-7), add Server option in Phase 2 (Cloud)

### 13.2 Node Execution: Sync vs Async?

**Current Plan:** Async via `AsyncTaskManager`
**Alternative:** Sync execution (blocks UI)

**Decision:** Stick with Async — essential for large datasets (1M+ rows)

### 13.3 Intermediate Dataset Persistence

**Option A: Keep in memory only**
- Pros: Faster, no disk I/O
- Cons: High memory usage, lost on crash

**Option B: Spill to disk (current plan)**
- Pros: Lower memory, survives crashes
- Cons: Slower, disk I/O overhead

**Decision:** Hybrid — keep in memory if under limit, spill to disk if memory pressure

### 13.4 Python vs C++ for Node Executors?

**Option A: C++ only (current plan)**
- Pros: Performance, no Python dependency for basic operations
- Cons: More code, slower development

**Option B: Python wrappers (pandas, numpy)**
- Pros: Rapid development, rich ecosystem
- Cons: Performance overhead, Python dependency

**Decision:** C++ for core operations (RemoveDuplicates, FillMissing), Python for advanced (TextVectorize with scikit-learn)

---

## 14. Conclusion

CyxWiz Engine 2.0 with Data Studio integration represents a **major architectural upgrade** that transforms the engine from a pure ML training tool into a **complete end-to-end data science platform**. By integrating KNIME-inspired visual data pipelines directly into the engine, we eliminate the friction of switching between tools and enable a seamless workflow from raw data to trained models.

**Key Success Criteria:**
1. ✅ Zero external tools needed (no Jupyter, no separate data prep scripts)
2. ✅ Visual pipeline builder separate from ML Node Editor (no confusion)
3. ✅ Direct dataset handoff (no export/import file dance)
4. ✅ SQL query editor for exploratory analysis (DuckDB)
5. ✅ Backward compatible with existing projects
6. ✅ Production-ready performance (1M rows in < 10s)

**Next Steps:**
1. Review and approve this architecture document
2. Begin Phase 1 implementation (Core Infrastructure)
3. Weekly progress reviews with stakeholders
4. Beta release to select users after Phase 5 (Node Editor Handoff)
5. Public release after Phase 7 (Save/Load & Polish)

---

**Document Status:** Ready for Review
**Estimated Implementation Time:** 11 weeks (with 2 developers)
**Target Release:** Q3 2026
