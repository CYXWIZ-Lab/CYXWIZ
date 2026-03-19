# CyxWiz Engine 2.0 — Data Studio Implementation Roadmap

**Document Version:** 1.0
**Date:** 2026-03-19
**Companion to:** `docs/engine_2.0_architecture.md`

---

## Phase 1: Core Infrastructure (Weeks 1-2)

### Week 1: Project Setup & Panel Foundation

**Day 1-2: File Structure Setup**

```bash
# Create directory structure
mkdir -p cyxwiz-engine/src/gui/data_studio
mkdir -p cyxwiz-engine/src/gui/data_studio/nodes
mkdir -p cyxwiz-engine/src/core/data_studio
mkdir -p cyxwiz-engine/tests/data_studio
```

**File Checklist:**
- [x] `src/gui/data_studio/pipeline_canvas.h/cpp` (skeleton)
- [x] `src/gui/data_studio/query_editor.h/cpp` (skeleton)
- [x] `src/gui/data_studio/analyzer.h/cpp` (skeleton)
- [x] `src/gui/data_studio/visualizer.h/cpp` (skeleton)
- [x] `src/gui/panels/data_studio_panel.h/cpp` (main container)
- [x] `src/core/data_studio/pipeline_executor.h/cpp` (execution engine)
- [x] `src/core/data_studio/node_registry.h/cpp` (node type catalog)

**Day 3-4: CMakeLists.txt Integration**

```cmake
# Add to cyxwiz-engine/CMakeLists.txt

# Data Studio sources
set(DATA_STUDIO_SOURCES
    src/gui/data_studio/pipeline_canvas.cpp
    src/gui/data_studio/query_editor.cpp
    src/gui/data_studio/analyzer.cpp
    src/gui/data_studio/visualizer.cpp
    src/gui/panels/data_studio_panel.cpp
    src/core/data_studio/pipeline_executor.cpp
    src/core/data_studio/node_registry.cpp
)

# DuckDB dependency
find_package(DuckDB REQUIRED)

target_sources(cyxwiz-engine PRIVATE
    ${DATA_STUDIO_SOURCES}
)

target_link_libraries(cyxwiz-engine PRIVATE
    duckdb::duckdb
)
```

**Day 5: DataStudioPanel Integration into MainWindow**

File: `cyxwiz-engine/src/gui/main_window.h`

```cpp
// Add include
#include "panels/data_studio_panel.h"

// Add member variable (around line 353)
std::unique_ptr<cyxwiz::DataStudioPanel> data_studio_panel_;

// Add accessor method (around line 162)
cyxwiz::DataStudioPanel* GetDataStudioPanel() { return data_studio_panel_.get(); }
```

File: `cyxwiz-engine/src/gui/main_window.cpp`

```cpp
// In constructor (around line 45)
MainWindow::MainWindow() {
    // ... existing initialization ...

    // Initialize Data Studio panel
    data_studio_panel_ = std::make_unique<cyxwiz::DataStudioPanel>();

    // ... rest of initialization
}

// In RegisterPanelsWithSidebar() (around line 850)
void MainWindow::RegisterPanelsWithSidebar() {
    // ... existing registrations ...

    // Register Data Studio
    sidebar_registry_["Data Studio"] = data_studio_panel_->GetVisiblePtr();
}

// In Render() (around line 650)
void MainWindow::Render() {
    // ... existing panels ...

    // Render Data Studio panel
    if (data_studio_panel_) {
        data_studio_panel_->Render();
    }
}
```

---

### Week 2: Separate ImNodes Context & Basic Nodes

**Day 6-7: Pipeline Canvas with Separate ImNodes Context**

File: `cyxwiz-engine/src/gui/data_studio/pipeline_canvas.cpp`

**CRITICAL IMPLEMENTATION DETAIL:**

```cpp
#include "pipeline_canvas.h"
#include <imnodes.h>

DataStudioPipelineCanvas::DataStudioPipelineCanvas() {
    // CRITICAL: Create separate ImNodes context to avoid conflicts with NodeEditor
    editor_context_ = ImNodes::EditorContextCreate();

    // Initialize node/pin/link ID counters with high offset to avoid collisions
    next_node_id_ = 1000000;
    next_pin_id_ = 1000000;
    next_link_id_ = 1000000;
}

DataStudioPipelineCanvas::~DataStudioPipelineCanvas() {
    if (editor_context_) {
        ImNodes::EditorContextFree(editor_context_);
    }
}

void DataStudioPipelineCanvas::Render() {
    ImGui::Begin("Data Studio — Pipeline Canvas", &show_window_);

    // CRITICAL: Set active context BEFORE any ImNodes calls
    ImNodes::EditorContextSet(editor_context_);

    ShowToolbar();

    // Begin node editor rendering
    ImNodes::BeginNodeEditor();

    RenderNodes();

    ImNodes::EndNodeEditor();

    HandleInteractions();

    ImGui::End();

    // CRITICAL: Reset context to nullptr after rendering
    // This prevents accidental cross-contamination with NodeEditor
    ImNodes::EditorContextSet(nullptr);
}
```

**Day 8-9: Implement FileInput and SaveDataset Nodes**

```cpp
// In pipeline_canvas.cpp

void DataStudioPipelineCanvas::AddNode(DataNodeType type, const std::string& name) {
    DataPipelineNode node;
    node.id = next_node_id_++;
    node.type = type;
    node.name = name;

    // Create pins based on node type
    switch (type) {
        case DataNodeType::FileInput: {
            // No input pins
            // Output pin: Dataset
            NodePin output;
            output.id = next_pin_id_++;
            output.type = PinType::Dataset;
            output.name = "dataset";
            output.is_input = false;
            node.outputs.push_back(output);

            // Default parameters
            node.parameters["path"] = "";
            node.parameters["format"] = "auto";
            break;
        }
        case DataNodeType::SaveDataset: {
            // Input pin: Dataset
            NodePin input;
            input.id = next_pin_id_++;
            input.type = PinType::Dataset;
            input.name = "dataset";
            input.is_input = true;
            node.inputs.push_back(input);

            // No output pins (terminal node)

            // Default parameters
            node.parameters["name"] = "output";
            node.parameters["version"] = "v1";
            break;
        }
        // ... other node types
    }

    nodes_.push_back(node);

    // Save undo state
    // ... (implement undo/redo later)
}
```

**Day 10: Basic Pipeline Execution**

```cpp
// In pipeline_executor.cpp

bool PipelineExecutor::Execute(
    const std::vector<DataPipelineNode>& nodes,
    const std::vector<DataPipelineLink>& links
) {
    // Topological sort
    auto sorted_ids = TopologicalSort(nodes, links);
    if (sorted_ids.empty()) {
        error_ = "Pipeline has cycles or is invalid";
        return false;
    }

    // Execute nodes in order
    for (int node_id : sorted_ids) {
        auto* node = FindNode(node_id, nodes);
        if (!node) continue;

        bool success = ExecuteNode(*node);
        if (!success) {
            error_ = "Node " + node->name + " failed: " + node->error_message;
            return false;
        }
    }

    return true;
}

bool PipelineExecutor::ExecuteNode(DataPipelineNode& node) {
    switch (node.type) {
        case DataNodeType::FileInput:
            return ExecuteFileInput(node);
        case DataNodeType::SaveDataset:
            return ExecuteSaveDataset(node);
        // ... other nodes
        default:
            node.error_message = "Node type not implemented";
            return false;
    }
}

bool PipelineExecutor::ExecuteFileInput(DataPipelineNode& node) {
    std::string path = node.parameters["path"];
    if (path.empty()) {
        node.error_message = "Path parameter is required";
        return false;
    }

    // Load dataset into DataRegistry
    try {
        auto handle = DataRegistry::Instance().LoadDataset(path);
        if (!handle.IsValid()) {
            node.error_message = "Failed to load dataset from " + path;
            return false;
        }

        // Store dataset name for downstream nodes
        std::string dataset_name = handle.GetName();
        node_dataset_map_[node.id] = dataset_name;

        node.executed = true;
        node.has_error = false;
        return true;
    } catch (const std::exception& e) {
        node.error_message = std::string("Exception: ") + e.what();
        return false;
    }
}

bool PipelineExecutor::ExecuteSaveDataset(DataPipelineNode& node) {
    // Get input dataset from upstream node
    std::string input_dataset = GetInputDatasetName(node);
    if (input_dataset.empty()) {
        node.error_message = "No input dataset connected";
        return false;
    }

    std::string output_name = node.parameters["name"];
    if (output_name.empty()) {
        node.error_message = "Name parameter is required";
        return false;
    }

    // Dataset is already in DataRegistry, just tag it with version
    // (In future: implement versioning system)
    output_dataset_name_ = input_dataset;  // For now, just pass through

    node.executed = true;
    node.has_error = false;
    return true;
}
```

**Phase 1 Deliverable Test:**

```cpp
// tests/data_studio/test_phase1.cpp
TEST_CASE("Phase 1: Basic Pipeline Execution", "[data_studio][phase1]") {
    // Setup: Load a test CSV into DataRegistry
    auto handle = DataRegistry::Instance().LoadDataset("test_data/sample.csv", "sample");
    REQUIRE(handle.IsValid());

    DataStudioPipelineCanvas canvas;

    // Add FileInput node
    canvas.AddNode(DataNodeType::FileInput, "Input");
    auto& input_node = canvas.GetNodes()[0];
    input_node.parameters["path"] = "test_data/sample.csv";

    // Add SaveDataset node
    canvas.AddNode(DataNodeType::SaveDataset, "Output");
    auto& output_node = canvas.GetNodes()[1];
    output_node.parameters["name"] = "output";

    // Connect nodes
    canvas.CreateLink(
        input_node.outputs[0].id,  // FileInput output
        output_node.inputs[0].id   // SaveDataset input
    );

    // Execute pipeline
    canvas.RunPipeline();

    // Wait for completion (async)
    while (canvas.IsPipelineRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Verify
    REQUIRE(!canvas.HasError());
    REQUIRE(canvas.GetOutputDatasetName().has_value());

    // Cleanup
    DataRegistry::Instance().UnloadAll();
}
```

---

## Phase 2: Tabular Transformations (Weeks 3-4)

### Week 3: Core Tabular Nodes

**Day 11-12: RemoveDuplicates Node**

```cpp
// In src/gui/data_studio/nodes/tabular_nodes.cpp

bool PipelineExecutor::ExecuteRemoveDuplicates(DataPipelineNode& node) {
    std::string input_dataset = GetInputDatasetName(node);
    if (input_dataset.empty()) {
        node.error_message = "No input dataset";
        return false;
    }

    auto handle = DataRegistry::Instance().GetDataset(input_dataset);
    if (!handle.IsValid()) {
        node.error_message = "Input dataset not found: " + input_dataset;
        return false;
    }

    // Get parameters
    std::string subset_str = node.parameters["subset"];  // Comma-separated column names
    std::string keep = node.parameters["keep"];  // "first", "last", or "none"

    // Parse subset columns
    std::vector<std::string> subset_cols;
    if (!subset_str.empty()) {
        // Split by comma
        size_t pos = 0;
        std::string s = subset_str;
        while ((pos = s.find(",")) != std::string::npos) {
            subset_cols.push_back(s.substr(0, pos));
            s.erase(0, pos + 1);
        }
        subset_cols.push_back(s);
    }

    // Get raw dataset (must be CSV/Tabular for now)
    auto* raw_dataset = handle.GetUnderlyingDataset();
    // TODO: Implement GetRows() method on Dataset interface

    // For now, implement simple duplicate removal based on all columns
    size_t original_size = handle.Size();

    // Create intermediate dataset with duplicates removed
    std::string output_name = "ds_pipeline_" + std::to_string(node.id);

    // TODO: Implement actual deduplication logic
    // For MVP, we can use a simple approach:
    // 1. Iterate through dataset
    // 2. Hash each row
    // 3. Keep track of seen hashes
    // 4. Create new dataset with unique rows

    // Store output dataset name
    node_dataset_map_[node.id] = output_name;

    node.executed = true;
    node.has_error = false;
    return true;
}
```

**Implementation Note:**
For tabular operations, we need to extend the `Dataset` interface to support:
- Row-wise access (not just sample-wise)
- Column access
- Metadata (column names, types)

**Suggested Dataset Extension:**

```cpp
// In data_registry.h

// Add to Dataset interface:
class Dataset {
public:
    // Existing methods...

    // NEW: Tabular access (optional, only for CSV/Tabular datasets)
    virtual bool IsTabular() const { return false; }
    virtual std::vector<std::string> GetColumnNames() const { return {}; }
    virtual size_t GetRowCount() const { return Size(); }
    virtual std::vector<std::string> GetRow(size_t index) const { return {}; }
    virtual std::map<std::string, std::vector<std::string>> GetAllRows() const { return {}; }
};
```

**Day 13-14: FillMissing Node**

```cpp
bool PipelineExecutor::ExecuteFillMissing(DataPipelineNode& node) {
    std::string input_dataset = GetInputDatasetName(node);
    auto handle = DataRegistry::Instance().GetDataset(input_dataset);

    std::string strategy = node.parameters["strategy"];  // mean, median, mode, ffill, bfill, constant
    std::string columns_str = node.parameters["columns"];  // Comma-separated or "all"
    std::string value = node.parameters["value"];  // For constant strategy

    // Supported strategies:
    // - mean: Fill with column mean (numeric only)
    // - median: Fill with column median (numeric only)
    // - mode: Fill with most frequent value
    // - ffill: Forward fill (propagate last valid value)
    // - bfill: Backward fill
    // - constant: Fill with user-specified value

    // TODO: Implement missing value detection and imputation
    // For numeric columns: use ArrayFire for fast computation
    // For categorical: use mode

    std::string output_name = "ds_pipeline_" + std::to_string(node.id);
    node_dataset_map_[node.id] = output_name;

    node.executed = true;
    return true;
}
```

**Day 15-16: FilterRows Node**

```cpp
bool PipelineExecutor::ExecuteFilterRows(DataPipelineNode& node) {
    std::string input_dataset = GetInputDatasetName(node);
    auto handle = DataRegistry::Instance().GetDataset(input_dataset);

    std::string method = node.parameters["method"];  // iqr, zscore, hard_bounds
    std::string column = node.parameters["column"];
    float min_val = std::stof(node.parameters["min"]);
    float max_val = std::stof(node.parameters["max"]);

    if (method == "iqr") {
        // Compute Q1, Q3, IQR
        // Filter rows where column value is outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    } else if (method == "zscore") {
        // Compute mean, std
        // Filter rows where |z-score| > threshold
    } else if (method == "hard_bounds") {
        // Filter rows where column value is outside [min_val, max_val]
    }

    // TODO: Implement filtering logic

    std::string output_name = "ds_pipeline_" + std::to_string(node.id);
    node_dataset_map_[node.id] = output_name;

    node.executed = true;
    return true;
}
```

---

### Week 4: Complete Tabular Node Suite

**Remaining Nodes to Implement:**
- TypeCast
- SelectColumns
- DropColumns
- RenameColumns
- SortRows
- MergeDatasets

**Implementation Pattern (same for all):**
1. Get input dataset from upstream node
2. Parse node parameters
3. Apply transformation (using appropriate library: ArrayFire for numeric, custom logic for strings)
4. Create output dataset with unique name `ds_pipeline_{node_id}`
5. Store in `DataRegistry`
6. Return success

**Testing Strategy:**

Create comprehensive test suite:

```cpp
// tests/data_studio/test_tabular_nodes.cpp

TEST_CASE("RemoveDuplicates - Keep First", "[tabular]") { /* ... */ }
TEST_CASE("RemoveDuplicates - Keep Last", "[tabular]") { /* ... */ }
TEST_CASE("FillMissing - Mean", "[tabular]") { /* ... */ }
TEST_CASE("FillMissing - Median", "[tabular]") { /* ... */ }
TEST_CASE("FillMissing - Forward Fill", "[tabular]") { /* ... */ }
TEST_CASE("FilterRows - IQR Method", "[tabular]") { /* ... */ }
TEST_CASE("FilterRows - Hard Bounds", "[tabular]") { /* ... */ }
// ... etc.
```

---

## Phase 3: Analysis & Visualization Tabs (Week 5)

### Day 17-18: DataStudioAnalyzer (Reuse Existing Panels)

```cpp
// In data_studio/analyzer.cpp

DataStudioAnalyzer::DataStudioAnalyzer() {
    // Initialize existing panels (composition pattern)
    profiler_panel_ = std::make_unique<DataProfilerPanel>();
    correlation_panel_ = std::make_unique<CorrelationMatrixPanel>();
    missing_value_panel_ = std::make_unique<MissingValuePanel>();
    outlier_panel_ = std::make_unique<OutlierDetectionPanel>();
}

void DataStudioAnalyzer::Render() {
    if (current_dataset_.empty()) {
        ImGui::TextDisabled("No dataset selected");
        return;
    }

    ImGui::BeginTabBar("AnalysisTabBar");

    if (ImGui::BeginTabItem("Descriptive Stats")) {
        RenderDescriptiveStats();
        ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Distribution")) {
        RenderDistribution();
        ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Correlation")) {
        RenderCorrelation();
        ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Missing Values")) {
        RenderMissingValues();
        ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Outliers")) {
        RenderOutliers();
        ImGui::EndTabItem();
    }

    ImGui::EndTabBar();
}

void DataStudioAnalyzer::SetDataset(const std::string& name) {
    current_dataset_ = name;

    // Trigger analysis on all panels
    auto handle = DataRegistry::Instance().GetDataset(name);
    if (handle.IsValid()) {
        profiler_panel_->SetDataset(name);
        correlation_panel_->SetDataset(name);
        missing_value_panel_->SetDataset(name);
        outlier_panel_->SetDataset(name);
    }
}
```

**No New Code Needed** — This is just a wrapper around existing panels!

---

### Day 19-20: DataStudioVisualizer (ImPlot Charts)

```cpp
// In data_studio/visualizer.cpp

void DataStudioVisualizer::Render() {
    if (current_dataset_.empty()) {
        ImGui::TextDisabled("No dataset selected");
        return;
    }

    // Chart type selector
    const char* chart_types[] = { "Bar", "Line", "Scatter", "Histogram", "Heatmap" };
    int current_type = static_cast<int>(selected_chart_);
    if (ImGui::Combo("Chart Type", &current_type, chart_types, IM_ARRAYSIZE(chart_types))) {
        selected_chart_ = static_cast<ChartType>(current_type);
    }

    // Column selection
    auto handle = DataRegistry::Instance().GetDataset(current_dataset_);
    if (handle.IsValid() && handle.GetUnderlyingDataset()->IsTabular()) {
        auto columns = handle.GetUnderlyingDataset()->GetColumnNames();

        // X-axis column
        if (ImGui::BeginCombo("X Column", x_column_.c_str())) {
            for (const auto& col : columns) {
                bool selected = (col == x_column_);
                if (ImGui::Selectable(col.c_str(), selected)) {
                    x_column_ = col;
                }
            }
            ImGui::EndCombo();
        }

        // Y-axis column
        if (ImGui::BeginCombo("Y Column", y_column_.c_str())) {
            for (const auto& col : columns) {
                bool selected = (col == y_column_);
                if (ImGui::Selectable(col.c_str(), selected)) {
                    y_column_ = col;
                }
            }
            ImGui::EndCombo();
        }
    }

    ImGui::Separator();

    // Render chart
    switch (selected_chart_) {
        case ChartType::Bar:
            RenderBarChart();
            break;
        case ChartType::Line:
            RenderLineChart();
            break;
        case ChartType::Scatter:
            RenderScatterPlot();
            break;
        case ChartType::Histogram:
            RenderHistogram();
            break;
        case ChartType::Heatmap:
            RenderHeatmap();
            break;
    }
}

void DataStudioVisualizer::RenderBarChart() {
    if (x_column_.empty() || y_column_.empty()) return;

    // Get data from dataset
    auto handle = DataRegistry::Instance().GetDataset(current_dataset_);
    // TODO: Extract column data
    // For now, use dummy data

    if (ImPlot::BeginPlot("Bar Chart")) {
        // TODO: Plot bars using ImPlot::PlotBars()
        ImPlot::EndPlot();
    }
}
```

---

## Phase 4: DuckDB Query Editor (Week 6)

### Day 21-23: DuckDB Integration

**Step 1: Add DuckDB to vcpkg.json**

```json
{
  "dependencies": [
    "duckdb"
  ]
}
```

**Step 2: Install DuckDB**

```bash
cd cyxwiz-engine
vcpkg install
```

**Step 3: Implement Query Editor**

```cpp
// In data_studio/query_editor.cpp

#include <duckdb.hpp>

DataStudioQueryEditor::DataStudioQueryEditor() {
    // Create in-memory DuckDB instance
    db_ = std::make_unique<duckdb::DuckDB>(nullptr);  // nullptr = in-memory
    conn_ = std::make_unique<duckdb::Connection>(*db_);
}

void DataStudioQueryEditor::RegisterDataset(const std::string& name) {
    auto handle = DataRegistry::Instance().GetDataset(name);
    if (!handle.IsValid()) return;

    // Create DuckDB table from dataset
    // For tabular data:
    auto* dataset = handle.GetUnderlyingDataset();
    if (!dataset->IsTabular()) return;

    auto columns = dataset->GetColumnNames();
    auto rows = dataset->GetAllRows();

    // Build CREATE TABLE statement
    std::string create_sql = "CREATE TABLE " + name + " (";
    for (size_t i = 0; i < columns.size(); i++) {
        if (i > 0) create_sql += ", ";
        create_sql += columns[i] + " VARCHAR";  // Default to VARCHAR, improve later
    }
    create_sql += ")";

    conn_->Query(create_sql);

    // Insert rows
    for (const auto& [col_name, col_data] : rows) {
        // TODO: Batch insert for performance
    }
}

void DataStudioQueryEditor::ExecuteQuery() {
    has_error_ = false;
    has_results_ = false;
    result_columns_.clear();
    result_rows_.clear();

    auto start = std::chrono::high_resolution_clock::now();

    try {
        auto result = conn_->Query(query_buffer_);

        if (result->HasError()) {
            has_error_ = true;
            error_message_ = result->GetError();
            return;
        }

        // Extract column names
        for (size_t i = 0; i < result->ColumnCount(); i++) {
            result_columns_.push_back(result->ColumnName(i));
        }

        // Extract rows
        while (true) {
            auto chunk = result->Fetch();
            if (!chunk || chunk->size() == 0) break;

            for (size_t row = 0; row < chunk->size(); row++) {
                std::vector<std::string> row_data;
                for (size_t col = 0; col < chunk->ColumnCount(); col++) {
                    auto value = chunk->GetValue(col, row);
                    row_data.push_back(value.ToString());
                }
                result_rows_.push_back(row_data);
            }
        }

        has_results_ = true;
        result_count_ = result_rows_.size();

    } catch (const std::exception& e) {
        has_error_ = true;
        error_message_ = e.what();
    }

    auto end = std::chrono::high_resolution_clock::now();
    query_time_ms_ = std::chrono::duration<float, std::milli>(end - start).count();
}
```

**Day 24-25: Query Results Display + Save as Dataset**

```cpp
void DataStudioQueryEditor::RenderResults() {
    if (!has_results_) return;

    ImGui::Text("Query executed in %.2f ms | %zu rows", query_time_ms_, result_count_);

    if (ImGui::Button("Save as Dataset")) {
        // TODO: Convert query result to new dataset in DataRegistry
    }

    ImGui::Separator();

    // Render table
    if (ImGui::BeginTable("QueryResults", result_columns_.size(),
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY)) {
        // Header
        for (const auto& col : result_columns_) {
            ImGui::TableSetupColumn(col.c_str());
        }
        ImGui::TableHeadersRow();

        // Rows
        for (const auto& row : result_rows_) {
            ImGui::TableNextRow();
            for (size_t i = 0; i < row.size(); i++) {
                ImGui::TableSetColumnIndex(i);
                ImGui::Text("%s", row[i].c_str());
            }
        }

        ImGui::EndTable();
    }
}
```

---

## Phase 5: Node Editor Handoff (Week 7)

### Day 26-27: DeployToNodeEditor Node

```cpp
// In tabular_nodes.cpp

bool PipelineExecutor::ExecuteDeployToNodeEditor(DataPipelineNode& node) {
    std::string input_dataset = GetInputDatasetName(node);
    if (input_dataset.empty()) {
        node.error_message = "No input dataset";
        return false;
    }

    std::string output_name = node.parameters["name"];
    if (output_name.empty()) {
        output_name = "deployed_" + std::to_string(node.id);
    }

    // Tag dataset with user-friendly name
    // (For now, just store in output_dataset_name_)
    output_dataset_name_ = input_dataset;

    // Signal to DataStudioPanel that deployment is ready
    deployment_ready_ = true;

    node.executed = true;
    return true;
}
```

### Day 28-29: Integration with NodeEditor

```cpp
// In data_studio_panel.cpp

void DataStudioPanel::OnDeployToNodeEditor() {
    auto output_dataset = pipeline_canvas_->GetOutputDatasetName();
    if (!output_dataset.has_value()) {
        // Show error popup
        ImGui::OpenPopup("No Output Dataset");
        return;
    }

    // Get MainWindow (passed via constructor)
    auto* node_editor = main_window_->GetNodeEditor();
    if (!node_editor) {
        return;
    }

    // Handoff dataset to Node Editor
    node_editor->SetDatasetFromDataStudio(output_dataset.value());

    // Show Node Editor
    node_editor->Show();

    // Show success notification
    ImGui::InsertNotification({
        ImGuiToastType_Success,
        3000,
        "Dataset deployed to Node Editor: %s",
        output_dataset.value().c_str()
    });
}
```

```cpp
// In node_editor.cpp (NEW METHOD)

void NodeEditor::SetDatasetFromDataStudio(const std::string& dataset_name) {
    // Find or create DatasetInput node
    MLNode* dataset_input = nullptr;
    for (auto& node : nodes_) {
        if (node.type == NodeType::DatasetInput) {
            dataset_input = &node;
            break;
        }
    }

    if (!dataset_input) {
        // Create new DatasetInput node at center
        AddNodeFromMenu(NodeType::DatasetInput, "Data Input");
        dataset_input = &nodes_.back();
    }

    // Update dataset reference
    UpdateDatasetNodeName(dataset_name);

    // Trigger shape inference
    if (shape_inference_) {
        shape_inference_->InferShapes(nodes_, links_);
    }

    // Save undo state
    SaveUndoState();

    // Frame the DatasetInput node (zoom to it)
    selected_node_ids_ = {dataset_input->id};
    FrameSelected();
}
```

### Day 30: End-to-End Testing

**Test Script:**

```python
# tests/integration/test_e2e_handoff.py

def test_end_to_end_workflow():
    # 1. Load raw data
    ds_panel = main_window.GetDataStudioPanel()
    ds_panel.OnDataImported("test_data/properties_raw.csv", "properties_raw")

    # 2. Build pipeline
    canvas = ds_panel.GetPipelineCanvas()
    canvas.AddNode(DataNodeType.FileInput, "Input")
    canvas.AddNode(DataNodeType.RemoveDuplicates, "Dedup")
    canvas.AddNode(DataNodeType.FillMissing, "Fill")
    canvas.AddNode(DataNodeType.StandardScale, "Scale")
    canvas.AddNode(DataNodeType.DeployToNodeEditor, "Deploy")

    # Connect nodes
    # ... (connect in sequence)

    # 3. Run pipeline
    canvas.RunPipeline()
    while canvas.IsPipelineRunning():
        time.sleep(0.1)

    assert not canvas.HasError()

    # 4. Deploy to Node Editor
    ds_panel.OnDeployToNodeEditor()

    # 5. Verify Node Editor received dataset
    node_editor = main_window.GetNodeEditor()
    data_input_nodes = [n for n in node_editor.GetNodes() if n.type == NodeType.DatasetInput]
    assert len(data_input_nodes) == 1
    assert data_input_nodes[0].parameters["dataset_name"] == "properties_v1"

    # 6. Build simple ML model
    node_editor.AddNode(NodeType.Dense, "Dense 128")
    node_editor.AddNode(NodeType.ReLU, "ReLU")
    node_editor.AddNode(NodeType.Dense, "Dense 1")
    # ... connect nodes

    # 7. Start training
    node_editor.OnTrainClick()

    # Wait for first epoch
    time.sleep(5)

    # 8. Verify training is using correct dataset
    # Check training logs/metrics
    assert training_running
    assert current_epoch > 0
```

---

## Remaining Phases (Weeks 8-11)

**Phase 6 (Weeks 8-9):** Advanced Nodes
- Text processing nodes (TextClean, TextTokenize, etc.)
- Time-series nodes (TSWindow, TSFeatures, etc.)
- Feature engineering nodes (PCA, PolynomialFeatures, etc.)

**Phase 7 (Week 10):** Save/Load & Polish
- Pipeline serialization to JSON
- Load pipeline from file
- UI polish (icons, tooltips, error messages)
- User documentation

**Phase 8 (Week 11):** Performance Optimization
- Lazy evaluation
- Parallel node execution
- Memory optimization
- Streaming support

---

## Daily Checklist Template

**Morning:**
- [ ] Pull latest changes from `main`
- [ ] Review previous day's code
- [ ] Check for new issues/PRs

**Development:**
- [ ] Write failing test first (TDD)
- [ ] Implement feature
- [ ] Run tests (`ctest`)
- [ ] Fix any warnings/errors
- [ ] Code review (self or peer)

**End of Day:**
- [ ] Commit changes with descriptive message
- [ ] Push to feature branch
- [ ] Update progress tracker
- [ ] Document any blockers

---

## Success Metrics

**Phase 1 Complete:**
- [ ] Data Studio panel visible in MainWindow
- [ ] Can add FileInput and SaveDataset nodes
- [ ] Can execute trivial pipeline
- [ ] All Phase 1 tests passing

**Phase 2 Complete:**
- [ ] All 10 tabular nodes implemented
- [ ] Can build real cleaning pipelines
- [ ] Use Case 1 from docs works end-to-end
- [ ] All Phase 2 tests passing

**Phase 3 Complete:**
- [ ] Analysis tab shows stats/correlation
- [ ] Visualization tab shows charts
- [ ] All Phase 3 tests passing

**Phase 4 Complete:**
- [ ] Can execute SQL queries
- [ ] Results display correctly
- [ ] Can save query results as dataset
- [ ] All Phase 4 tests passing

**Phase 5 Complete:**
- [ ] Deploy to Node Editor works
- [ ] Can train model on Data Studio output
- [ ] End-to-end test passes
- [ ] All Phase 5 tests passing

---

**Status:** Ready for Implementation
**Next Step:** Begin Phase 1, Day 1
