# Data Studio — Design Patterns & Edge Case Handling

**Document Version:** 1.0
**Date:** 2026-03-19
**Companion to:** `engine_2.0_architecture.md` and `implementation_roadmap.md`

---

## 1. Critical Design Patterns

### 1.1 Pattern: Separate ImNodes Context (ImGui Context Isolation)

**Problem:**
ImNodes uses global state tied to an `ImNodesEditorContext`. If two editors share the same context, node positions, IDs, and rendering state collide, causing visual corruption and crashes.

**Solution:**
```cpp
class DataStudioPipelineCanvas {
private:
    ImNodesEditorContext* editor_context_;  // SEPARATE from NodeEditor

public:
    DataStudioPipelineCanvas() {
        // Create dedicated context
        editor_context_ = ImNodes::EditorContextCreate();
    }

    ~DataStudioPipelineCanvas() {
        ImNodes::EditorContextFree(editor_context_);
    }

    void Render() {
        // CRITICAL: Set context BEFORE any ImNodes calls
        ImNodes::EditorContextSet(editor_context_);

        ImNodes::BeginNodeEditor();
        RenderNodes();
        ImNodes::EndNodeEditor();

        // CRITICAL: Reset context AFTER rendering
        ImNodes::EditorContextSet(nullptr);
    }
};
```

**Why This Works:**
- Each editor has its own `ImNodesEditorContext`
- Setting/resetting context in `Render()` ensures no leakage
- Even if both editors render in same frame, they don't interfere

**Anti-Pattern (DO NOT DO):**
```cpp
// BAD: Sharing context
ImNodesEditorContext* global_context = ImNodes::EditorContextCreate();

void DataStudioRender() {
    ImNodes::EditorContextSet(global_context);  // Collision!
}

void NodeEditorRender() {
    ImNodes::EditorContextSet(global_context);  // Same context, different state!
}
```

---

### 1.2 Pattern: ID Offset for Collision Avoidance

**Problem:**
Node, pin, and link IDs are integers. If Data Studio and ML Node Editor both start at ID 1, handoff becomes ambiguous.

**Solution:**
```cpp
class DataStudioPipelineCanvas {
    int next_node_id_ = 1000000;  // Start at 1M
    int next_pin_id_ = 1000000;
    int next_link_id_ = 1000000;
};

class NodeEditor {
    int next_node_id_ = 1;  // Start at 1
    int next_pin_id_ = 1;
    int next_link_id_ = 1;
};
```

**Benefits:**
- No ID collision even if both systems active
- Easy to debug: IDs >= 1M → Data Studio, IDs < 1M → ML Node Editor
- Serialization unambiguous (can detect which system a node belongs to)

---

### 1.3 Pattern: Intermediate Dataset Hiding

**Problem:**
Pipeline creates many intermediate datasets (`ds_pipeline_1000001`, `ds_pipeline_1000002`, etc.). User doesn't need to see these in UI.

**Solution:**
```cpp
// In DataRegistry::ListDatasets()
std::vector<DatasetInfo> DataRegistry::ListDatasets() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<DatasetInfo> result;

    for (const auto& [name, dataset] : datasets_) {
        // Hide internal pipeline datasets
        if (name.starts_with("ds_pipeline_")) {
            continue;
        }

        result.push_back(dataset->GetInfo());
    }

    return result;
}
```

**Alternative: Explicit Visibility Flag**
```cpp
struct DatasetInfo {
    bool is_internal = false;  // Hide from UI if true
};
```

---

### 1.4 Pattern: Async Pipeline Execution with Progress Callbacks

**Problem:**
Large datasets (1M+ rows) block UI if executed synchronously.

**Solution:**
```cpp
void DataStudioPipelineCanvas::RunPipeline() {
    if (is_running_) return;

    is_running_ = true;

    // Use AsyncTaskManager for background execution
    auto task_id = AsyncTaskManager::Instance().SubmitTask(
        "Pipeline Execution",
        [this]() {
            return ExecutePipelineAsync();
        },
        [this](bool success) {
            is_running_ = false;
            if (!success) {
                // Show error popup
            }
        },
        [this](float progress, const std::string& status) {
            // Update progress bar in toolbar
            current_progress_ = progress;
            current_status_ = status;
        }
    );
}

bool DataStudioPipelineCanvas::ExecutePipelineAsync() {
    auto sorted_ids = TopologicalSort();
    size_t total_nodes = sorted_ids.size();

    for (size_t i = 0; i < sorted_ids.size(); i++) {
        int node_id = sorted_ids[i];
        auto* node = FindNodeById(node_id);

        // Update progress
        float progress = static_cast<float>(i) / total_nodes;
        std::string status = "Executing: " + node->name;
        AsyncTaskManager::Instance().ReportProgress(
            task_id_, progress, status
        );

        // Execute node
        bool success = ExecuteNode(*node);
        if (!success) {
            return false;
        }
    }

    return true;
}
```

---

### 1.5 Pattern: Dataset Lineage Tracking

**Problem:**
When pipeline fails at node 5, user needs to know which intermediate dataset to inspect.

**Solution:**
```cpp
struct DatasetLineage {
    std::string dataset_name;
    int source_node_id;
    std::string source_node_type;
    std::chrono::system_clock::time_point created_at;
    std::vector<std::string> parent_datasets;
};

class PipelineExecutor {
    std::map<std::string, DatasetLineage> lineage_map_;

    void TrackLineage(int node_id, const std::string& output_dataset,
                      const std::vector<std::string>& input_datasets) {
        lineage_map_[output_dataset] = {
            .dataset_name = output_dataset,
            .source_node_id = node_id,
            .source_node_type = NodeTypeName(FindNode(node_id)->type),
            .created_at = std::chrono::system_clock::now(),
            .parent_datasets = input_datasets
        };
    }
};
```

**UI Display:**
```
Dataset: ds_pipeline_1000003
  Created by: FilterRows (node 1000003)
  Time: 2026-03-19 14:23:45
  Parents:
    - ds_pipeline_1000002 (FillMissing)
    - ds_pipeline_1000001 (RemoveDuplicates)

  [View Dataset] [View Node Config] [Trace Back]
```

---

## 2. Edge Case Handling

### 2.1 Edge Case: Empty Dataset

**Scenario:** User imports empty CSV (0 rows).

**Handling:**
```cpp
bool PipelineExecutor::ExecuteFileInput(DataPipelineNode& node) {
    auto handle = DataRegistry::Instance().LoadDataset(path);

    if (handle.Size() == 0) {
        node.error_message = "Dataset is empty (0 rows)";
        node.has_error = true;
        return false;  // Stop pipeline
    }

    // Continue...
}
```

**User Feedback:**
```
❌ Node "File Input" failed:
   Dataset is empty (0 rows)

   File: properties_raw.csv
   Possible causes:
   - File contains only headers
   - File is corrupted
   - Incorrect delimiter (expected comma, found semicolon?)

   [View File] [Change Delimiter] [Skip Node]
```

---

### 2.2 Edge Case: All Values Null After FillMissing

**Scenario:** Column has 100% null rate, FillMissing can't compute mean/median.

**Handling:**
```cpp
bool PipelineExecutor::ExecuteFillMissing(DataPipelineNode& node) {
    std::string strategy = node.parameters["strategy"];
    std::string column = node.parameters["column"];

    // Check null rate
    float null_rate = ComputeNullRate(input_dataset, column);

    if (null_rate >= 0.99f) {
        // OPTION 1: Warning, drop column
        node.has_error = false;  // Not a failure
        node.warning_message = "Column '" + column + "' is 99% null, dropping";
        // Drop column from output dataset
        return true;

        // OPTION 2: Hard error
        node.error_message = "Column '" + column + "' has no valid values";
        return false;
    }

    // Continue...
}
```

---

### 2.3 Edge Case: Cycle in Pipeline Graph

**Scenario:** User accidentally creates A → B → C → A.

**Detection:**
```cpp
std::vector<int> DataStudioPipelineCanvas::TopologicalSort() {
    std::map<int, int> in_degree;
    std::map<int, std::vector<int>> adj_list;

    // Build graph
    for (const auto& node : nodes_) {
        in_degree[node.id] = 0;
    }

    for (const auto& link : links_) {
        in_degree[link.to_node]++;
        adj_list[link.from_node].push_back(link.to_node);
    }

    // Kahn's algorithm
    std::queue<int> queue;
    for (const auto& [node_id, degree] : in_degree) {
        if (degree == 0) {
            queue.push(node_id);
        }
    }

    std::vector<int> sorted;
    while (!queue.empty()) {
        int node_id = queue.front();
        queue.pop();
        sorted.push_back(node_id);

        for (int neighbor : adj_list[node_id]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                queue.push(neighbor);
            }
        }
    }

    if (sorted.size() != nodes_.size()) {
        // CYCLE DETECTED
        return {};  // Empty = invalid
    }

    return sorted;
}
```

**User Feedback:**
```
❌ Pipeline Validation Failed:
   Cycle detected in graph

   Affected nodes: FileInput → RemoveDuplicates → FillMissing → FileInput

   [Highlight Cycle] [Auto-Fix] [Cancel]
```

---

### 2.4 Edge Case: Memory Limit Exceeded During Execution

**Scenario:** Pipeline creates 10 intermediate datasets, exceeds 4GB limit.

**Handling:**
```cpp
bool PipelineExecutor::ExecuteNode(DataPipelineNode& node) {
    // Check memory before executing
    size_t current_memory = DataRegistry::Instance().GetTotalMemoryUsage();
    size_t memory_limit = DataRegistry::Instance().GetMemoryLimit();

    if (current_memory >= memory_limit * 0.9f) {
        // Approaching limit, trigger eviction
        DataRegistry::Instance().TrimMemory(memory_limit * 0.7f);
    }

    // Execute node
    bool success = /* ... */;

    // Check memory after executing
    current_memory = DataRegistry::Instance().GetTotalMemoryUsage();
    if (current_memory > memory_limit) {
        node.error_message = "Memory limit exceeded: " +
            FormatBytes(current_memory) + " / " + FormatBytes(memory_limit);
        return false;
    }

    return success;
}
```

**User Feedback:**
```
⚠️ Memory Warning:
   Pipeline is using 3.8 GB / 4.0 GB (95%)

   Options:
   - Increase memory limit in Settings
   - Enable streaming mode (slower but uses less memory)
   - Delete unused datasets

   [Increase Limit] [Enable Streaming] [Continue Anyway]
```

---

### 2.5 Edge Case: Column Not Found After Upstream Modification

**Scenario:**
1. User builds pipeline: FileInput → DropColumns(drop="age") → FillMissing(column="age")
2. FillMissing expects "age" column, but it was dropped upstream

**Detection:**
```cpp
bool PipelineExecutor::ExecuteFillMissing(DataPipelineNode& node) {
    std::string column = node.parameters["column"];

    auto input_dataset = GetInputDatasetName(node);
    auto handle = DataRegistry::Instance().GetDataset(input_dataset);
    auto columns = handle.GetUnderlyingDataset()->GetColumnNames();

    if (std::find(columns.begin(), columns.end(), column) == columns.end()) {
        node.error_message = "Column '" + column + "' not found in input dataset";

        // Suggest available columns
        std::string suggestion = "Available columns: ";
        for (size_t i = 0; i < columns.size(); i++) {
            if (i > 0) suggestion += ", ";
            suggestion += columns[i];
        }
        node.error_message += "\n" + suggestion;

        return false;
    }

    // Continue...
}
```

**Prevention (Shape Inference for Data Pipelines):**
```cpp
class DataPipelineValidator {
    void ValidateColumnFlow() {
        // For each node, track which columns exist at that point
        std::map<int, std::set<std::string>> node_columns;

        for (int node_id : TopologicalSort()) {
            auto* node = FindNode(node_id);

            // Get input columns
            std::set<std::string> input_cols = GetInputColumns(node_id);

            // Apply node transformation
            std::set<std::string> output_cols = ApplyColumnTransform(node, input_cols);

            // Store for downstream nodes
            node_columns[node_id] = output_cols;

            // Check if node references non-existent column
            if (node->type == DataNodeType::FillMissing) {
                std::string col = node->parameters["column"];
                if (input_cols.find(col) == input_cols.end()) {
                    // WARNING: Column doesn't exist at this point
                    AddValidationWarning(node_id, "Column '" + col + "' not found");
                }
            }
        }
    }
};
```

---

### 2.6 Edge Case: Dataset Deleted While Pipeline Running

**Scenario:**
1. User starts pipeline using "raw_data"
2. While running, user deletes "raw_data" from DataRegistry

**Protection:**
```cpp
class PipelineExecutor {
    std::set<std::string> referenced_datasets_;

    bool Execute() {
        // Lock all referenced datasets
        for (const auto& dataset_name : referenced_datasets_) {
            DataRegistry::Instance().LockDataset(dataset_name);
        }

        // Execute pipeline
        bool success = /* ... */;

        // Unlock datasets
        for (const auto& dataset_name : referenced_datasets_) {
            DataRegistry::Instance().UnlockDataset(dataset_name);
        }

        return success;
    }
};
```

```cpp
// In data_registry.cpp
void DataRegistry::UnloadDataset(const std::string& name) {
    if (IsDatasetLocked(name)) {
        // Cannot unload, show error
        throw std::runtime_error("Dataset '" + name + "' is in use by Data Studio pipeline");
    }

    // Safe to unload
    datasets_.erase(name);
}
```

---

### 2.7 Edge Case: DuckDB Query Timeout

**Scenario:** User runs query that takes > 60 seconds (GROUP BY on 100M rows).

**Handling:**
```cpp
void DataStudioQueryEditor::ExecuteQuery() {
    std::atomic<bool> query_complete{false};
    std::string error_msg;

    // Run query in separate thread
    std::thread query_thread([&]() {
        try {
            auto result = conn_->Query(query_buffer_);
            // Process result...
            query_complete = true;
        } catch (const std::exception& e) {
            error_msg = e.what();
            query_complete = true;
        }
    });

    // Wait with timeout
    auto start = std::chrono::steady_clock::now();
    constexpr auto TIMEOUT = std::chrono::seconds(60);

    while (!query_complete) {
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed > TIMEOUT) {
            // Timeout, cancel query
            conn_->Interrupt();  // DuckDB supports query interruption
            query_thread.join();

            has_error_ = true;
            error_message_ = "Query timeout (>60s). Consider adding WHERE clause or LIMIT.";
            return;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    query_thread.join();
}
```

---

### 2.8 Edge Case: Node Parameter Missing/Invalid

**Scenario:** User creates FilterRows node but forgets to set "min" parameter.

**Validation:**
```cpp
bool DataStudioPipelineCanvas::ValidatePipeline(std::string& error) {
    for (const auto& node : nodes_) {
        // Check required parameters
        auto required_params = GetRequiredParameters(node.type);

        for (const auto& param : required_params) {
            if (node.parameters.find(param) == node.parameters.end() ||
                node.parameters.at(param).empty()) {
                error = "Node '" + node.name + "' is missing required parameter: " + param;
                return false;
            }
        }

        // Type validation
        if (node.type == DataNodeType::FilterRows) {
            try {
                float min_val = std::stof(node.parameters.at("min"));
                float max_val = std::stof(node.parameters.at("max"));

                if (min_val >= max_val) {
                    error = "Node '" + node.name + "': min must be < max";
                    return false;
                }
            } catch (const std::exception& e) {
                error = "Node '" + node.name + "': Invalid numeric parameter";
                return false;
            }
        }
    }

    return true;
}
```

---

### 2.9 Edge Case: Multiple Output Nodes (Diamond Graph)

**Scenario:**
```
FileInput → RemoveDuplicates ─┬→ SaveDataset("cleaned")
                                └→ ExportFile("export.csv")
```

**Handling:**
```cpp
std::optional<std::string> DataStudioPipelineCanvas::GetOutputDatasetName() const {
    // Find all terminal nodes (no outgoing links)
    std::vector<int> terminal_node_ids;
    for (const auto& node : nodes_) {
        bool has_outgoing = false;
        for (const auto& link : links_) {
            if (link.from_node == node.id) {
                has_outgoing = true;
                break;
            }
        }

        if (!has_outgoing && (node.type == DataNodeType::SaveDataset ||
                             node.type == DataNodeType::DeployToNodeEditor)) {
            terminal_node_ids.push_back(node.id);
        }
    }

    if (terminal_node_ids.empty()) {
        return std::nullopt;  // No output node
    }

    if (terminal_node_ids.size() > 1) {
        // Multiple output nodes, use first DeployToNodeEditor node
        for (int node_id : terminal_node_ids) {
            auto* node = FindNodeById(node_id);
            if (node->type == DataNodeType::DeployToNodeEditor) {
                return node_dataset_map_.at(node_id);
            }
        }

        // No DeployToNodeEditor, use first SaveDataset
        return node_dataset_map_.at(terminal_node_ids[0]);
    }

    return node_dataset_map_.at(terminal_node_ids[0]);
}
```

---

## 3. Performance Optimization Patterns

### 3.1 Pattern: Lazy Evaluation (Only Execute Dirty Nodes)

**Problem:** User modifies parameter in node 3 of 10-node pipeline. Don't re-execute nodes 1-2 (unchanged).

**Solution:**
```cpp
class PipelineExecutor {
    std::set<int> dirty_nodes_;  // Nodes that need re-execution

    void MarkNodeDirty(int node_id) {
        dirty_nodes_.insert(node_id);

        // Mark all downstream nodes dirty
        for (const auto& link : links_) {
            if (link.from_node == node_id) {
                MarkNodeDirty(link.to_node);  // Recursive
            }
        }
    }

    bool Execute() {
        auto sorted_ids = TopologicalSort();

        for (int node_id : sorted_ids) {
            if (dirty_nodes_.find(node_id) == dirty_nodes_.end()) {
                // Node is clean, skip execution
                continue;
            }

            bool success = ExecuteNode(FindNode(node_id));
            if (!success) return false;

            // Mark as clean
            dirty_nodes_.erase(node_id);
        }

        return true;
    }
};
```

---

### 3.2 Pattern: Parallel Execution (Independent Branches)

**Problem:** Pipeline has two independent branches (A→B, C→D), execute sequentially wastes time.

**Solution:**
```cpp
bool PipelineExecutor::Execute() {
    auto sorted_ids = TopologicalSort();

    // Group nodes by execution level (nodes at same level can run in parallel)
    std::vector<std::vector<int>> execution_levels;
    std::map<int, int> node_level;

    for (int node_id : sorted_ids) {
        int level = 0;

        // Find max level of parents
        for (const auto& link : links_) {
            if (link.to_node == node_id) {
                level = std::max(level, node_level[link.from_node] + 1);
            }
        }

        node_level[node_id] = level;

        if (level >= execution_levels.size()) {
            execution_levels.resize(level + 1);
        }
        execution_levels[level].push_back(node_id);
    }

    // Execute level by level
    for (const auto& level_nodes : execution_levels) {
        // Execute all nodes in this level in parallel
        std::vector<std::future<bool>> futures;

        for (int node_id : level_nodes) {
            futures.push_back(std::async(std::launch::async, [this, node_id]() {
                return ExecuteNode(FindNode(node_id));
            }));
        }

        // Wait for all to complete
        for (auto& future : futures) {
            if (!future.get()) {
                return false;  // One node failed
            }
        }
    }

    return true;
}
```

---

### 3.3 Pattern: Streaming for Large Datasets (100M+ Rows)

**Problem:** Loading 100M row CSV into memory fails.

**Solution:**
```cpp
bool PipelineExecutor::ExecuteFileInput(DataPipelineNode& node) {
    std::string path = node.parameters["path"];
    size_t file_size = std::filesystem::file_size(path);

    constexpr size_t STREAMING_THRESHOLD = 1ULL * 1024 * 1024 * 1024;  // 1GB

    if (file_size > STREAMING_THRESHOLD) {
        // Enable streaming mode
        StreamingConfig config;
        config.enabled = true;
        config.buffer_size = 10000;
        config.chunk_size = 1000;

        auto handle = DataRegistry::Instance().LoadStreamingDataset(path, config);
        // ... downstream nodes must support streaming

        return true;
    }

    // Normal load
    auto handle = DataRegistry::Instance().LoadDataset(path);
    return true;
}
```

---

## 4. Testing Strategies

### 4.1 Unit Test Template

```cpp
// tests/data_studio/test_node_execution.cpp

class PipelineExecutorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test dataset
        CreateTestCSV("test_input.csv", {
            {"id", "name", "age", "city"},
            {"1", "Alice", "25", "NYC"},
            {"2", "Bob", "30", "LA"},
            {"1", "Alice", "25", "NYC"},  // Duplicate
            {"3", "Charlie", "", "SF"}    // Missing age
        });

        DataRegistry::Instance().LoadDataset("test_input.csv", "test");
    }

    void TearDown() override {
        DataRegistry::Instance().UnloadAll();
        std::filesystem::remove("test_input.csv");
    }
};

TEST_F(PipelineExecutorTest, RemoveDuplicates_KeepFirst) {
    DataPipelineNode node;
    node.type = DataNodeType::RemoveDuplicates;
    node.parameters["subset"] = "id,name";
    node.parameters["keep"] = "first";

    PipelineExecutor executor;
    bool success = executor.ExecuteRemoveDuplicates(node);

    ASSERT_TRUE(success);
    EXPECT_FALSE(node.has_error);

    auto output = DataRegistry::Instance().GetDataset(node_dataset_map_[node.id]);
    EXPECT_EQ(output.Size(), 3);  // 4 rows → 3 after removing 1 duplicate
}
```

---

### 4.2 Integration Test: End-to-End Pipeline

```cpp
TEST(DataStudioIntegration, FullCleaningPipeline) {
    // Load raw data
    auto raw = DataRegistry::Instance().LoadDataset("properties_raw.csv", "raw");

    // Build pipeline
    DataStudioPipelineCanvas canvas;
    canvas.AddNode(DataNodeType::FileInput, "Input");
    canvas.AddNode(DataNodeType::RemoveDuplicates, "Dedup");
    canvas.AddNode(DataNodeType::FillMissing, "Fill");
    canvas.AddNode(DataNodeType::FilterRows, "Filter");
    canvas.AddNode(DataNodeType::StandardScale, "Scale");
    canvas.AddNode(DataNodeType::SaveDataset, "Output");

    // Connect nodes (simplified)
    // ... connect in sequence

    // Execute
    canvas.RunPipeline();

    // Wait for completion
    while (canvas.IsPipelineRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Verify
    ASSERT_FALSE(canvas.HasError());

    auto output_name = canvas.GetOutputDatasetName();
    ASSERT_TRUE(output_name.has_value());

    auto output = DataRegistry::Instance().GetDataset(output_name.value());
    ASSERT_TRUE(output.IsValid());

    // Check statistics
    EXPECT_LT(output.Size(), raw.Size());  // Rows removed
    // ... more assertions
}
```

---

### 4.3 Performance Benchmark

```cpp
// tests/benchmarks/pipeline_benchmarks.cpp

static void BM_Pipeline_10Nodes_1MRows(benchmark::State& state) {
    auto dataset = CreateLargeDataset(1000000, 50);  // 1M rows, 50 cols

    DataStudioPipelineCanvas canvas;
    // Add 10 nodes: Input → 8 transforms → Output

    for (auto _ : state) {
        canvas.RunPipeline();
        while (canvas.IsPipelineRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        benchmark::DoNotOptimize(canvas);
    }

    state.SetItemsProcessed(state.iterations() * 1000000);
}
BENCHMARK(BM_Pipeline_10Nodes_1MRows);
```

---

## 5. Documentation & User Guides

### 5.1 Tooltip Text (In-App Help)

**For Each Node Type:**

```cpp
const char* GetNodeTooltip(DataNodeType type) {
    switch (type) {
        case DataNodeType::RemoveDuplicates:
            return "Remove duplicate rows from dataset.\n\n"
                   "Parameters:\n"
                   "  subset: Columns to check (comma-separated). Empty = all columns.\n"
                   "  keep: 'first', 'last', or 'none'\n\n"
                   "Example: subset='id,email' keeps first row for each unique (id, email) pair.";

        case DataNodeType::FillMissing:
            return "Impute missing values (nulls) in dataset.\n\n"
                   "Strategies:\n"
                   "  mean: Fill with column mean (numeric only)\n"
                   "  median: Fill with column median (numeric only)\n"
                   "  mode: Fill with most frequent value\n"
                   "  ffill: Forward fill (propagate last valid value)\n"
                   "  constant: Fill with specified value\n\n"
                   "Example: strategy='median', columns='age,income'";

        // ... etc.
    }
}
```

---

### 5.2 Error Message Templates

**Consistent Error Format:**

```
❌ Node "{node_name}" failed:
   {primary_error_message}

   {context_information}

   Suggestions:
   - {suggestion_1}
   - {suggestion_2}

   [View Details] [Skip Node] [Stop Pipeline]
```

**Example:**

```
❌ Node "Fill Missing" failed:
   Column "salary" is 100% null, cannot compute mean

   File: employees.csv
   Column: salary (Float64)
   Null count: 10,000 / 10,000 (100%)

   Suggestions:
   - Change strategy to 'constant' and specify a default value
   - Drop this column using DropColumns node
   - Check data source for errors

   [View Data] [Change Strategy] [Drop Column]
```

---

## 6. Backward Compatibility Checklist

### 6.1 Version 1.0 Project Files

**Test Case:**
1. Create project in current CyxWiz (no Data Studio)
2. Save project as `test_v1.cyxproject`
3. Build CyxWiz 2.0 with Data Studio
4. Open `test_v1.cyxproject`
5. **Expected:** Project loads without errors, Data Studio panel empty

**Implementation:**
```cpp
// In project_manager.cpp
bool ProjectManager::LoadProject(const std::string& filepath) {
    auto j = LoadJSON(filepath);

    std::string version = j.value("version", "1.0");

    if (version == "1.0") {
        // Legacy format, no data_studio_pipeline key
        LoadNodeGraph(j["node_graph"]);
        LoadDatasets(j["datasets"]);
        // Data Studio pipeline will be empty
    } else if (version == "2.0") {
        // New format with Data Studio
        LoadNodeGraph(j["node_graph"]);
        LoadDataStudioPipeline(j["data_studio_pipeline"]);
        LoadDatasets(j["datasets"]);
    }

    return true;
}
```

---

## 7. Conclusion

This document provides **critical implementation patterns** and **edge case handling strategies** for Data Studio. Key takeaways:

1. **Separate ImNodes Context** — Non-negotiable, prevents visual corruption
2. **ID Offset** — Simple collision avoidance strategy
3. **Async Execution** — Essential for large datasets (1M+ rows)
4. **Comprehensive Validation** — Catch errors before execution
5. **User-Friendly Error Messages** — Guide users to solutions
6. **Backward Compatibility** — Preserve existing workflows

**Next Steps:**
- Review this document alongside `engine_2.0_architecture.md`
- Begin implementation following `implementation_roadmap.md`
- Test edge cases as features are completed
- Update this document with new patterns discovered during development

---

**Document Status:** Ready for Implementation
