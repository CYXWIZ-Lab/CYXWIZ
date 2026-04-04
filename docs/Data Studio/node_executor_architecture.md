# Node Executor Architecture Design

**Version**: 1.0
**Date**: 2026-03-23
**Status**: Draft

---

## 1. Problem Statement

Currently we have **duplicate systems**:
- **Tools Menu → Panel**: Full UI with execution logic (e.g., `KMeansPanel`)
- **Node Browser → Node**: Metadata only, NO execution path

Since CyxWiz Studio is now **node-based** (KNIME-style), nodes should be the single source of truth.

---

## 2. Design Goals

| Goal | Description |
|------|-------------|
| **Node-First** | All analytics execute through the node system |
| **Shared Logic** | One executor class per algorithm, used everywhere |
| **Rich Properties** | Properties Panel shows full configuration (like KNIME dialogs) |
| **Pipeline Integration** | Nodes work in DuckDB/Arrow pipelines |
| **Code Generation** | Nodes can generate sklearn/scipy Python code |

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NODE SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐     ┌──────────────────┐     ┌──────────────┐    │
│  │ Node Browser │────>│  Canvas Node     │────>│  Properties  │    │
│  │              │     │  (KMeansCluster) │     │    Panel     │    │
│  └──────────────┘     └────────┬─────────┘     └──────┬───────┘    │
│                                │                       │            │
│                                ▼                       ▼            │
│                     ┌─────────────────────────────────────────┐    │
│                     │         NodeExecutor<KMeans>            │    │
│                     │  ┌─────────────────────────────────┐    │    │
│                     │  │  - Configure(params)            │    │    │
│                     │  │  - Execute(input) -> output     │    │    │
│                     │  │  - GetProgress() -> float       │    │    │
│                     │  │  - GenerateCode(framework)      │    │    │
│                     │  │  - GetVisualization() -> ImGui  │    │    │
│                     │  └─────────────────────────────────┘    │    │
│                     └──────────────────┬──────────────────────┘    │
│                                        │                            │
└────────────────────────────────────────┼────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BACKEND (cyxwiz-backend)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │ clustering.h│  │ statistics.h│  │  signal.h   │                  │
│  │ KMeans      │  │ Correlation │  │  FFT        │                  │
│  │ DBSCAN      │  │ PCA         │  │  Wavelet    │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. NodeExecutor Base Class

```cpp
// src/core/node_executors/node_executor.h

#pragma once
#include <string>
#include <functional>
#include <arrow/table.h>
#include <imgui.h>

namespace cyxwiz {

enum class ExecutorState {
    Idle,
    Configuring,
    Executing,
    Completed,
    Error
};

/**
 * Base class for all node executors
 * Each analytics node type has a corresponding executor
 */
class INodeExecutor {
public:
    virtual ~INodeExecutor() = default;

    // === Configuration ===
    virtual void Configure(const NodeProperties& props) = 0;
    virtual NodeProperties GetDefaultProperties() const = 0;
    virtual bool ValidateConfiguration(std::string& error) const = 0;

    // === Execution ===
    virtual void Execute(const std::vector<std::shared_ptr<arrow::Table>>& inputs) = 0;
    virtual void Cancel() = 0;
    virtual ExecutorState GetState() const = 0;
    virtual float GetProgress() const = 0;
    virtual std::string GetStatusMessage() const = 0;

    // === Results ===
    virtual std::vector<std::shared_ptr<arrow::Table>> GetOutputs() const = 0;
    virtual bool HasVisualization() const = 0;

    // === UI Integration ===
    // Renders configuration UI in Properties Panel
    virtual void RenderConfigUI() = 0;
    // Renders results/visualization (optional, for complex nodes)
    virtual void RenderResultsUI() {}

    // === Code Generation ===
    virtual std::string GenerateCode(CodeFramework framework) const = 0;
};

/**
 * Template base for typed executors
 */
template<typename TConfig, typename TResult>
class NodeExecutor : public INodeExecutor {
protected:
    TConfig config_;
    TResult result_;
    ExecutorState state_ = ExecutorState::Idle;
    float progress_ = 0.0f;
    std::string status_message_;
    std::atomic<bool> cancel_requested_{false};

public:
    const TConfig& GetConfig() const { return config_; }
    const TResult& GetResult() const { return result_; }

    ExecutorState GetState() const override { return state_; }
    float GetProgress() const override { return progress_; }
    std::string GetStatusMessage() const override { return status_message_; }

    void Cancel() override { cancel_requested_ = true; }
};

} // namespace cyxwiz
```

---

## 5. Example: KMeans Executor

```cpp
// src/core/node_executors/kmeans_executor.h

#pragma once
#include "node_executor.h"
#include <cyxwiz/clustering.h>

namespace cyxwiz {

struct KMeansConfig {
    int n_clusters = 3;
    int max_iter = 300;
    int n_init = 10;
    int init_method = 1;  // 0=random, 1=kmeans++
    double tolerance = 1e-4;
    std::vector<int> feature_columns;  // Which columns to use
};

struct KMeansOutput {
    KMeansResult result;              // From backend
    std::vector<int> labels;          // Cluster assignments
    std::vector<std::vector<double>> centroids;
    double inertia;
    int n_iterations;
    ElbowAnalysis elbow;              // Optional elbow analysis
};

class KMeansExecutor : public NodeExecutor<KMeansConfig, KMeansOutput> {
public:
    // === Configuration ===
    void Configure(const NodeProperties& props) override {
        config_.n_clusters = props.GetInt("n_clusters", 3);
        config_.max_iter = props.GetInt("max_iter", 300);
        config_.init_method = props.GetEnum("init", {"random", "k-means++"}, 1);
        // ... etc
    }

    NodeProperties GetDefaultProperties() const override {
        NodeProperties props;
        props.Set("n_clusters", 3, "Number of clusters", 2, 100);
        props.Set("max_iter", 300, "Maximum iterations", 1, 1000);
        props.Set("init", "k-means++", "Initialization method", {"random", "k-means++"});
        return props;
    }

    bool ValidateConfiguration(std::string& error) const override {
        if (config_.n_clusters < 2) {
            error = "Need at least 2 clusters";
            return false;
        }
        return true;
    }

    // === Execution ===
    void Execute(const std::vector<std::shared_ptr<arrow::Table>>& inputs) override {
        state_ = ExecutorState::Executing;
        progress_ = 0.0f;

        // Extract data from Arrow table
        auto data = ExtractFeatureMatrix(inputs[0], config_.feature_columns);

        // Use backend KMeans (same as KMeansPanel uses!)
        cyxwiz::KMeans kmeans(config_.n_clusters);
        kmeans.SetMaxIterations(config_.max_iter);
        kmeans.SetInitMethod(config_.init_method == 0 ? InitMethod::Random : InitMethod::KMeansPP);

        // Run with progress callback
        result_.result = kmeans.Fit(data, [this](int iter, double inertia) {
            progress_ = float(iter) / config_.max_iter;
            status_message_ = fmt::format("Iteration {}, inertia: {:.2f}", iter, inertia);
            return !cancel_requested_;
        });

        result_.labels = result_.result.labels;
        result_.centroids = result_.result.centroids;
        result_.inertia = result_.result.inertia;

        state_ = ExecutorState::Completed;
        progress_ = 1.0f;
    }

    std::vector<std::shared_ptr<arrow::Table>> GetOutputs() const override {
        // Output 1: Original data with cluster labels column
        // Output 2: Centroids table
        return { CreateLabeledTable(), CreateCentroidsTable() };
    }

    bool HasVisualization() const override { return true; }

    // === UI Integration ===
    void RenderConfigUI() override {
        ImGui::SliderInt("Clusters (k)", &config_.n_clusters, 2, 20);
        ImGui::SliderInt("Max Iterations", &config_.max_iter, 10, 1000);

        const char* init_methods[] = { "Random", "K-Means++" };
        ImGui::Combo("Initialization", &config_.init_method, init_methods, 2);

        if (ImGui::Button("Run Elbow Analysis")) {
            RunElbowAnalysis();
        }

        // Show elbow plot if available
        if (result_.elbow.scores.size() > 0) {
            RenderElbowPlot();
        }
    }

    void RenderResultsUI() override {
        // Scatter plot with cluster colors
        if (state_ == ExecutorState::Completed) {
            RenderScatterPlot();
            RenderCentroidsTable();
        }
    }

    // === Code Generation ===
    std::string GenerateCode(CodeFramework framework) const override {
        switch (framework) {
            case CodeFramework::PyTorch:
            case CodeFramework::PyCyxWiz:
                return GeneratePyCyxWizCode();
            default:
                return GenerateSklearnCode();
        }
    }

private:
    std::string GenerateSklearnCode() const {
        return fmt::format(R"(
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters={},
    max_iter={},
    init='{}',
    n_init={}
)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_
)",
            config_.n_clusters,
            config_.max_iter,
            config_.init_method == 0 ? "random" : "k-means++",
            config_.n_init
        );
    }

    std::string GeneratePyCyxWizCode() const {
        return fmt::format(R"(
import pycyxwiz as cyx

kmeans = cyx.KMeans(n_clusters={})
result = kmeans.fit(X)
labels = result.labels
centroids = result.centroids
)",
            config_.n_clusters
        );
    }

    void RenderElbowPlot() {
        // ImPlot elbow chart
    }

    void RenderScatterPlot() {
        // ImPlot scatter with cluster colors
    }
};

} // namespace cyxwiz
```

---

## 6. Integration with Node Editor

```cpp
// In node_editor.cpp - when executing a node

void NodeEditor::ExecuteNode(MLNode& node) {
    // Get or create executor for this node type
    auto executor = GetOrCreateExecutor(node.type);

    // Configure from node properties
    executor->Configure(node.properties);

    // Validate
    std::string error;
    if (!executor->ValidateConfiguration(error)) {
        node.execution_state = NodeExecutionState::Error;
        node.error_message = error;
        return;
    }

    // Get input data from connected nodes
    auto inputs = CollectInputs(node);

    // Execute (async)
    executor->Execute(inputs);

    // Store outputs for downstream nodes
    node.outputs = executor->GetOutputs();
    node.execution_state = NodeExecutionState::Completed;
}

INodeExecutor* NodeEditor::GetOrCreateExecutor(NodeType type) {
    if (executors_.find(type) == executors_.end()) {
        executors_[type] = CreateExecutor(type);
    }
    return executors_[type].get();
}

std::unique_ptr<INodeExecutor> NodeEditor::CreateExecutor(NodeType type) {
    switch (type) {
        case NodeType::KMeansCluster:
            return std::make_unique<KMeansExecutor>();
        case NodeType::DBSCANCluster:
            return std::make_unique<DBSCANExecutor>();
        case NodeType::PCANode:
            return std::make_unique<PCAExecutor>();
        case NodeType::CorrelationMatrix:
            return std::make_unique<CorrelationExecutor>();
        // ... etc
        default:
            return nullptr;
    }
}
```

---

## 7. Properties Panel Integration

```cpp
// In properties.cpp

void PropertiesPanel::RenderNodeProperties(MLNode& node) {
    auto executor = node_editor_->GetOrCreateExecutor(node.type);
    if (!executor) {
        ImGui::TextDisabled("No configuration available");
        return;
    }

    // Let executor render its own config UI
    executor->RenderConfigUI();

    // Render results if node has executed
    if (node.execution_state == NodeExecutionState::Completed) {
        if (executor->HasVisualization()) {
            if (ImGui::CollapsingHeader("Results", ImGuiTreeNodeFlags_DefaultOpen)) {
                executor->RenderResultsUI();
            }
        }
    }
}
```

---

## 8. What Happens to Existing Panels?

### Option A: Deprecate Panels (Recommended)

1. Move all logic from `KMeansPanel` to `KMeansExecutor`
2. Remove `KMeansPanel` class
3. Remove Tools menu callbacks
4. Users use Node Browser → drag K-Means node → configure in Properties

### Option B: Panels as "Detached Views"

1. Keep panels but they just display executor results
2. Panel becomes a floating window showing `executor->RenderResultsUI()`
3. Double-click node → opens detached panel view

### Option C: Gradual Migration

1. Phase 1: Create executors, panels still work
2. Phase 2: Add "Also available as node" badge to Tools menu
3. Phase 3: Deprecation warning on panels
4. Phase 4: Remove panels

---

## 9. File Structure

```
cyxwiz-engine/src/core/
├── node_executors/
│   ├── node_executor.h              # Base class + interface
│   ├── node_executor_factory.cpp    # Factory for creating executors
│   │
│   ├── analytics/
│   │   ├── kmeans_executor.h/cpp
│   │   ├── dbscan_executor.h/cpp
│   │   ├── pca_executor.h/cpp
│   │   ├── correlation_executor.h/cpp
│   │   └── statistics_executor.h/cpp
│   │
│   ├── preprocessing/
│   │   ├── normalizer_executor.h/cpp
│   │   ├── scaler_executor.h/cpp
│   │   └── encoder_executor.h/cpp
│   │
│   ├── signal/
│   │   ├── fft_executor.h/cpp
│   │   ├── filter_executor.h/cpp
│   │   └── wavelet_executor.h/cpp
│   │
│   └── text/
│       ├── tokenizer_executor.h/cpp
│       ├── tfidf_executor.h/cpp
│       └── sentiment_executor.h/cpp
```

---

## 10. Migration Priority

| Priority | Nodes | Reason |
|----------|-------|--------|
| **P1** | KMeans, PCA, Correlation, DataProfiler | Most used analytics |
| **P1** | StandardScaler, MinMaxScaler | Core preprocessing |
| **P2** | DBSCAN, Hierarchical, t-SNE | Clustering & viz |
| **P2** | CrossValidation, ConfusionMatrix | Model evaluation |
| **P3** | FFT, Spectrogram, Filter | Signal processing |
| **P3** | Tokenizer, TF-IDF, Sentiment | Text processing |
| **P4** | Calculator, Regex, UnitConverter | Utilities |

---

## 11. Code Generation for Analytics Nodes

Add to `node_editor_codegen.cpp`:

```cpp
case NodeType::KMeansCluster: {
    auto executor = GetOrCreateExecutor(NodeType::KMeansCluster);
    code += executor->GenerateCode(framework);
    break;
}

case NodeType::PCANode: {
    auto executor = GetOrCreateExecutor(NodeType::PCANode);
    code += executor->GenerateCode(framework);
    break;
}
// ... etc
```

---

## 12. Benefits

1. **Single Source of Truth**: One executor class per algorithm
2. **No Duplicate Code**: Panel and Node use same executor
3. **Better Testing**: Executor can be unit tested independently
4. **Code Generation**: Executor knows how to generate sklearn/scipy code
5. **Rich UI**: Executor provides both config UI and results visualization
6. **Pipeline Integration**: Executor outputs Arrow tables for chaining

---

## 13. Next Steps

1. [ ] Create `INodeExecutor` base class
2. [ ] Implement `KMeansExecutor` as proof of concept
3. [ ] Integrate with Properties Panel
4. [ ] Add code generation for KMeans node
5. [ ] Migrate remaining analytics nodes
6. [ ] Deprecate separate panels
