# CyxWiz Studio Major Upgrade - Design Document

**Version**: 1.0
**Date**: 2026-03-22
**Status**: Draft - Awaiting Approval

---

## 1. Executive Summary

Transform CyxWiz Studio into a KNIME-style visual analytics platform where **everything is a node**. This upgrade introduces:

- **Node Browser Panel**: Categorized, searchable node library
- **Info Panel**: Context-sensitive help and documentation
- **Unified Canvas**: Data pipelines + ML models in one workspace
- **Tool-to-Node Migration**: Convert 70+ menu tools to draggable nodes

---

## 2. Goals

| Goal | Description |
|------|-------------|
| **KNIME Parity** | Match KNIME's node-based workflow UX |
| **Discoverability** | Users find tools via browsing, not memorizing menus |
| **Unified Workflow** | Data prep → ML training in one canvas |
| **Extensibility** | Easy to add new nodes (plugins, templates) |
| **Documentation** | In-app help for every node |

---

## 3. Current State Analysis

### 3.1 Existing Node Categories (24)

```cpp
enum class NodeCategory {
    DataSources,      // CSV, SQL, HDF5, API
    DataTransform,    // Filter, Join, GroupBy
    Analytics,        // Stats, Visualize, Correlation
    Preprocessing,    // Normalize, Scale, Encode
    Layers,           // Dense, Conv2D, LSTM
    Activation,       // ReLU, Sigmoid, Softmax
    Pooling,          // MaxPool, AvgPool
    Normalization,    // BatchNorm, LayerNorm
    Attention,        // MultiHeadAttention, Transformer
    Recurrent,        // RNN, LSTM, GRU
    ShapeOps,         // Reshape, Permute, Squeeze
    MergeOps,         // Concatenate, Add, Multiply
    Training,         // Optimizer, Loss, LR Scheduler
    Regularization,   // L1, L2, Dropout
    Utility,          // Lambda, Identity, Constant
    Signal,           // Sliders, Sine, Scope
    DataPipeline,     // DatasetInput, DataLoader
    DNN,              // Pre-trained models
    TextProcessing,   // Tokenizer, Vocabulary
    Upsampling,       // ConvTranspose, PixelShuffle
    TimeSeries,       // Window, Features, Split
    Audio,            // Spectrogram, MFCC
    RL,               // Gym, ReplayBuffer, Policy
    DataExport,       // Export CSV, Parquet, SQL
    Plugin            // Plugin-defined nodes
};
```

### 3.2 Existing Node Types (77+)

See `node_editor.h` lines 63-278 for complete list.

### 3.3 Tools Menu Components (70+)

Currently in `toolbar_tools_menu.cpp` - all are dialog/panel-based:

| Category | Tools Count | Examples |
|----------|-------------|----------|
| Advanced | 2 | Hyperparameter Search, Model Serving |
| Model Export | 3 | Save Model, Binary↔Directory |
| Testing | 5 | Run Test, View Results, Compare, Export |
| Model Analysis | 4 | Model Summary, FLOPs, Architecture Diagram, LR Finder |
| Data Science | 8 | Data Profiler, Missing Value, Outlier Detection, Correlation, Normalization, Standardization, Log Transform, Box-Cox |
| Statistics | 4 | Descriptive Stats, Hypothesis Testing, Regression, Distribution Fitter |
| Machine Learning | 10 | Cross-Validation, Confusion Matrix, ROC/AUC, PR Curve, Learning Curves, K-Means, DBSCAN, Hierarchical, GMM, Feature Importance |
| Deep Learning | 3 | Grad-CAM, Saliency Maps, NAS |
| Linear Algebra | 5 | Matrix Calculator, Eigen, SVD, QR, Cholesky |
| Signal Processing | 5 | FFT, Spectrogram, Filter Designer, Convolution, Wavelet |
| Optimization | 6 | Gradient Descent, Convexity, LP, QP, Differentiation, Integration |
| Time Series | 5 | Decomposition, ACF/PACF, Stationarity, Seasonality, Forecasting |
| Text Processing | 5 | Tokenization, Word Frequency, TF-IDF, Embeddings, Sentiment |
| Utilities | 6 | Calculator, Unit Converter, Random Generator, Hash, JSON Viewer, Regex |

---

## 4. New UI Layout

### 4.1 Three-Panel Layout

```
+--------------------+--------------------------------+------------------+
|    NODE BROWSER    |        CYXWIZ STUDIO          |   PROPERTIES     |
|    (Left Panel)    |         (Canvas)              |   (Right Top)    |
+--------------------+                                +------------------+
| [Search...]        |                                |   NODE INFO      |
+--------------------+    +-------+     +-------+     |   (Right Bottom) |
| > I/O              |    | CSV   |---->| View  |     +------------------+
|   CSV Reader       |    +-------+     +-------+     | Description:     |
|   Excel Reader     |         |                      | Reads CSV files  |
|   Parquet Reader   |         v                      | into Arrow table |
|   JSON Reader      |    +-------+     +-------+     |                  |
|   SQL Query        |    | Dense |---->|  MSE  |     | Inputs:          |
|   REST API         |    +-------+     +-------+     | - File path      |
|   [Show All...]    |         |                      | - Delimiter      |
+--------------------+         v                      | - Has header     |
| > Manipulation     |    +-------+                   |                  |
|   Row Filter       |    |  SGD  |                   | Outputs:         |
|   Column Filter    |    +-------+                   | - Table (Arrow)  |
|   Joiner           |                                |                  |
|   GroupBy          |                                | [View Docs]      |
|   Sorter           |                                +------------------+
|   Missing Value    |                                |
|   [Show All...]    |                                |
+--------------------+                                |
| > Views            |                                |
| > Analytics        |                                |
| > ML Layers        |                                |
| > Training         |                                |
| > Workflow         |                                |
+--------------------+--------------------------------+
```

### 4.2 Panel Specifications

#### Node Browser Panel (Left)

| Feature | Description |
|---------|-------------|
| **Width** | 220px default, resizable 180-300px |
| **Search** | Filter nodes by name, fuzzy matching |
| **Categories** | Collapsible sections with icons |
| **Nodes per Category** | Show 6 most-used, "Show All" for rest |
| **Drag & Drop** | Drag node to canvas to create instance |
| **Favorites** | Star frequently-used nodes (persist) |
| **Recent** | Show last 10 used nodes at top |

#### Properties Panel (Right Top)

| Feature | Description |
|---------|-------------|
| **Width** | 280px default, resizable 220-400px |
| **Context** | Updates when node selected |
| **Sections** | General, Parameters, Advanced |
| **Validation** | Real-time parameter validation |
| **Presets** | Save/load parameter presets |

#### Info Panel (Right Bottom)

| Feature | Description |
|---------|-------------|
| **Height** | 200px default, resizable 150-350px |
| **Content** | Description, inputs, outputs, examples |
| **Links** | "View Full Docs" opens external docs |
| **Examples** | Mini workflow screenshots |

---

## 5. Node Categorization (KNIME-Aligned)

### 5.1 New Category Structure

| Category | Icon | Description | Node Count |
|----------|------|-------------|------------|
| **I/O** | `ICON_FA_FILE_IMPORT` | Read/write data files | 15 |
| **Manipulation** | `ICON_FA_WAND_MAGIC_SPARKLES` | Transform, filter, join | 20 |
| **Views** | `ICON_FA_CHART_LINE` | Visualizations, charts | 15 |
| **DB** | `ICON_FA_DATABASE` | Database connectors | 12 |
| **Analytics** | `ICON_FA_MAGNIFYING_GLASS_CHART` | Statistics, ML algorithms | 25 |
| **ML Layers** | `ICON_FA_LAYER_GROUP` | Neural network layers | 30 |
| **Training** | `ICON_FA_GRADUATION_CAP` | Loss, optimizer, metrics | 15 |
| **Workflow** | `ICON_FA_SITEMAP` | Loops, conditionals, variables | 10 |
| **Utilities** | `ICON_FA_TOOLBOX` | Calculator, converter, regex | 10 |

### 5.2 Category Details

#### I/O (Input/Output)

**Top 6 (Always Visible):**
1. CSV Reader
2. Excel Reader
3. Parquet Reader
4. JSON Reader
5. SQL Query
6. REST API

**Show All:**
- HDF5 Reader, ARFF Reader, Table Creator, File Reader
- CSV Writer, Excel Writer, Parquet Writer, JSON Writer
- Image Reader, Audio Reader, Video Reader

#### Manipulation (Data Transform)

**Top 6:**
1. Row Filter
2. Column Filter
3. Joiner
4. GroupBy
5. Sorter
6. Missing Value

**Show All:**
- Concatenate, Pivot, Unpivot, String Manipulation
- Rule Engine, Math Formula, Column Rename
- Duplicate Filter, Row Sampler, Normalizer, One-Hot Encoder

#### Views (Visualization)

**Top 6:**
1. Bar Chart
2. Line Plot
3. Scatter Plot
4. Histogram
5. Table View
6. Heatmap

**Show All:**
- Pie Chart, Box Plot, Density Plot, ROC Curve
- Confusion Matrix View, Learning Curves, PR Curve
- 3D Scatter, Parallel Coordinates, Tree View

#### Analytics (Statistics & ML)

**Top 6:**
1. Descriptive Statistics
2. Correlation Matrix
3. PCA
4. K-Means
5. Decision Tree Learner
6. Random Forest Learner

**Show All:**
- Hypothesis Testing, Regression, Distribution Fitter
- DBSCAN, Hierarchical Clustering, GMM
- SVM, KNN, Naive Bayes, Logistic Regression
- Cross-Validation, Feature Importance, t-SNE, UMAP

#### ML Layers (Neural Networks)

**Top 6:**
1. Dense
2. Conv2D
3. LSTM
4. Dropout
5. BatchNorm
6. Attention

**Show All:**
- Conv1D, Conv3D, ConvTranspose, Pooling (Max, Avg, Global)
- GRU, RNN, Embedding
- LayerNorm, GroupNorm, InstanceNorm
- Transformer, MultiHeadAttention
- Reshape, Flatten, Permute, Concatenate, Add

#### Training (Optimization)

**Top 6:**
1. MSE Loss
2. CrossEntropy Loss
3. Adam Optimizer
4. SGD Optimizer
5. LR Scheduler
6. Metrics

**Show All:**
- MAE, Huber, Focal, Triplet, Contrastive Loss
- AdamW, RMSprop, Momentum
- StepLR, CosineAnnealing, ReduceOnPlateau
- Accuracy, F1, Precision, Recall

#### Workflow (Control Flow)

**Top 6:**
1. Loop Start
2. Loop End
3. IF Switch
4. Variable Creator
5. Table Row to Variable
6. Breakpoint

**Show All:**
- Counting Loop, Group Loop, Recursive Loop
- CASE Switch, Try/Catch, Wait
- Variable to Table, Merge Variables

#### Utilities (Tools)

**Top 6:**
1. Calculator
2. Unit Converter
3. Random Generator
4. Hash Generator
5. JSON Viewer
6. Regex Tester

**Show All:**
- Color Picker, Date/Time Formatter
- Base64 Encoder/Decoder, URL Encoder

---

## 6. Node Metadata Structure

### 6.1 Node Definition Schema

```cpp
struct NodeMetadata {
    // Identity
    NodeType type;
    NodeCategory category;
    std::string name;           // "CSV Reader"
    std::string icon;           // ICON_FA_FILE_CSV

    // Discovery
    std::vector<std::string> keywords;  // {"csv", "file", "read", "import"}
    int usage_count;            // For "most used" sorting
    bool is_favorite;           // User-starred

    // Documentation
    std::string description;    // Brief description
    std::string help_text;      // Detailed help
    std::string example_usage;  // Code/workflow example

    // Ports
    std::vector<PortDefinition> inputs;
    std::vector<PortDefinition> outputs;

    // Parameters
    std::vector<ParameterDefinition> parameters;

    // State
    NodeImplementationStatus status;  // Implemented, Template, Deprecated
};

enum class NodeImplementationStatus {
    Implemented,    // Fully working
    Template,       // Defined but not implemented (future)
    Deprecated,     // Being phased out
    External        // Requires external integration (KNIME)
};

struct PortDefinition {
    std::string name;
    PinType type;
    bool required;
    std::string description;
};

struct ParameterDefinition {
    std::string name;
    std::string type;           // "string", "int", "float", "enum", "bool", "file"
    std::string default_value;
    std::string description;
    std::vector<std::string> enum_values;  // For enum type
    std::string validation;     // Regex or range "0-100"
};
```

### 6.2 Example Node Definition

```cpp
NodeMetadata csv_reader = {
    .type = NodeType::CSVFile,
    .category = NodeCategory::DataSources,
    .name = "CSV Reader",
    .icon = ICON_FA_FILE_CSV,
    .keywords = {"csv", "file", "read", "import", "comma", "separated"},
    .usage_count = 0,
    .is_favorite = false,
    .description = "Reads a CSV file into an Arrow table",
    .help_text = R"(
        The CSV Reader node loads comma-separated value files into memory
        as Arrow tables for efficient processing.

        Supports:
        - Custom delimiters (comma, tab, semicolon, pipe)
        - Header row detection
        - Column type inference
        - Missing value handling
        - Large file streaming
    )",
    .example_usage = "Connect to Row Filter or GroupBy for data processing",
    .inputs = {},
    .outputs = {
        {"Table", PinType::Dataset, true, "The loaded data as Arrow table"}
    },
    .parameters = {
        {"file_path", "file", "", "Path to CSV file", {}, "*.csv"},
        {"delimiter", "enum", ",", "Column separator", {",", "\\t", ";", "|"}, ""},
        {"has_header", "bool", "true", "First row contains column names", {}, ""},
        {"skip_rows", "int", "0", "Number of rows to skip at start", {}, "0-1000"},
        {"encoding", "enum", "utf-8", "File encoding", {"utf-8", "latin-1", "ascii"}, ""}
    },
    .status = NodeImplementationStatus::Implemented
};
```

---

## 7. Implementation Phases

### Phase 1: Node Browser Panel (Week 1-2)

**Files to Create:**
- `src/gui/panels/node_browser_panel.h`
- `src/gui/panels/node_browser_panel.cpp`

**Tasks:**
1. Create `NodeBrowserPanel` class
2. Implement collapsible category sections
3. Implement search with fuzzy matching
4. Implement drag-and-drop to canvas
5. Add "Show All" expansion
6. Persist favorites and recent nodes
7. Wire to MainWindow

**Dependencies:** None (uses existing node types)

### Phase 2: Info Panel (Week 2-3)

**Files to Create:**
- `src/gui/panels/node_info_panel.h`
- `src/gui/panels/node_info_panel.cpp`
- `src/core/node_metadata_registry.h`
- `src/core/node_metadata_registry.cpp`

**Tasks:**
1. Create `NodeMetadata` struct
2. Create `NodeMetadataRegistry` singleton
3. Populate metadata for all existing nodes
4. Create `NodeInfoPanel` UI
5. Connect to node selection events
6. Add "View Full Docs" link

**Dependencies:** Phase 1

### Phase 3: Properties Panel Enhancement (Week 3-4)

**Files to Modify:**
- `src/gui/properties.h`
- `src/gui/properties.cpp`

**Tasks:**
1. Add parameter sections (General, Parameters, Advanced)
2. Implement validation with error display
3. Add parameter presets save/load
4. Support all parameter types (string, int, float, enum, bool, file)
5. Real-time canvas update on parameter change

**Dependencies:** Phase 2

### Phase 4: Tool-to-Node Migration (Week 4-8)

**Priority Order:**

| Priority | Nodes | Reason |
|----------|-------|--------|
| P1 | Data Profiler, Missing Value, Correlation Matrix | Core data exploration |
| P1 | K-Means, PCA, Decision Tree | Core ML |
| P2 | ROC Curve, Confusion Matrix, Learning Curves | Model evaluation |
| P2 | Normalization, Standardization | Data preprocessing |
| P3 | FFT, Spectrogram, Filter Designer | Signal processing |
| P3 | Tokenization, TF-IDF, Sentiment | Text processing |
| P4 | Calculator, Unit Converter, Regex | Utilities |

**Migration Strategy:**
1. Create node version alongside existing panel
2. Share computation logic (refactor to common class)
3. Mark menu item as "Also available as node"
4. After testing, deprecate menu item
5. Remove menu item in future release

### Phase 5: Templates for Future Nodes (Week 8-9)

**Files to Create:**
- `resources/node_templates/io_nodes.json`
- `resources/node_templates/db_nodes.json`
- `resources/node_templates/analytics_nodes.json`

**Template Format:**
```json
{
  "nodes": [
    {
      "type": "PostgreSQLConnector",
      "category": "DB",
      "name": "PostgreSQL Connector",
      "icon": "database",
      "status": "template",
      "description": "Connect to PostgreSQL database",
      "inputs": [],
      "outputs": [
        {"name": "Connection", "type": "DBConnection"}
      ],
      "parameters": [
        {"name": "host", "type": "string", "default": "localhost"},
        {"name": "port", "type": "int", "default": "5432"},
        {"name": "database", "type": "string", "default": ""},
        {"name": "username", "type": "string", "default": ""},
        {"name": "password", "type": "password", "default": ""}
      ],
      "implementation_notes": "Use libpq or KNIME open-source connector"
    }
  ]
}
```

**Template Behavior:**
- Appear in Node Browser with "Coming Soon" badge
- Cannot be dragged to canvas
- Show planned functionality in Info Panel
- Track user interest (click count)

### Phase 6: Polish & Documentation (Week 9-10)

**Tasks:**
1. Keyboard shortcuts (Ctrl+N = new node, / = search)
2. Context menu on node (Copy, Paste, Delete, Help)
3. Tooltips on all UI elements
4. User guide in docs/
5. Video tutorial storyboard

---

## 8. Technical Specifications

### 8.1 File Structure

```
cyxwiz-engine/src/
├── gui/
│   ├── panels/
│   │   ├── node_browser_panel.h/cpp      [NEW]
│   │   ├── node_info_panel.h/cpp         [NEW]
│   │   └── properties.h/cpp              [MODIFY]
│   └── node_editor.h/cpp                 [MODIFY]
├── core/
│   ├── node_metadata_registry.h/cpp      [NEW]
│   └── node_metadata.h                   [NEW]
└── resources/
    └── node_templates/                   [NEW]
        ├── io_nodes.json
        ├── db_nodes.json
        └── analytics_nodes.json
```

### 8.2 Class Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        MainWindow                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │NodeBrowser   │  │  NodeEditor  │  │    PropertiesPanel   │   │
│  │   Panel      │  │   (Canvas)   │  │    + NodeInfoPanel   │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
│         │                 │                      │               │
│         │   OnNodeDrag()  │                      │               │
│         ├────────────────>│                      │               │
│         │                 │  OnNodeSelected()    │               │
│         │                 ├─────────────────────>│               │
│         │                 │                      │               │
└─────────┴─────────────────┴──────────────────────┴───────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  NodeMetadataRegistry   │
              │  (Singleton)            │
              │  - GetMetadata(type)    │
              │  - GetByCategory(cat)   │
              │  - Search(query)        │
              │  - GetMostUsed(n)       │
              │  - GetRecent(n)         │
              └─────────────────────────┘
```

### 8.3 Key Interfaces

```cpp
// Node Browser Panel
class NodeBrowserPanel {
public:
    void Render();
    void SetOnNodeDrag(std::function<void(NodeType)> callback);
    void SetSearchQuery(const std::string& query);
    void ToggleFavorite(NodeType type);

private:
    void RenderSearchBar();
    void RenderCategory(NodeCategory category);
    void RenderNodeItem(const NodeMetadata& node);
    bool BeginDragNode(NodeType type);

    std::string search_query_;
    std::set<NodeCategory> expanded_categories_;
    std::function<void(NodeType)> on_node_drag_;
};

// Node Info Panel
class NodeInfoPanel {
public:
    void Render();
    void SetSelectedNode(NodeType type);
    void ClearSelection();

private:
    void RenderDescription();
    void RenderPorts();
    void RenderParameters();
    void RenderExamples();

    NodeType selected_type_;
    const NodeMetadata* metadata_;
};

// Node Metadata Registry
class NodeMetadataRegistry {
public:
    static NodeMetadataRegistry& Instance();

    const NodeMetadata* GetMetadata(NodeType type) const;
    std::vector<NodeMetadata> GetByCategory(NodeCategory category) const;
    std::vector<NodeMetadata> Search(const std::string& query) const;
    std::vector<NodeMetadata> GetMostUsed(size_t count) const;
    std::vector<NodeMetadata> GetRecent(size_t count) const;
    std::vector<NodeMetadata> GetFavorites() const;

    void IncrementUsage(NodeType type);
    void AddToRecent(NodeType type);
    void ToggleFavorite(NodeType type);

    void LoadTemplates(const std::string& path);
    void SaveUserPreferences();

private:
    std::unordered_map<NodeType, NodeMetadata> metadata_;
    std::vector<NodeType> recent_nodes_;
    std::set<NodeType> favorites_;
};
```

### 8.4 Drag & Drop Implementation

```cpp
// In NodeBrowserPanel::RenderNodeItem()
if (ImGui::Selectable(node.name.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick)) {
    // Double-click adds to center of canvas
    if (ImGui::IsMouseDoubleClicked(0)) {
        on_node_drag_(node.type);
    }
}

// Begin drag source
if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
    ImGui::SetDragDropPayload("NODE_TYPE", &node.type, sizeof(NodeType));
    ImGui::Text("+ %s", node.name.c_str());
    ImGui::EndDragDropSource();
}

// In NodeEditor canvas area
if (ImGui::BeginDragDropTarget()) {
    if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("NODE_TYPE")) {
        NodeType type = *(const NodeType*)payload->Data;
        ImVec2 mouse_pos = ImGui::GetMousePos();
        CreateNodeAtPosition(type, mouse_pos);
    }
    ImGui::EndDragDropTarget();
}
```

---

## 9. Risk Assessment

### 9.1 Risk Matrix

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Scope creep | High | Medium | Strict phase boundaries, MVP first |
| Performance with many nodes | Medium | Low | Virtualized list, lazy loading |
| Breaking existing workflows | High | Low | Keep right-click menu working |
| Inconsistent node behavior | Medium | Medium | Standardized NodeMetadata struct |
| User confusion during transition | Medium | Medium | "Also in menu" badges, tutorial |
| Technical debt (duplicate code) | High | Medium | Tool panels share logic with nodes |
| Memory bloat | Medium | Low | Metadata loaded on-demand |
| Plugin compatibility | Medium | Low | Plugin nodes use same NodeMetadata struct |
| Cross-platform drag-drop | Medium | Medium | Test on Windows/Linux/macOS early |
| Node parameter persistence | Medium | Low | Save/load handles new node types |

### 9.2 Backend Impact Clarification: DuckDB/Arrow

**IMPORTANT: This upgrade does NOT affect DuckDB/Arrow backend.**

The upgrade is **UI-only**. The execution layer remains completely unchanged:

```
┌─────────────────────────────────────────────────────────────────┐
│                   UI LAYER (what's changing)                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Node Browser │    │   Canvas     │    │  Properties  │       │
│  │   (NEW)      │───>│  (existing)  │<───│  (existing)  │       │
│  └──────────────┘    └──────┬───────┘    └──────────────┘       │
├─────────────────────────────┼───────────────────────────────────┤
│                   EXECUTION LAYER (unchanged)                    │
│                             ▼                                    │
│                   ┌──────────────────┐                          │
│                   │ PipelineExecutor │                          │
│                   └────────┬─────────┘                          │
│              ┌─────────────┼─────────────┐                      │
│              ▼             ▼             ▼                      │
│        ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│        │  DuckDB  │  │  Arrow   │  │  Data    │                │
│        │Connector │  │ Dataset  │  │ Registry │                │
│        └──────────┘  └──────────┘  └──────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

**Components NOT affected:**

| Component | File | Status |
|-----------|------|--------|
| DuckDBConnector | `src/core/duckdb_connector.h/cpp` | Unchanged |
| ArrowDataset | `src/core/arrow_dataset.h/cpp` | Unchanged |
| ArrowConverters | `src/core/arrow_converters.h/cpp` | Unchanged |
| PipelineExecutor | `src/core/pipeline_executor.h/cpp` | Unchanged |
| DataRegistry | `src/core/data_registry.h/cpp` | Unchanged |
| ExecutionMode::DuckDBPipeline | `src/gui/node_editor.h` | Unchanged |

**Why no impact:**
- Node Browser is purely UI for discovering nodes
- Same `NodeType` enum values used
- Same node creation code path
- Same `PipelineExecutor::ExecutePipeline()` called on execution
- Same DuckDB SQL generated and executed

---

## 10. Success Metrics

| Metric | Target |
|--------|--------|
| Node Browser load time | < 100ms |
| Search response time | < 50ms |
| Drag-drop latency | < 16ms (60fps) |
| Node coverage | 80% of Tools menu as nodes |
| User adoption | 60% use Node Browser vs right-click |

---

## 11. Open Questions

1. **DB Connectors**: Build in-house or wait for KNIME integration?
2. **Loop Nodes**: How to handle iteration state in canvas?
3. **Template Nodes**: Allow clicking for "request feature" feedback?
4. **Undo/Redo**: Extend to node parameter changes?
5. **Multi-select**: Drag multiple nodes from browser?

---

## 12. Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | | | |
| Project Lead | | | |

---

## Appendix A: Node Inventory (KNIME Analysis)

### A.1 Summary

| Category | Currently Have | Can Add | Priority |
|----------|----------------|---------|----------|
| **I/O** | 7 | 17 | P1 |
| **Manipulation** | 10 | 18 | P1 |
| **Views** | 6 | 6 | P2 |
| **Analytics/ML** | 8 | 19 | P1 |
| **JSON/XML** | 1 | 8 | P2 |
| **Workflow** | 0 | 8 | P3 |
| **Reporting** | 0 | 5 | P3 |
| **Widgets** | 0 | 7 | P3 |
| **Model I/O** | 1 (menu) | 4 | P1 |
| **Total** | 33 | **92** | |

### A.2 Model Save/Load Clarification

**IMPORTANT:** Model save/load exists in codebase but only in Tools menu, NOT as nodes.

**Existing Infrastructure (src/core/):**
| File | Purpose |
|------|---------|
| `model_format.h` | Format definitions (CyxModel, ONNX, Safetensors, GGUF) |
| `model_exporter.h/cpp` | Export trained models |
| `model_importer.h/cpp` | Import/load models |
| `cyxmodel_format.h/cpp` | Native .cyxmodel format (ZIP archive) |
| `checkpoint_manager.h/cpp` | Training checkpoints |

**Supported Export Formats:**
| Format | Extension | Features |
|--------|-----------|----------|
| CyxModel | `.cyxmodel` | Native: weights + optimizer + history + graph |
| ONNX | `.onnx` | Interoperability with PyTorch/TensorFlow |
| Safetensors | `.safetensors` | Safe, fast tensor serialization |
| GGUF | `.gguf` | GGML format for LLM deployment |

**Quantization Options:** FP16, BF16, INT8, INT4, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0

**Nodes to Add (P1):**
| Node | Description |
|------|-------------|
| `ModelWriter` | Save trained model to file (output node) |
| `ModelReader` | Load trained model from file (input node) |
| `CheckpointSaver` | Save checkpoint during training loop |
| `CheckpointLoader` | Resume training from checkpoint |

### A.3 I/O Nodes

| Node | Description | Have? | Priority |
|------|-------------|-------|----------|
| CSV Reader | Read CSV files | ✅ | - |
| Excel Reader | Read .xlsx/.xls | ✅ | - |
| Parquet Reader | Read Parquet | ✅ | - |
| JSON Reader | Read JSON | ✅ | - |
| HDF5 Reader | Read HDF5 | ✅ | - |
| SQL Query | Execute SQL | ✅ | - |
| REST API | HTTP requests | ✅ | - |
| **Google Sheets Connector** | Read/write Google Sheets | ❌ | P2 |
| **Google Drive Connector** | Access Google Drive | ❌ | P3 |
| **Image Reader** | Batch read images | ❌ | P1 |
| **Line Reader** | Read line by line | ❌ | P2 |
| **PMML Reader** | Import PMML models | ❌ | P3 |
| **PMML Writer** | Export PMML models | ❌ | P3 |
| **Model Reader** | Load trained model | ❌ | P1 |
| **Model Writer** | Save trained model | ❌ | P1 |
| **Send Email** | Send with attachments | ❌ | P3 |
| **Decompress Files** | Unzip, untar | ❌ | P2 |
| **Create Folder** | Create directory | ❌ | P3 |
| **Delete Files** | Remove files | ❌ | P3 |
| **Transfer Files** | Copy/move files | ❌ | P3 |
| **Fixed Width Reader** | Fixed-column files | ❌ | P3 |
| **Table Creator** | Create table manually | ❌ | P2 |
| **Checkpoint Saver** | Save training checkpoint | ❌ | P1 |
| **Checkpoint Loader** | Resume from checkpoint | ❌ | P1 |

### A.4 Manipulation Nodes

| Node | Description | Have? | Priority |
|------|-------------|-------|----------|
| Row Filter | Filter by condition | ✅ | - |
| Column Filter | Select columns | ✅ | - |
| Joiner | Join tables | ✅ | - |
| GroupBy | Aggregate | ✅ | - |
| Sorter | Sort rows | ✅ | - |
| Missing Value | Handle nulls | ✅ | - |
| Normalizer | Scale values | ✅ | - |
| One-Hot Encoder | Categorical encoding | ✅ | - |
| Concatenate | Stack tables | ✅ | - |
| Rename Columns | Rename | ✅ | - |
| **Cross Joiner** | Cartesian product | ❌ | P2 |
| **Table Transposer** | Rows ↔ Columns | ❌ | P1 |
| **Unpivot** | Wide to long | ❌ | P1 |
| **Column Combiner** | Merge columns | ❌ | P2 |
| **Column Splitter** | Split by delimiter | ❌ | P1 |
| **String Replacer** | Find/replace | ❌ | P1 |
| **Column Aggregator** | Aggregate multiple | ❌ | P2 |
| **Ungroup** | Explode list column | ❌ | P2 |
| **Type Converter** | Double→Int, etc. | ❌ | P1 |
| **Number Rounder** | Round decimals | ❌ | P2 |
| **One to Many** | Duplicate rows | ❌ | P3 |
| **Rank** | Add rank column | ❌ | P1 |
| **Numeric Binner** | Bin into categories | ❌ | P1 |
| **Category to Number** | Label encoding | ❌ | P1 |
| **Row ID** | Generate identifiers | ❌ | P2 |
| **Math Formula** | Column expressions | ❌ | P1 |
| **Rule Engine** | If-then-else rules | ❌ | P2 |
| **Duplicate Filter** | Remove duplicates | ✅ | - |

### A.5 Views/Visualization Nodes

| Node | Description | Have? | Priority |
|------|-------------|-------|----------|
| Bar Chart | Bar visualization | ✅ | - |
| Line Plot | Line chart | ✅ | - |
| Scatter Plot | XY scatter | ✅ | - |
| Histogram | Distribution | ✅ | - |
| Heatmap | 2D heatmap | ✅ | - |
| Table View | Data table | ✅ | - |
| **Pie Chart** | Pie/donut | ❌ | P1 |
| **Box Plot** | Box and whisker | ❌ | P1 |
| **Parallel Coordinates** | Multi-dim | ❌ | P2 |
| **Sunburst Chart** | Hierarchical pie | ❌ | P3 |
| **Scatter Matrix** | Pairwise scatter | ❌ | P2 |
| **Stacked Area Chart** | Stacked series | ❌ | P2 |

### A.6 Analytics/ML Nodes

| Node | Description | Have? | Priority |
|------|-------------|-------|----------|
| Descriptive Stats | Summary stats | ✅ | - |
| Correlation Matrix | Correlations | ✅ | - |
| PCA | Dim reduction | ✅ | - |
| K-Means | Clustering | ✅ | - |
| Decision Tree | Tree classifier | ✅ | - |
| Random Forest | Ensemble | ✅ | - |
| DBSCAN | Density clustering | ✅ | - |
| Hierarchical Clustering | Dendogram | ✅ | - |
| **Gradient Boosted Trees** | XGBoost-style | ❌ | P1 |
| **SVM Learner** | Support vectors | ❌ | P1 |
| **Naive Bayes** | Naive Bayes | ❌ | P1 |
| **K Nearest Neighbor** | KNN | ❌ | P1 |
| **Logistic Regression** | Logistic | ❌ | P1 |
| **Linear Regression** | Linear model | ❌ | P1 |
| **Polynomial Regression** | Polynomial fit | ❌ | P2 |
| **Association Rules** | Market basket | ❌ | P3 |
| **k-Medoids** | Alt to k-means | ❌ | P3 |
| **Silhouette Score** | Cluster quality | ❌ | P2 |
| **Low Variance Filter** | Feature selection | ❌ | P2 |
| **Feature Selection** | Auto feature select | ❌ | P2 |
| **Crosstab** | Cross tabulation | ❌ | P2 |
| **Distance Matrix** | Compute distances | ❌ | P2 |
| **Similarity Search** | Find similar rows | ❌ | P3 |
| **Tree Ensemble** | Ensemble predictor | ❌ | P2 |
| **Scorer** | Evaluate predictions | ❌ | P1 |
| **Partitioner** | Train/test split | ❌ | P1 |
| **X-Partitioner** | Cross-val split | ❌ | P2 |

### A.7 JSON/XML Nodes

| Node | Description | Have? | Priority |
|------|-------------|-------|----------|
| JSON Reader | Read JSON file | ✅ | - |
| **JSON Path** | Extract with JSONPath | ❌ | P1 |
| **JSON to Table** | Flatten JSON | ❌ | P1 |
| **Table to JSON** | Convert to JSON | ❌ | P1 |
| **XML Reader** | Read XML file | ❌ | P2 |
| **XPath** | Extract with XPath | ❌ | P2 |
| **XSLT** | Transform XML | ❌ | P3 |
| **JSON Schema Validator** | Validate JSON | ❌ | P2 |
| **JSON Diff** | Compare JSON | ❌ | P3 |

### A.8 Workflow Control Nodes

| Node | Description | Have? | Priority |
|------|-------------|-------|----------|
| **Loop Start** | Begin iteration | ❌ | P2 |
| **Loop End** | End iteration | ❌ | P2 |
| **Counting Loop** | Fixed iterations | ❌ | P2 |
| **IF Switch** | Conditional branch | ❌ | P2 |
| **CASE Switch** | Multi-way branch | ❌ | P3 |
| **Try/Catch** | Error handling | ❌ | P3 |
| **Wait** | Pause execution | ❌ | P3 |
| **Breakpoint** | Debug pause | ❌ | P3 |

### A.9 Reporting Nodes

| Node | Description | Have? | Priority |
|------|-------------|-------|----------|
| **Report PDF Writer** | Export to PDF | ❌ | P2 |
| **Report HTML Writer** | Export to HTML | ❌ | P2 |
| **Report Template** | Define layout | ❌ | P3 |
| **Data to Report** | Add data section | ❌ | P3 |
| **Image to Report** | Add charts | ❌ | P3 |

### A.10 Widget Nodes (Interactive Inputs)

| Node | Description | Have? | Priority |
|------|-------------|-------|----------|
| **String Widget** | Text input | ❌ | P2 |
| **Integer Widget** | Number input | ❌ | P2 |
| **Selection Widget** | Dropdown | ❌ | P2 |
| **File Upload Widget** | File picker | ❌ | P2 |
| **Date/Time Widget** | Date picker | ❌ | P3 |
| **Slider Widget** | Range slider | ❌ | P2 |
| **Credentials Widget** | Secure input | ❌ | P3 |

## Appendix B: KNIME Reference Images

Located in `docs/Data Studio/knime/`:
- `io1.png` - `io4.png`: I/O nodes
- `manipulation1.png` - `manipulation5.png`: Manipulation nodes
- `views1.png` - `views3.png`: View nodes
- `DB1.png`, `DB2.png`: Database nodes
- `analytics1.png` - `analytics5.png`: Analytics nodes
- `workflow1.png`, `workflow2.png`: Workflow control nodes
- `tool&services.png`: HTTP/API nodes

## Appendix C: Migration Mapping

| Tools Menu Item | New Node Name | Priority |
|-----------------|---------------|----------|
| Data Profiler | DataProfiler | P1 |
| Missing Value Analysis | MissingValueAnalyzer | P1 |
| Correlation Matrix | CorrelationMatrix | P1 |
| K-Means Clustering | KMeansLearner | P1 |
| PCA | PCANode | P1 |
| Decision Tree | DecisionTreeLearner | P1 |
| ROC Curve / AUC | ROCCurve | P2 |
| Confusion Matrix | ConfusionMatrixView | P2 |
| Cross-Validation | CrossValidator | P2 |
| Normalization | Normalizer | P2 |
| Standardization | Standardizer | P2 |
| FFT | FFTNode | P3 |
| Spectrogram | SpectrogramNode | P3 |
| Tokenization | Tokenizer | P3 |
| TF-IDF | TFIDFVectorizer | P3 |
| Calculator | CalculatorNode | P4 |
| Regex Tester | RegexTester | P4 |

## Appendix D: Template Nodes (Future Implementation)

Template nodes appear in Node Browser with "Coming Soon" badge. They are defined but not implemented, serving as:
- User expectation signaling
- Future roadmap visibility
- Development planning reference

### D.1 Summary

| Category | Count | Complexity | Target Phase |
|----------|-------|------------|--------------|
| DB Connectors | 9 | Medium | Phase 2 (SQLite first) |
| Cloud Storage | 7 | Medium | When cloud features needed |
| External ML Services | 7 | Low-Medium | On user request |
| Advanced Workflow | 7 | High | Phase 3 |
| Reporting & Export | 6 | Medium | Phase 2 |
| Advanced Visualization | 6 | High | Phase 3+ |
| Big Data | 5 | High | Future roadmap |
| Deep Learning Advanced | 6 | High | Future roadmap |
| **Total Templates** | **53** | | |

### D.2 Database Connectors

| Template Node | Description | Future Dependency |
|---------------|-------------|-------------------|
| PostgreSQL Connector | Connect to PostgreSQL | libpq or ODBC |
| MySQL Connector | Connect to MySQL | mysql-connector-c++ |
| SQLite Connector | Connect to SQLite | sqlite3 (have it!) |
| MongoDB Connector | Connect to MongoDB | mongocxx driver |
| Oracle Connector | Connect to Oracle DB | Oracle client libs |
| MS SQL Connector | Connect to SQL Server | ODBC |
| Snowflake Connector | Connect to Snowflake | Snowflake SDK |
| DB Query Executor | Execute SQL on connection | Depends on connector |
| DB Table Writer | Write table to database | Depends on connector |

**Implementation Note:** Start with SQLite (already have dependency), then PostgreSQL.

### D.3 Cloud Storage

| Template Node | Description | Future Dependency |
|---------------|-------------|-------------------|
| AWS S3 Reader | Read from S3 bucket | aws-sdk-cpp |
| AWS S3 Writer | Write to S3 bucket | aws-sdk-cpp |
| Azure Blob Reader | Read from Azure Blob | azure-storage-cpp |
| Azure Blob Writer | Write to Azure Blob | azure-storage-cpp |
| Google Cloud Storage | Read/write GCS | google-cloud-cpp |
| Google Drive Connector | Access Google Drive | Google API + OAuth |
| Google Sheets Connector | Read/write Sheets | Google API + OAuth |

**Implementation Note:** Requires OAuth flow for Google services.

### D.4 External ML Services

| Template Node | Description | Future Dependency |
|---------------|-------------|-------------------|
| Google AutoML | Train with Vertex AI | Vertex AI SDK |
| Azure AutoML | Train with Azure ML | Azure ML SDK |
| AWS SageMaker | Train with SageMaker | AWS SDK |
| OpenAI Embeddings | Generate embeddings | REST API (curl) |
| HuggingFace Hub | Download models | huggingface_hub |
| MLflow Tracker | Log experiments | mlflow SDK |
| Weights & Biases | Log experiments | wandb SDK |

**Implementation Note:** REST API nodes (OpenAI) are easiest to implement first.

### D.5 Advanced Workflow Control

| Template Node | Description | Future Dependency |
|---------------|-------------|-------------------|
| Recursive Loop Start | Loop with recursion | State management design |
| Recursive Loop End | End recursive loop | State management design |
| Parallel Executor | Run branches in parallel | Thread pool |
| Map Node | Apply to each row | Iterator pattern |
| Reduce Node | Aggregate results | Accumulator pattern |
| Workflow Call | Execute sub-workflow | Metanode system |
| Component I/O | Reusable components | Component framework |

**Implementation Note:** Requires careful design for state management and execution model.

### D.6 Reporting & Export

| Template Node | Description | Future Dependency |
|---------------|-------------|-------------------|
| PDF Report Writer | Export to PDF | libharu or PDFium |
| HTML Report Writer | Export to HTML | Template engine |
| PowerPoint Writer | Export to PPTX | libpptx |
| Word Document Writer | Export to DOCX | libdocx |
| Dashboard Creator | Interactive HTML dashboard | HTML/JS generation |
| Email Sender | Send via SMTP | libcurl + SMTP |

**Implementation Note:** HTML Writer is simplest; PDF needs external library.

### D.7 Advanced Visualization

| Template Node | Description | Future Dependency |
|---------------|-------------|-------------------|
| 3D Scatter Plot | 3D point cloud | ImPlot3D or OpenGL |
| Interactive Dashboard | Web-based charts | Embedded browser/WebView |
| Geospatial Map | Map visualization | Mapbox GL or Leaflet |
| Network Graph | Node-link diagram | Force-directed algorithm |
| Sankey Diagram | Flow visualization | Custom renderer |
| Treemap | Hierarchical rectangles | Custom renderer |

**Implementation Note:** Consider ImPlot3D for 3D; others need custom rendering.

### D.8 Big Data & Streaming

| Template Node | Description | Future Dependency |
|---------------|-------------|-------------------|
| Apache Spark Connector | Distributed processing | Apache Spark |
| Dask Connector | Python parallel | Dask |
| Ray Connector | Distributed ML | Ray |
| Kafka Consumer | Read from Kafka | librdkafka |
| Kafka Producer | Write to Kafka | librdkafka |

**Implementation Note:** Enterprise feature; low priority for initial release.

### D.9 Deep Learning Advanced

| Template Node | Description | Future Dependency |
|---------------|-------------|-------------------|
| SHAP Explainer | Feature importance | SHAP library (Python) |
| LIME Explainer | Local interpretability | LIME library (Python) |
| Hyperband Tuner | Hyperparameter search | Algorithm implementation |
| Neural Architecture Search | Auto architecture | NAS framework |
| Federated Learning | Privacy-preserving ML | FL framework |
| Model Quantizer | Compress model | TensorRT or ONNX Runtime |

**Implementation Note:** SHAP/LIME can use Python bridge; others are complex.

### D.10 Template JSON Schema

Templates are defined in `resources/node_templates/*.json`:

```json
{
  "templates": [
    {
      "type": "PostgreSQLConnector",
      "category": "DB",
      "name": "PostgreSQL Connector",
      "icon": "ICON_FA_DATABASE",
      "status": "template",
      "badge": "Coming Soon",
      "description": "Connect to a PostgreSQL database server",
      "tooltip": "Requires PostgreSQL client library (planned for Phase 2)",
      "inputs": [],
      "outputs": [
        {"name": "DB Connection", "type": "DBConnection", "description": "Database connection handle"}
      ],
      "parameters": [
        {"name": "host", "type": "string", "default": "localhost", "description": "Database host"},
        {"name": "port", "type": "int", "default": "5432", "description": "Port number"},
        {"name": "database", "type": "string", "default": "", "description": "Database name"},
        {"name": "username", "type": "string", "default": "", "description": "Username"},
        {"name": "password", "type": "password", "default": "", "description": "Password"}
      ],
      "implementation_notes": "Use libpq for connection. Consider connection pooling.",
      "target_phase": "Phase 2",
      "dependencies": ["libpq"],
      "effort_estimate": "Medium",
      "user_votes": 0
    }
  ]
}
```

### D.11 Template Behavior in UI

1. **Node Browser**: Shows with grayed icon + "Coming Soon" badge
2. **Drag to Canvas**: Blocked with tooltip "This node is planned for future release"
3. **Info Panel**: Shows full description + "Vote for this feature" button
4. **Vote Tracking**: Increment `user_votes` to prioritize development
5. **Search**: Templates appear in search results (filtered by toggle)
