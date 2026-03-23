# Properties Panel Enhancement - Design Document

**Version**: 1.0
**Date**: 2026-03-23
**Status**: Design Phase

---

## 1. Executive Summary

Enhance the Properties Panel to provide KNIME-style configuration dialogs for complex nodes. Simple nodes keep inline properties, while complex nodes (data readers, ML algorithms, etc.) get dedicated configuration dialogs with live data preview.

---

## 2. Current State Analysis

### 2.1 Existing Components (Reusable)

| Component | File | Purpose | Reuse Potential |
|-----------|------|---------|-----------------|
| **TableViewer** | `src/gui/panels/table_viewer.h/cpp` | Multi-tab data viewer, pagination, filtering | High - embed in dialogs |
| **CSVDataset** | `src/core/datasets/csv_dataset.h/cpp` | CSV parsing with Arrow | High - data loading |
| **ArrowDataset** | `src/core/arrow_dataset.h/cpp` | Arrow table wrapper | High - data storage |
| **ArrowConverters** | `src/core/arrow_converters.h/cpp` | Type conversion | Medium |
| **DuckDBConnector** | `src/core/duckdb_connector.h/cpp` | SQL queries on data | Medium |
| **Properties Panel** | `src/gui/properties.h/cpp` | Basic parameter editing | Refactor |

### 2.2 Current Properties Panel Features

```
+---------------------------+
| Properties                |
+---------------------------+
| [General Section]         |
|   Node Name: [TextField]  |
|   Position: X, Y          |
+---------------------------+
| [Parameters Section]      |
|   param1: [TextField]     |
|   param2: [Dropdown]      |
|   param3: [Checkbox]      |
+---------------------------+
| [Advanced Section]        |
|   (collapsed)             |
+---------------------------+
```

**Limitations:**
- No data preview for I/O nodes
- No column selection UI
- No live validation
- Complex nodes need many parameters that clutter the panel

---

## 3. Node Classification by Configuration Complexity

### 3.1 Complexity Levels

| Level | Description | Configuration UI | Example Nodes |
|-------|-------------|------------------|---------------|
| **Simple** | 0-3 parameters, no data preview needed | Inline in Properties Panel | Dense, ReLU, Dropout, Flatten |
| **Medium** | 4-8 parameters, optional preview | Expandable section OR small dialog | K-Means, PCA, StandardScaler |
| **Complex** | Many parameters, requires data preview | Dedicated dialog with preview | CSV Reader, SQL Query, Data Profiler |
| **Visual** | Requires visual configuration | Specialized editor | Filter Designer, Regex Tester |

### 3.2 Node Dialog Requirements Matrix

#### Data Source Nodes (Complex - Need Dialog)

| Node | Dialog Features Needed |
|------|------------------------|
| **CSV Reader** | File picker, delimiter selector, header checkbox, encoding dropdown, column type override, **table preview** |
| **Excel Reader** | File picker, sheet selector, range input, header checkbox, **table preview** |
| **Parquet Reader** | File picker, column selector, row group filter, **schema preview** |
| **JSON Reader** | File picker, JSONPath expression, flatten options, **tree preview** |
| **SQL Query** | Connection config, SQL editor with syntax highlight, **result preview** |
| **HDF5 Reader** | File picker, dataset selector, slice input, **data preview** |
| **REST API** | URL input, method selector, headers editor, body editor, auth config, **response preview** |
| **Dataset Input** | Dataset browser, split selector, batch config, **sample preview** |

#### Data Transform Nodes (Medium - Expandable or Small Dialog)

| Node | Dialog Features Needed |
|------|------------------------|
| **Row Filter** | Column selector, condition builder, **preview filtered rows** |
| **Column Filter** | Column multi-select with checkboxes, **preview selected columns** |
| **Joiner** | Left/right column selectors, join type dropdown, **preview join result** |
| **GroupBy** | Group columns selector, aggregation builder, **preview grouped data** |
| **Sorter** | Sort column selector, order toggle, **preview sorted data** |
| **Missing Value** | Column selector, strategy per column, **preview changes** |
| **Normalizer** | Column selector, method dropdown, **preview normalized values** |
| **One-Hot Encoder** | Column selector, drop first checkbox, **preview encoded columns** |

#### Analytics Nodes (Medium to Complex)

| Node | Dialog Features Needed |
|------|------------------------|
| **K-Means** | n_clusters slider, init method, max_iter, **cluster visualization preview** |
| **PCA** | n_components, whiten checkbox, **variance explained chart** |
| **Decision Tree** | max_depth, min_samples, criterion, **tree visualization** |
| **Random Forest** | n_estimators, max_depth, **feature importance preview** |
| **Correlation Matrix** | Column selector, method dropdown, **heatmap preview** |
| **Data Profiler** | Column selector, stats toggles, **full profiling report dialog** |

#### ML Layer Nodes (Simple - Inline Properties)

| Node | Inline Parameters |
|------|-------------------|
| **Dense** | units, activation |
| **Conv2D** | filters, kernel_size, strides, padding, activation |
| **LSTM** | units, return_sequences, dropout |
| **Dropout** | rate |
| **BatchNorm** | momentum, epsilon |
| **Attention** | num_heads, key_dim |

#### Training Nodes (Simple to Medium)

| Node | Configuration Type |
|------|-------------------|
| **Optimizer** | Inline: lr, momentum, weight_decay |
| **Loss Function** | Inline: loss type dropdown |
| **LR Scheduler** | Medium: schedule type, parameters, **preview schedule curve** |
| **Metrics** | Inline: metric checkboxes |

#### Utility Nodes (Visual - Specialized Dialogs)

| Node | Specialized UI |
|------|----------------|
| **Calculator** | Expression editor with variable list |
| **Regex Tester** | Regex input, test string, **live match highlighting** |
| **Filter Designer** | Frequency response plot, pole-zero editor |
| **JSONPath Extractor** | JSONPath input, JSON tree view, **path result preview** |

---

## 4. Dialog System Architecture

### 4.1 Class Hierarchy

```
NodeConfigDialog (Base)
├── DataSourceConfigDialog
│   ├── CSVReaderConfigDialog
│   ├── ExcelReaderConfigDialog
│   ├── ParquetReaderConfigDialog
│   ├── JSONReaderConfigDialog
│   ├── SQLQueryConfigDialog
│   └── RESTAPIConfigDialog
├── DataTransformConfigDialog
│   ├── FilterConfigDialog (Row/Column)
│   ├── JoinerConfigDialog
│   ├── GroupByConfigDialog
│   └── MissingValueConfigDialog
├── AnalyticsConfigDialog
│   ├── ClusteringConfigDialog
│   ├── DimensionReductionConfigDialog
│   └── DataProfilerConfigDialog
└── UtilityConfigDialog
    ├── RegexTesterDialog
    ├── FilterDesignerDialog
    └── CalculatorDialog
```

### 4.2 Base Dialog Interface

```cpp
class NodeConfigDialog {
public:
    virtual ~NodeConfigDialog() = default;

    // Open dialog for a specific node
    virtual bool Open(MLNode* node) = 0;

    // Render dialog (call each frame while open)
    virtual void Render() = 0;

    // Check if dialog is open
    virtual bool IsOpen() const = 0;

    // Get/set configuration
    virtual NodeConfig GetConfig() const = 0;
    virtual void SetConfig(const NodeConfig& config) = 0;

protected:
    // Common UI helpers
    void RenderHeader(const std::string& title, const std::string& icon);
    void RenderFooter();  // OK, Cancel, Apply buttons
    void RenderPreviewSection();

    // Data preview (reuses TableViewer)
    void ShowDataPreview(std::shared_ptr<arrow::Table> data);

    MLNode* node_ = nullptr;
    bool is_open_ = false;
    bool config_changed_ = false;
};
```

### 4.3 Dialog Layout Template

```
+------------------------------------------------------------------+
| [Icon] CSV Reader Configuration                           [X]    |
+------------------------------------------------------------------+
|  +------------------------+  +-------------------------------+   |
|  | Settings               |  | Preview                       |   |
|  +------------------------+  +-------------------------------+   |
|  | File: [________] [...]|  | +---------------------------+ |   |
|  | Delimiter: [,  v]     |  | | Col1 | Col2 | Col3 | Col4 | |   |
|  | Has Header: [x]       |  | |------|------|------|------| |   |
|  | Encoding: [UTF-8 v]   |  | | val  | val  | val  | val  | |   |
|  |                        |  | | val  | val  | val  | val  | |   |
|  | [Column Types...]     |  | | val  | val  | val  | val  | |   |
|  |                        |  | +---------------------------+ |   |
|  | Skip Rows: [0]        |  | Showing 100 of 1,234 rows     |   |
|  | Max Rows: [10000]     |  +-------------------------------+   |
|  +------------------------+                                      |
+------------------------------------------------------------------+
| [Apply]                               [Cancel]        [OK]       |
+------------------------------------------------------------------+
```

### 4.4 Properties Panel Integration

```cpp
class PropertiesPanel {
    // ...

    void RenderNodeProperties(MLNode& node) {
        auto* metadata = registry_.GetMetadata(node.type);

        if (metadata->RequiresConfigDialog()) {
            // Show "Configure..." button
            if (ImGui::Button(ICON_FA_GEAR " Configure...")) {
                OpenConfigDialog(node);
            }

            // Show summary of current config
            RenderConfigSummary(node);
        } else {
            // Render inline parameters
            RenderInlineParameters(node, metadata);
        }
    }

    void OpenConfigDialog(MLNode& node) {
        auto dialog = DialogFactory::Create(node.type);
        dialog->Open(&node);
        active_dialog_ = std::move(dialog);
    }
};
```

---

## 5. Detailed Dialog Specifications

### 5.1 CSV Reader Configuration Dialog

**Size:** 900x600 (resizable, min 700x450)

**Layout:**
```
+------------------------------------------------------------------+
| [CSV] CSV Reader                                          [X]    |
+==================================================================+
| File Selection                                                    |
| +--------------------------------------------------------------+ |
| | Path: [/path/to/file.csv                    ] [Browse...]    | |
| | Recent: [dropdown of recent files]                           | |
| +--------------------------------------------------------------+ |
|                                                                   |
| +------------------------+  +----------------------------------+ |
| | Parsing Options        |  | Data Preview                     | |
| +------------------------+  +----------------------------------+ |
| | Delimiter:  [, v]     |  | +------------------------------+ | |
| |   , (Comma)           |  | | Name  | Age | City   | Sal   | | |
| |   ; (Semicolon)       |  | |-------|-----|--------|-------| | |
| |   \t (Tab)            |  | | John  | 32  | NYC    | 75000 | | |
| |   | (Pipe)            |  | | Jane  | 28  | LA     | 82000 | | |
| |   Other: [_]          |  | | Bob   | 45  | Chicago| 95000 | | |
| |                        |  | +------------------------------+ | |
| | Header Row: [x]       |  | Rows: 1,234 | Cols: 4            | |
| | Skip Rows:  [0    ]   |  +----------------------------------+ |
| | Encoding:   [UTF-8 v] |                                       |
| | Quote Char: [" v]     |  +----------------------------------+ |
| |                        |  | Column Configuration             | |
| | [ ] Limit Rows        |  +----------------------------------+ |
| |     Max: [10000]      |  | Name   | Type     | Include      | |
| +------------------------+  | Name   | [String v] | [x]        | |
|                             | Age    | [Int64  v] | [x]        | |
|                             | City   | [String v] | [x]        | |
|                             | Salary | [Float64v] | [x]        | |
|                             +----------------------------------+ |
+------------------------------------------------------------------+
|                                    [Cancel]    [Apply]    [OK]   |
+------------------------------------------------------------------+
```

**Features:**
- Live preview updates as settings change
- Auto-detect delimiter from first few lines
- Column type inference with override
- Recent files dropdown
- Error display for parse failures

### 5.2 Row Filter Configuration Dialog

**Size:** 800x500

**Layout:**
```
+------------------------------------------------------------------+
| [Filter] Row Filter                                       [X]    |
+==================================================================+
| Filter Conditions                                                 |
| +--------------------------------------------------------------+ |
| | Column     | Operator      | Value                  | [X]   | |
| | [Age    v] | [>         v] | [30                   ]| [Del] | |
| | [City   v] | [equals    v] | [NYC                  ]| [Del] | |
| | [Salary v] | [between   v] | [50000] and [100000  ]| [Del] | |
| +--------------------------------------------------------------+ |
| [+ Add Condition]                                                 |
|                                                                   |
| Combine: ( ) AND all conditions   (x) OR any condition           |
|                                                                   |
| +--------------------------------------------------------------+ |
| | Preview (Matching Rows: 234 of 1,234)                        | |
| +--------------------------------------------------------------+ |
| | Name  | Age | City   | Salary |                               |
| | John  | 32  | NYC    | 75000  |                               |
| | Jane  | 35  | NYC    | 82000  |                               |
| +--------------------------------------------------------------+ |
+------------------------------------------------------------------+
|                                    [Cancel]    [Apply]    [OK]   |
+------------------------------------------------------------------+
```

### 5.3 K-Means Clustering Configuration Dialog

**Size:** 850x550

**Layout:**
```
+------------------------------------------------------------------+
| [Cluster] K-Means Clustering                              [X]    |
+==================================================================+
| +---------------------------+  +-------------------------------+ |
| | Algorithm Settings        |  | Cluster Preview               | |
| +---------------------------+  +-------------------------------+ |
| |                           |  |     [2D Scatter Plot]         | |
| | Number of Clusters (K)    |  |                               | |
| | [----o-----------] 5      |  |   *  *    o  o  o             | |
| |  2              20        |  |  * *  *  o  o   o             | |
| |                           |  |    * *     o o                 | |
| | Initialization Method     |  |                  + + +        | |
| | (x) k-means++             |  |               +  + +  +       | |
| | ( ) Random                |  |                + +            | |
| | ( ) Manual                |  |                               | |
| |                           |  | Legend: [*] C1 [o] C2 [+] C3  | |
| | Max Iterations: [300]     |  +-------------------------------+ |
| | Tolerance: [0.0001]       |                                    |
| | Random Seed: [42]         |  +-------------------------------+ |
| |                           |  | Elbow Plot (Choose K)         | |
| | Feature Columns:          |  +-------------------------------+ |
| | [x] Age                   |  | Inertia                       | |
| | [x] Salary                |  |  |    *                        | |
| | [ ] City (categorical)    |  |  |      *  *                   | |
| |                           |  |  |           *  *  *  *        | |
| +---------------------------+  |  +--+--+--+--+--+--+--+-> K    | |
|                                +-------------------------------+ |
+------------------------------------------------------------------+
|                                    [Cancel]    [Apply]    [OK]   |
+------------------------------------------------------------------+
```

### 5.4 Data Profiler Configuration Dialog

**Size:** 1000x700 (large, detailed report)

**Layout:**
```
+------------------------------------------------------------------+
| [Profile] Data Profiler                                   [X]    |
+==================================================================+
| Dataset: iris.csv (150 rows, 5 columns)                          |
| +--------------------------------------------------------------+ |
| | [Overview] [Columns] [Correlations] [Missing] [Duplicates]   | |
| +--------------------------------------------------------------+ |
| |                                                               | |
| | OVERVIEW                                                      | |
| | +------------------+  +------------------+  +---------------+ | |
| | | Dataset Stats    |  | Variable Types   |  | Alerts        | | |
| | | Rows: 150        |  | Numeric: 4       |  | [!] 3 high    | | |
| | | Cols: 5          |  | Categorical: 1   |  |     correlation| |
| | | Missing: 0%      |  | DateTime: 0      |  | [!] 0 missing | | |
| | | Duplicates: 0    |  | Text: 0          |  |               | | |
| | +------------------+  +------------------+  +---------------+ | |
| |                                                               | |
| | COLUMN DETAILS                                                | |
| | +----------------------------------------------------------+ | |
| | | Column       | Type    | Missing | Unique | Mean    | ... | |
| | | sepal_length | float64 | 0%      | 35     | 5.84    |     | |
| | | sepal_width  | float64 | 0%      | 23     | 3.05    |     | |
| | | petal_length | float64 | 0%      | 43     | 3.76    |     | |
| | | species      | string  | 0%      | 3      | -       |     | |
| | +----------------------------------------------------------+ | |
| |                                                               | |
| +--------------------------------------------------------------+ |
+------------------------------------------------------------------+
| [Export Report...]                         [Close]               |
+------------------------------------------------------------------+
```

---

## 6. Implementation Plan

### Phase 1: Foundation (Core Infrastructure)

| Task | Description | Files |
|------|-------------|-------|
| 1.1 | Create `NodeConfigDialog` base class | `src/gui/dialogs/node_config_dialog.h/cpp` |
| 1.2 | Create `DialogFactory` for dialog creation | `src/gui/dialogs/dialog_factory.h/cpp` |
| 1.3 | Add `RequiresConfigDialog()` to NodeMetadata | `src/core/node_metadata.h` |
| 1.4 | Integrate dialog opening in Properties Panel | `src/gui/properties.cpp` |
| 1.5 | Create embedded TableViewer widget | `src/gui/widgets/table_preview.h/cpp` |

### Phase 2: Data Source Dialogs

| Task | Description | Priority |
|------|-------------|----------|
| 2.1 | CSV Reader Configuration Dialog | P1 |
| 2.2 | Excel Reader Configuration Dialog | P1 |
| 2.3 | Parquet Reader Configuration Dialog | P2 |
| 2.4 | JSON Reader Configuration Dialog | P2 |
| 2.5 | SQL Query Configuration Dialog | P2 |
| 2.6 | REST API Configuration Dialog | P3 |

### Phase 3: Data Transform Dialogs

| Task | Description | Priority |
|------|-------------|----------|
| 3.1 | Row Filter Configuration Dialog | P1 |
| 3.2 | Column Filter Configuration Dialog | P1 |
| 3.3 | Joiner Configuration Dialog | P2 |
| 3.4 | GroupBy Configuration Dialog | P2 |
| 3.5 | Missing Value Configuration Dialog | P2 |

### Phase 4: Analytics Dialogs

| Task | Description | Priority |
|------|-------------|----------|
| 4.1 | Data Profiler Dialog (full report) | P1 |
| 4.2 | K-Means Clustering Dialog | P2 |
| 4.3 | PCA Dialog with variance plot | P2 |
| 4.4 | Correlation Matrix Dialog | P2 |

### Phase 5: Utility Dialogs

| Task | Description | Priority |
|------|-------------|----------|
| 5.1 | Regex Tester Dialog (live matching) | P2 |
| 5.2 | Filter Designer Dialog (frequency response) | P3 |
| 5.3 | Calculator Dialog (expression builder) | P3 |

---

## 7. UI/UX Guidelines

### 7.1 Dialog Standards

- **Minimum Size:** 600x400 for simple, 900x600 for complex
- **Resizable:** Yes, with minimum constraints
- **Modal:** Yes (blocks main window interaction)
- **Keyboard:** Escape to cancel, Enter to confirm
- **Preview:** Updates within 200ms of setting change

### 7.2 Button Placement

```
[Apply]                                    [Cancel]    [OK]
```

- **Apply:** Save without closing (left side)
- **Cancel:** Discard changes and close (right side)
- **OK:** Save and close (rightmost)

### 7.3 Color Coding

- **Error:** Red border/background for invalid inputs
- **Warning:** Yellow for potential issues (e.g., large dataset)
- **Success:** Green checkmark for validated inputs
- **Info:** Blue for hints and tips

### 7.4 Preview Behavior

- Preview shows first 100-1000 rows
- "Loading..." spinner for large datasets
- Error message if preview fails
- Row count shown: "Showing 100 of 10,234 rows"

---

## 8. Technical Considerations

### 8.1 Performance

- **Lazy Loading:** Only load preview data when dialog opens
- **Debouncing:** Wait 200ms after settings change before updating preview
- **Caching:** Cache parsed data to avoid re-reading file
- **Background Thread:** Load large files in background

### 8.2 Memory Management

- Dialogs own their preview data (unique_ptr)
- Release preview data when dialog closes
- Limit preview to configurable row count (default 1000)

### 8.3 Error Handling

- Show inline error messages near problematic fields
- Disable OK button if configuration is invalid
- Allow Cancel even with errors
- Log errors for debugging

---

## 9. File Structure

```
cyxwiz-engine/src/gui/
├── dialogs/
│   ├── node_config_dialog.h          # Base class
│   ├── node_config_dialog.cpp
│   ├── dialog_factory.h              # Factory for creating dialogs
│   ├── dialog_factory.cpp
│   ├── data_source/
│   │   ├── csv_reader_dialog.h
│   │   ├── csv_reader_dialog.cpp
│   │   ├── excel_reader_dialog.h
│   │   ├── excel_reader_dialog.cpp
│   │   └── ...
│   ├── data_transform/
│   │   ├── row_filter_dialog.h
│   │   ├── row_filter_dialog.cpp
│   │   └── ...
│   ├── analytics/
│   │   ├── data_profiler_dialog.h
│   │   ├── data_profiler_dialog.cpp
│   │   └── ...
│   └── utility/
│       ├── regex_tester_dialog.h
│       └── ...
├── widgets/
│   ├── table_preview.h               # Reusable table preview widget
│   ├── table_preview.cpp
│   ├── column_selector.h             # Multi-column checkbox list
│   ├── column_selector.cpp
│   ├── condition_builder.h           # Filter condition UI
│   └── condition_builder.cpp
└── properties.cpp                    # Updated to use dialogs
```

---

## 10. Success Metrics

| Metric | Target |
|--------|--------|
| Dialog open time | < 500ms |
| Preview load time | < 1s for 10K rows |
| Memory usage per dialog | < 50MB preview data |
| User clicks to configure CSV | 3-5 clicks |
| Error discovery rate | 100% before OK pressed |

---

## 11. Open Questions

1. **File Caching:** Should we cache file reads across dialog open/close?
2. **Undo/Redo:** Support configuration history within dialog?
3. **Templates:** Allow saving/loading dialog configurations as presets?
4. **Multi-Node:** Configure multiple nodes of same type at once?
5. **Validation:** Real-time vs on-apply validation?

---

## 12. Appendix: Node-to-Dialog Mapping

### Full List of Nodes Requiring Dialogs

| NodeType | Dialog Class | Complexity | Priority |
|----------|--------------|------------|----------|
| CSVFile | CSVReaderDialog | Complex | P1 |
| ExcelFile | ExcelReaderDialog | Complex | P1 |
| ParquetFile | ParquetReaderDialog | Medium | P2 |
| JSONFile | JSONReaderDialog | Medium | P2 |
| SQLQuery | SQLQueryDialog | Complex | P2 |
| HDF5Dataset | HDF5ReaderDialog | Medium | P3 |
| RESTAPISource | RESTAPIDialog | Complex | P3 |
| FilterRows | RowFilterDialog | Medium | P1 |
| SelectColumns | ColumnFilterDialog | Simple | P1 |
| JoinTables | JoinerDialog | Medium | P2 |
| GroupByAggregate | GroupByDialog | Medium | P2 |
| FillMissingValues | MissingValueDialog | Medium | P2 |
| KMeansCluster | KMeansDialog | Medium | P2 |
| PCANode | PCADialog | Medium | P2 |
| DataProfiler | DataProfilerDialog | Complex | P1 |
| CorrelationMatrix | CorrelationDialog | Medium | P2 |
| RegexTester | RegexTesterDialog | Visual | P2 |
| FilterDesigner | FilterDesignerDialog | Visual | P3 |
| CalculatorNode | CalculatorDialog | Visual | P3 |

### Nodes with Inline Properties Only

All ML Layer nodes (Dense, Conv2D, LSTM, etc.), Activation nodes, Pooling nodes, Training nodes (Optimizer, Loss, Metrics), and basic utility nodes remain with inline Properties Panel configuration.

---

## 13. Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | | | |
| Project Lead | | | |
