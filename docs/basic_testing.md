# Basic Testing & Model Import/Export Design

## Overview

This document outlines the architecture for adding model testing, weight export, and weight import capabilities to CyxWiz Engine. These features complete the ML pipeline: **Data Processing → Training → Testing → Export/Deploy**.

## Current State Summary

| Component | Status | Key Files |
|-----------|--------|-----------|
| Training System | Done | `training_executor.cpp`, `training_manager.cpp` |
| Checkpoint System | Done | `checkpoint_manager.cpp` (binary `.bin` + JSON) |
| Graph Compiler | Done | `graph_compiler.cpp` |
| Code Generation | Done | `node_editor_codegen.cpp` (PyTorch/TF/Keras) |
| Model Testing | **Missing** | - |
| Weight Export | **Missing** | Only internal `.bin` format |
| Weight Import | **Missing** | Only internal checkpoint loading |

---

## Part 1: Model Testing System

### 1.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Testing Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│   │ Test Data   │───▶│ TestExecutor │───▶│ TestingMetrics  │    │
│   │ (Batcher)   │    │              │    │ - Accuracy      │    │
│   └─────────────┘    │ Forward Pass │    │ - Precision     │    │
│                      │    Only      │    │ - Recall        │    │
│   ┌─────────────┐    │              │    │ - F1 Score      │    │
│   │ Trained     │───▶│ No Gradient  │    │ - Confusion Mat │    │
│   │ Model       │    │ Computation  │    │ - Per-class     │    │
│   └─────────────┘    └──────────────┘    └─────────────────┘    │
│                                                   │              │
│                                          ┌────────▼────────┐    │
│                                          │  Test Results   │    │
│                                          │  Panel (UI)     │    │
│                                          └─────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Data Structures

```cpp
// File: cyxwiz-engine/src/core/test_executor.h

#pragma once
#include <vector>
#include <map>
#include <string>
#include "training_executor.h"

namespace cyxwiz {

/// Per-class metrics for detailed analysis
struct ClassMetrics {
    std::string class_name;
    int class_id;
    int true_positives = 0;
    int false_positives = 0;
    int false_negatives = 0;
    int true_negatives = 0;
    float precision = 0.0f;
    float recall = 0.0f;
    float f1_score = 0.0f;
    int support = 0;  // Number of samples in this class
};

/// Confusion matrix representation
struct ConfusionMatrix {
    int num_classes;
    std::vector<std::vector<int>> matrix;  // [actual][predicted]
    std::vector<std::string> class_names;

    void Compute(const std::vector<int>& predictions,
                 const std::vector<int>& ground_truth);
    float GetAccuracy() const;
    std::vector<ClassMetrics> GetPerClassMetrics() const;
};

/// Complete test results
struct TestResults {
    // Overall metrics
    float accuracy = 0.0f;
    float macro_precision = 0.0f;   // Average across classes
    float macro_recall = 0.0f;
    float macro_f1 = 0.0f;
    float weighted_f1 = 0.0f;       // Weighted by class support

    // Loss metrics
    float test_loss = 0.0f;

    // Detailed results
    ConfusionMatrix confusion_matrix;
    std::vector<ClassMetrics> per_class_metrics;

    // Sample-level results (for detailed inspection)
    std::vector<int> predictions;
    std::vector<int> ground_truth;
    std::vector<float> confidences;  // Max softmax probability
    std::vector<std::vector<float>> all_probabilities;  // Full softmax output

    // Misclassified samples (indices into test set)
    std::vector<size_t> misclassified_indices;

    // Timing
    float total_time_seconds = 0.0f;
    float samples_per_second = 0.0f;
    int total_samples = 0;

    // State
    bool is_complete = false;
    std::string status_message;
};

/// Test progress callback
struct TestProgress {
    int current_batch;
    int total_batches;
    int samples_processed;
    int total_samples;
    float current_accuracy;  // Running accuracy
};

using TestProgressCallback = std::function<void(const TestProgress&)>;
using TestCompleteCallback = std::function<void(const TestResults&)>;

/// Executes model testing on test dataset
class TestExecutor {
public:
    TestExecutor();
    ~TestExecutor();

    /// Initialize with trained model and test data
    bool Initialize(
        std::shared_ptr<SequentialModel> model,
        const std::string& dataset_name,
        const PreprocessingConfig& preprocessing
    );

    /// Run testing (async)
    void RunTestAsync(
        TestProgressCallback on_progress = nullptr,
        TestCompleteCallback on_complete = nullptr
    );

    /// Run testing (sync, blocking)
    TestResults RunTest();

    /// Stop testing
    void Stop();

    /// Get results (after completion)
    const TestResults& GetResults() const { return results_; }

    /// Check if running
    bool IsRunning() const { return is_running_; }

    /// Configuration
    void SetBatchSize(int batch_size) { batch_size_ = batch_size; }
    void SetClassNames(const std::vector<std::string>& names) { class_names_ = names; }

private:
    std::shared_ptr<SequentialModel> model_;
    std::unique_ptr<DatasetBatcher> batcher_;
    TestResults results_;

    int batch_size_ = 32;
    std::vector<std::string> class_names_;

    std::atomic<bool> is_running_{false};
    std::atomic<bool> stop_requested_{false};

    void ProcessBatch(const Batch& batch);
    int ArgMax(const std::vector<float>& probs);
};

} // namespace cyxwiz
```

### 1.3 Testing Manager (Singleton)

```cpp
// File: cyxwiz-engine/src/core/test_manager.h

#pragma once
#include "test_executor.h"
#include "training_manager.h"

namespace cyxwiz {

/// Singleton manager for model testing
class TestManager {
public:
    static TestManager& Instance();

    /// Start test using current trained model
    bool StartTest(
        const std::string& checkpoint_path = "",  // Empty = use current model
        TestProgressCallback on_progress = nullptr,
        TestCompleteCallback on_complete = nullptr
    );

    /// Start test with specific model file
    bool StartTestFromCheckpoint(
        const std::string& checkpoint_dir,
        const std::string& dataset_name,
        TestProgressCallback on_progress = nullptr,
        TestCompleteCallback on_complete = nullptr
    );

    /// Stop running test
    void StopTest();

    /// Get current results
    const TestResults& GetResults() const;

    /// State queries
    bool IsTestRunning() const;
    bool HasResults() const;

    /// Export results
    bool ExportResultsToCSV(const std::string& filepath);
    bool ExportResultsToJSON(const std::string& filepath);
    bool ExportConfusionMatrixImage(const std::string& filepath);

private:
    TestManager() = default;
    std::unique_ptr<TestExecutor> executor_;
    TestResults last_results_;
    bool has_results_ = false;
};

} // namespace cyxwiz
```

### 1.4 UI Components

#### Test Results Panel

```cpp
// File: cyxwiz-engine/src/gui/panels/test_results_panel.h

class TestResultsPanel : public Panel {
public:
    void Render() override;

private:
    void RenderOverviewTab();      // Accuracy, loss, timing
    void RenderConfusionMatrix();  // Visual confusion matrix
    void RenderPerClassMetrics();  // Table of per-class precision/recall/F1
    void RenderMisclassified();    // Gallery of misclassified samples
    void RenderExportOptions();    // Export buttons

    int selected_tab_ = 0;
    bool show_percentages_ = true;  // Confusion matrix: counts vs percentages
    int selected_class_ = -1;       // For filtering misclassified
};
```

#### Toolbar Integration

```
Train Menu:
  ├─ Start Training (Ctrl+T)
  ├─ Pause Training
  ├─ Stop Training
  ├─ ─────────────────
  ├─ Run Test (Ctrl+Shift+T)      <── NEW
  ├─ Test from Checkpoint...       <── NEW
  └─ View Test Results             <── NEW
```

---

## Part 2: Weight Export System

### 2.1 Supported Export Formats

| Format | Extension | Use Case | Priority |
|--------|-----------|----------|----------|
| CyxWiz Native | `.cyx` | Internal use, full fidelity | P0 |
| ONNX | `.onnx` | Cross-framework, deployment | P0 |
| PyTorch | `.pth` / `.pt` | PyTorch ecosystem | P1 |
| TensorFlow SavedModel | `saved_model/` | TF Serving, TFLite | P1 |
| Keras H5 | `.h5` | Legacy Keras | P2 |
| NumPy Archive | `.npz` | Research, debugging | P2 |

### 2.2 Export Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Model Export Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌─────────────┐                                                │
│   │ Trained     │                                                │
│   │ Sequential  │                                                │
│   │ Model       │                                                │
│   └──────┬──────┘                                                │
│          │                                                        │
│          ▼                                                        │
│   ┌──────────────┐                                               │
│   │ ModelExporter│                                               │
│   │              │                                               │
│   │ GetParameters│───┬──▶ ONNXExporter ──▶ model.onnx           │
│   │ GetArchitect │   │                                           │
│   │              │   ├──▶ PyTorchExporter ──▶ model.pth         │
│   └──────────────┘   │                                           │
│                      ├──▶ TFExporter ──▶ saved_model/           │
│                      │                                           │
│                      ├──▶ KerasExporter ──▶ model.h5            │
│                      │                                           │
│                      └──▶ NumpyExporter ──▶ weights.npz         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Core Data Structures

```cpp
// File: cyxwiz-engine/src/core/model_exporter.h

#pragma once
#include <string>
#include <vector>
#include <map>
#include <functional>

namespace cyxwiz {

/// Export format enumeration
enum class ExportFormat {
    CyxWizNative,   // .cyx - Full model with metadata
    ONNX,           // .onnx - Open Neural Network Exchange
    PyTorch,        // .pth - PyTorch state dict
    TensorFlow,     // saved_model/ directory
    KerasH5,        // .h5 - Legacy Keras format
    NumpyArchive    // .npz - Raw numpy arrays
};

/// Export configuration
struct ExportConfig {
    ExportFormat format = ExportFormat::CyxWizNative;
    std::string output_path;

    // ONNX specific
    int onnx_opset_version = 13;
    bool onnx_include_metadata = true;
    std::vector<size_t> dynamic_batch_axes;  // Axes with dynamic size

    // PyTorch specific
    bool pytorch_include_optimizer = false;
    bool pytorch_use_torchscript = false;  // Trace to TorchScript

    // TensorFlow specific
    bool tf_include_signatures = true;
    std::string tf_serving_tag = "serve";

    // General
    bool include_training_metadata = true;
    bool compress = false;  // gzip compression
    std::string description;
};

/// Export result
struct ExportResult {
    bool success = false;
    std::string output_path;
    std::string error_message;
    size_t file_size_bytes = 0;
    float export_time_seconds = 0.0f;

    // Validation info
    bool validation_passed = false;
    std::string validation_message;
};

/// Progress callback for async export
using ExportProgressCallback = std::function<void(float progress, const std::string& status)>;

/// Model export manager
class ModelExporter {
public:
    ModelExporter();
    ~ModelExporter();

    /// Export trained model
    ExportResult Export(
        std::shared_ptr<SequentialModel> model,
        const ExportConfig& config,
        ExportProgressCallback on_progress = nullptr
    );

    /// Export from checkpoint
    ExportResult ExportFromCheckpoint(
        const std::string& checkpoint_path,
        const ExportConfig& config,
        ExportProgressCallback on_progress = nullptr
    );

    /// Validate exported model (load and compare outputs)
    bool ValidateExport(
        const std::string& export_path,
        std::shared_ptr<SequentialModel> original_model,
        const Tensor& sample_input
    );

    /// Get supported formats
    static std::vector<ExportFormat> GetSupportedFormats();
    static std::string GetFormatExtension(ExportFormat format);
    static std::string GetFormatDescription(ExportFormat format);

private:
    // Format-specific exporters
    ExportResult ExportCyxWiz(std::shared_ptr<SequentialModel> model, const ExportConfig& config);
    ExportResult ExportONNX(std::shared_ptr<SequentialModel> model, const ExportConfig& config);
    ExportResult ExportPyTorch(std::shared_ptr<SequentialModel> model, const ExportConfig& config);
    ExportResult ExportTensorFlow(std::shared_ptr<SequentialModel> model, const ExportConfig& config);
    ExportResult ExportKerasH5(std::shared_ptr<SequentialModel> model, const ExportConfig& config);
    ExportResult ExportNumpy(std::shared_ptr<SequentialModel> model, const ExportConfig& config);
};

} // namespace cyxwiz
```

### 2.4 CyxWiz Native Format (.cyx)

```
model.cyx (ZIP archive containing):
├── manifest.json          # Version, format info
├── architecture.json      # Layer definitions, shapes
├── metadata.json          # Training info, metrics
├── weights/
│   ├── layer_0_linear.bin # Binary weight tensors
│   ├── layer_0_bias.bin
│   ├── layer_1_linear.bin
│   └── ...
├── optimizer/             # Optional
│   └── state.bin
└── sample_input.bin       # Optional: for validation
```

**manifest.json:**
```json
{
    "format_version": "1.0",
    "cyxwiz_version": "0.3.0",
    "created_at": "2025-12-06T10:30:00Z",
    "model_name": "MNIST_Classifier",
    "description": "Simple MLP for MNIST digit classification",
    "framework": "cyxwiz-backend",
    "includes_optimizer": true,
    "includes_sample_input": true
}
```

**architecture.json:**
```json
{
    "input_shape": [1, 28, 28],
    "output_shape": [10],
    "layers": [
        {
            "name": "flatten",
            "type": "Flatten",
            "input_shape": [1, 28, 28],
            "output_shape": [784]
        },
        {
            "name": "dense_0",
            "type": "Linear",
            "input_features": 784,
            "output_features": 256,
            "has_bias": true,
            "activation": "relu"
        },
        {
            "name": "dense_1",
            "type": "Linear",
            "input_features": 256,
            "output_features": 10,
            "has_bias": true,
            "activation": "softmax"
        }
    ]
}
```

### 2.5 ONNX Export Implementation Notes

```cpp
// ONNX export requires building a graph representation
// Key dependencies: onnx.proto, protobuf

ExportResult ModelExporter::ExportONNX(
    std::shared_ptr<SequentialModel> model,
    const ExportConfig& config
) {
    // 1. Create ONNX ModelProto
    onnx::ModelProto model_proto;
    model_proto.set_ir_version(7);
    model_proto.set_producer_name("CyxWiz Engine");

    // 2. Create GraphProto
    auto* graph = model_proto.mutable_graph();
    graph->set_name("cyxwiz_model");

    // 3. Add input specification
    auto* input = graph->add_input();
    // ... shape info from model->GetInputShape()

    // 4. Traverse Sequential layers, add ONNX nodes
    std::string prev_output = "input";
    for (auto& module : model->GetModules()) {
        // Map CyxWiz layer types to ONNX operators
        // Linear → Gemm or MatMul+Add
        // ReLU → Relu
        // Softmax → Softmax
        // etc.
    }

    // 5. Add weights as initializers
    for (auto& [name, tensor] : model->GetParameters()) {
        auto* init = graph->add_initializer();
        init->set_name(name);
        // Copy tensor data to init->mutable_raw_data()
    }

    // 6. Write to file
    std::ofstream out(config.output_path, std::ios::binary);
    model_proto.SerializeToOstream(&out);
}
```

---

## Part 3: Weight Import System

### 3.1 Supported Import Formats

| Format | Detection | Notes |
|--------|-----------|-------|
| CyxWiz Native | `.cyx` extension | Full fidelity restore |
| CyxWiz Checkpoint | `metadata.json` in dir | Training state restore |
| ONNX | `.onnx` extension | Architecture + weights |
| PyTorch | `.pth`, `.pt` | State dict only |
| NumPy | `.npz` | Raw weights, needs arch |
| Keras H5 | `.h5` | Architecture + weights |

### 3.2 Import Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Model Import Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   External File                                                  │
│   ┌──────────┐                                                   │
│   │ .onnx    │───▶ ONNXImporter                                 │
│   │ .pth     │───▶ PyTorchImporter    ┌───────────────┐         │
│   │ .h5      │───▶ KerasImporter  ───▶│ ModelImporter │         │
│   │ .npz     │───▶ NumpyImporter      │               │         │
│   │ .cyx     │───▶ CyxWizImporter     │ Validate      │         │
│   └──────────┘                        │ BuildModel    │         │
│                                       │ LoadWeights   │         │
│                                       └───────┬───────┘         │
│                                               │                  │
│                                               ▼                  │
│                                       ┌───────────────┐         │
│                                       │ SequentialMod │         │
│                                       │ (Ready to use)│         │
│                                       └───────────────┘         │
│                                               │                  │
│                                    ┌──────────┴──────────┐      │
│                                    ▼                     ▼      │
│                              ┌──────────┐         ┌──────────┐  │
│                              │ Continue │         │ Run      │  │
│                              │ Training │         │ Inference│  │
│                              └──────────┘         └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Core Data Structures

```cpp
// File: cyxwiz-engine/src/core/model_importer.h

#pragma once
#include <string>
#include <vector>
#include <memory>

namespace cyxwiz {

/// Import source type (auto-detected or specified)
enum class ImportFormat {
    Auto,           // Detect from extension/content
    CyxWizNative,   // .cyx
    CyxWizCheckpoint, // checkpoint directory
    ONNX,           // .onnx
    PyTorch,        // .pth/.pt
    KerasH5,        // .h5
    NumpyArchive    // .npz
};

/// Import options
struct ImportConfig {
    ImportFormat format = ImportFormat::Auto;
    std::string source_path;

    // Weight mapping (for formats that need architecture separately)
    bool use_existing_architecture = false;  // Map weights to current graph

    // Validation
    bool validate_on_import = true;
    std::vector<size_t> expected_input_shape;  // Optional verification

    // Fine-tuning options
    bool freeze_imported_layers = false;  // Lock weights for transfer learning
    std::vector<std::string> layers_to_freeze;  // Specific layers

    // Device
    bool load_to_gpu = true;  // If available
};

/// Import result with loaded model
struct ImportResult {
    bool success = false;
    std::string error_message;

    // Loaded model (null if failed)
    std::shared_ptr<SequentialModel> model;

    // Detected information
    ImportFormat detected_format;
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    std::vector<std::string> layer_names;
    size_t total_parameters = 0;

    // Metadata (if available)
    std::string original_framework;
    std::string creation_date;
    std::map<std::string, std::string> custom_metadata;
};

/// Model import manager
class ModelImporter {
public:
    ModelImporter();
    ~ModelImporter();

    /// Import model from file
    ImportResult Import(const ImportConfig& config);

    /// Quick check if file is valid model
    static bool CanImport(const std::string& path);

    /// Detect format from file
    static ImportFormat DetectFormat(const std::string& path);

    /// Get model info without full import
    static ImportResult PeekModelInfo(const std::string& path);

    /// Apply imported weights to existing model
    bool ApplyWeightsToModel(
        std::shared_ptr<SequentialModel> target_model,
        const std::string& weights_path,
        bool strict = true  // Fail if shapes don't match
    );

private:
    // Format-specific importers
    ImportResult ImportCyxWiz(const ImportConfig& config);
    ImportResult ImportCyxWizCheckpoint(const ImportConfig& config);
    ImportResult ImportONNX(const ImportConfig& config);
    ImportResult ImportPyTorch(const ImportConfig& config);
    ImportResult ImportKerasH5(const ImportConfig& config);
    ImportResult ImportNumpy(const ImportConfig& config);

    // Helper to build Sequential from architecture info
    std::shared_ptr<SequentialModel> BuildModelFromArchitecture(
        const std::vector<LayerConfig>& layers
    );
};

} // namespace cyxwiz
```

### 3.4 Transfer Learning Support

```cpp
// For loading pre-trained weights and fine-tuning

struct TransferLearningConfig {
    std::string pretrained_model_path;

    // Layers to replace (e.g., replace classifier head)
    struct LayerReplacement {
        std::string original_layer_name;
        LayerConfig new_layer;
    };
    std::vector<LayerReplacement> replacements;

    // Freezing
    enum class FreezeMode {
        None,           // Train all layers
        AllExceptNew,   // Freeze pretrained, train new
        FirstN,         // Freeze first N layers
        Custom          // Custom list
    };
    FreezeMode freeze_mode = FreezeMode::AllExceptNew;
    int freeze_first_n = 0;
    std::vector<std::string> custom_freeze_list;

    // Learning rate schedule for fine-tuning
    float pretrained_lr_multiplier = 0.1f;  // Lower LR for frozen layers
};

class TransferLearningBuilder {
public:
    /// Load pretrained and modify for new task
    std::shared_ptr<SequentialModel> Build(const TransferLearningConfig& config);

    /// Add new classification head
    void ReplaceClassifier(
        std::shared_ptr<SequentialModel> model,
        int new_num_classes
    );

    /// Freeze layers
    void FreezeLayers(
        std::shared_ptr<SequentialModel> model,
        const std::vector<std::string>& layer_names
    );
};
```

---

## Part 4: UI Integration

### 4.1 New Menu Items

```
File Menu:
  ├─ New Project
  ├─ Open Project
  ├─ Save Project
  ├─ ─────────────────
  ├─ Import Model...        <── NEW (Ctrl+Shift+I)
  ├─ Export Model...        <── NEW (Ctrl+Shift+E)
  ├─ Export Weights Only... <── NEW
  └─ ...

Train Menu:
  ├─ Start Training
  ├─ ...
  ├─ ─────────────────
  ├─ Run Test              <── NEW (Ctrl+Shift+T)
  ├─ Test from Checkpoint  <── NEW
  └─ View Test Results     <── NEW
```

### 4.2 New Panels

#### Export Dialog Panel
```
┌─────────────────────────────────────────────────────────────┐
│ Export Model                                           [X]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Model: MNIST_Classifier                                    │
│  Parameters: 203,530                                        │
│  Input Shape: [1, 28, 28]                                   │
│  Output Shape: [10]                                         │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  Export Format:                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ○ CyxWiz Native (.cyx)  - Full fidelity            │   │
│  │ ● ONNX (.onnx)          - Cross-platform           │   │
│  │ ○ PyTorch (.pth)        - PyTorch ecosystem        │   │
│  │ ○ TensorFlow SavedModel - TF Serving               │   │
│  │ ○ Keras H5 (.h5)        - Legacy Keras             │   │
│  │ ○ NumPy (.npz)          - Raw weights              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Output Path: [______________________] [Browse...]          │
│                                                             │
│  Options:                                                   │
│  [✓] Include training metadata                              │
│  [✓] Validate after export                                  │
│  [ ] Include optimizer state                                │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│                              [Cancel]  [Export]             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Import Dialog Panel
```
┌─────────────────────────────────────────────────────────────┐
│ Import Model                                           [X]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Source: [_________________________] [Browse...]            │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  Detected Format: ONNX                                      │
│                                                             │
│  Model Info:                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Input Shape:  [1, 3, 224, 224]                      │   │
│  │ Output Shape: [1000]                                 │   │
│  │ Parameters:   11,689,512                             │   │
│  │ Layers:       50                                     │   │
│  │ Framework:    PyTorch (exported)                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Import Options:                                            │
│  ○ Load as new model (replace current graph)                │
│  ● Apply weights to current architecture                    │
│  ○ Transfer learning (freeze + new head)                    │
│                                                             │
│  [✓] Validate on import                                     │
│  [ ] Load to GPU                                            │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│                              [Cancel]  [Import]             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Test Results Panel
```
┌─────────────────────────────────────────────────────────────┐
│ Test Results                                           [X]  │
├─────────────────────────────────────────────────────────────┤
│ [Overview] [Confusion Matrix] [Per-Class] [Misclassified]   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Test Set: MNIST (10,000 samples)                           │
│  Model: mnist_classifier_v1                                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              OVERALL METRICS                         │   │
│  │                                                      │   │
│  │    Accuracy:  98.42%     Test Loss: 0.0523          │   │
│  │    Precision: 98.38%     Recall:    98.41%          │   │
│  │    F1 Score:  98.39%     Samples/s: 12,450          │   │
│  │                                                      │   │
│  │    Total Time: 0.8 seconds                          │   │
│  │    Misclassified: 158 samples                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  [Export to CSV]  [Export to JSON]  [Save Report]           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Node Editor Integration

#### New Nodes

```
┌─────────────────────┐
│ Load Weights        │
├─────────────────────┤
│ Path: [...]         │
│ Format: Auto ▼      │
│ Strict: [✓]         │
├─────────────────────┤
│            Model ●──│
└─────────────────────┘

┌─────────────────────┐
│ Save Weights        │
├─────────────────────┤
│──● Model            │
│ Path: [...]         │
│ Format: CyxWiz ▼    │
├─────────────────────┤
│         Success ●──│
└─────────────────────┘

┌─────────────────────┐
│ Test Node           │
├─────────────────────┤
│──● Model            │
│──● Test Data        │
│ Batch Size: 32      │
├─────────────────────┤
│        Results ●──│
│       Accuracy ●──│
└─────────────────────┘
```

---

## Part 5: Implementation Plan

### Phase 1: Testing System (Priority: High)

| Task | Files | Complexity |
|------|-------|------------|
| TestExecutor class | `test_executor.h/cpp` | Medium |
| TestManager singleton | `test_manager.h/cpp` | Low |
| Confusion matrix impl | `test_executor.cpp` | Low |
| TestResultsPanel UI | `test_results_panel.h/cpp` | Medium |
| Menu integration | `toolbar_other_menus.cpp` | Low |
| CMake updates | `CMakeLists.txt` | Low |

### Phase 2: Export System (Priority: High)

| Task | Files | Complexity |
|------|-------|------------|
| ModelExporter class | `model_exporter.h/cpp` | Medium |
| CyxWiz native export | `model_exporter.cpp` | Medium |
| ONNX export | `model_exporter_onnx.cpp` | High |
| NumPy export | `model_exporter.cpp` | Low |
| ExportDialog UI | `export_dialog.h/cpp` | Medium |
| Menu integration | `toolbar_file_menu.cpp` | Low |

### Phase 3: Import System (Priority: Medium)

| Task | Files | Complexity |
|------|-------|------------|
| ModelImporter class | `model_importer.h/cpp` | Medium |
| CyxWiz import | `model_importer.cpp` | Medium |
| ONNX import | `model_importer_onnx.cpp` | High |
| ImportDialog UI | `import_dialog.h/cpp` | Medium |
| Weight mapping | `model_importer.cpp` | Medium |

### Phase 4: Transfer Learning (Priority: Low)

| Task | Files | Complexity |
|------|-------|------------|
| TransferLearningBuilder | `transfer_learning.h/cpp` | Medium |
| Layer freezing | `sequential.h` modification | Low |
| UI for freeze options | `import_dialog.cpp` | Low |

---

## Part 6: File Structure

```
cyxwiz-engine/src/
├── core/
│   ├── test_executor.h          # NEW
│   ├── test_executor.cpp        # NEW
│   ├── test_manager.h           # NEW
│   ├── test_manager.cpp         # NEW
│   ├── model_exporter.h         # NEW
│   ├── model_exporter.cpp       # NEW
│   ├── model_exporter_onnx.cpp  # NEW
│   ├── model_importer.h         # NEW
│   ├── model_importer.cpp       # NEW
│   ├── model_importer_onnx.cpp  # NEW
│   └── transfer_learning.h      # NEW
│
└── gui/panels/
    ├── test_results_panel.h     # NEW
    ├── test_results_panel.cpp   # NEW
    ├── export_dialog.h          # NEW
    ├── export_dialog.cpp        # NEW
    ├── import_dialog.h          # NEW
    └── import_dialog.cpp        # NEW
```

---

## Part 7: Dependencies

### Required Libraries

| Library | Purpose | Status |
|---------|---------|--------|
| protobuf | ONNX serialization | Already have (gRPC) |
| onnx | ONNX format support | Need to add |
| HDF5/HighFive | Keras H5 format | Optional (already have) |
| nlohmann_json | JSON metadata | Already have |
| zlib | Compression | May need to add |

### ONNX Integration

```cmake
# CMakeLists.txt addition
find_package(ONNX QUIET)
if(ONNX_FOUND)
    target_link_libraries(cyxwiz-engine PRIVATE onnx)
    target_compile_definitions(cyxwiz-engine PRIVATE CYXWIZ_HAS_ONNX)
endif()
```

---

## Summary

This design adds three major capabilities to complete the ML pipeline:

1. **Testing**: Run inference on test data, compute metrics (accuracy, precision, recall, F1, confusion matrix), visualize misclassified samples

2. **Export**: Save trained models to portable formats (CyxWiz native, ONNX, PyTorch, TensorFlow) for deployment or sharing

3. **Import**: Load pre-trained models from external sources for inference or fine-tuning (transfer learning)

All features integrate with existing infrastructure:
- Uses `SequentialModel` and `Tensor` from cyxwiz-backend
- Leverages `CheckpointManager` for weight serialization
- Extends Node Editor with new node types
- Adds new panels and dialogs using existing ImGui patterns
- Follows async task pattern from `AsyncTaskManager`

The implementation is phased, starting with testing (most immediately useful), then export (enables sharing), then import (enables using external models).


  Features:
  - Train > Run Test (F7) - Runs inference on test data split
  - Train > View Test Results - Opens the TestResultsPanel
  - TestResultsPanel with 4 tabs:
    - Overview: Accuracy, loss, throughput, macro/weighted F1
    - Confusion Matrix: Color-coded, normalizable, with tooltips
    - Per-Class Metrics: Precision, recall, F1, TP/FP/support per class
    - Predictions: Scrollable list with misclassification filter
  - Export to CSV/JSON

 1. Phase 1 - Testing system (most immediately useful)
  2. Phase 2 - Export to CyxWiz native + ONNX
  3. Phase 3 - Import from ONNX + CyxWiz
  4. Phase 4 - Transfer learning helpers

    1. Train > Run Test (F7) - Run inference on test data
  2. Train > View Test Results - Open the Test Results panel

  To test the full workflow:
  1. Load or create a dataset (MNIST works well)
  2. Build a neural network in the node editor
  3. Train the model
  4. Click Train > Run Test to evaluate on test data
  5. View results in the Test Results panel with tabs for Overview, Confusion Matrix, Per-Class metrics, and
  Predictions

   1. Training Model Preservation
    - After training completes, the model and optimizer are preserved in TrainingManager
    - Accessible via GetLastTrainedModel(), GetLastOptimizer(), GetLastMetrics()
  2. Export Dialog Integration
    - When opening export, automatically loads trained model data
    - Passes current graph JSON for inclusion in .cyxmodel files
  3. Import Dialog Integration
    - On successful import, automatically loads graph into Node Editor
    - Shows the imported graph in the Node Editor panel
  4. Graph String Operations
    - GetGraphJson() - Export current node graph to JSON string
    - LoadGraphFromString() - Import graph from JSON string

  What Works Now

  - Export Flow: Train → Export → Model data passed to export dialog
  - Import Flow: Import .cyxmodel → Graph automatically loaded into Node Editor
  - Round-trip: Export graph → Import → Graph restored in editor
