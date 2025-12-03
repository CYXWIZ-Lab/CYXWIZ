# CyxWiz Dataset Manager Architecture

> **Vision**: A seamless data pipeline system that integrates with Asset Browser for file discovery, Node Editor for model training, and P2P network for distributed data loading.

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Integration Architecture](#3-integration-architecture)
4. [Asset Browser Integration](#4-asset-browser-integration)
5. [Node Editor Integration](#5-node-editor-integration)
6. [Data Pipeline Architecture](#6-data-pipeline-architecture)
7. [Dataset Types & Loaders](#7-dataset-types--loaders)
8. [Data Augmentation System](#8-data-augmentation-system)
9. [Distributed Data Loading](#9-distributed-data-loading)
10. [UI/UX Design](#10-uiux-design)
11. [Implementation Roadmap](#11-implementation-roadmap)

---

## 1. Executive Summary

### 1.1 Goals

- **Unified Experience**: Asset Browser discovers data, Dataset Manager processes it
- **Node Integration**: Dataset nodes in visual editor feed directly to training
- **Streaming Support**: Handle datasets larger than memory
- **Distributed Loading**: P2P data sharding across compute nodes
- **Format Agnostic**: Support CSV, images, MNIST, CIFAR, HuggingFace, custom

### 1.2 Key Principles

| Principle | Description |
|-----------|-------------|
| **Lazy Loading** | Don't load until needed, stream when possible |
| **Integration First** | Components talk to each other, not standalone |
| **Preview Everywhere** | Show data samples in Asset Browser, Node Editor, Dataset Manager |
| **Non-Blocking** | All loading operations are async with progress |

---

## 2. Current State Analysis

### 2.1 What We Have

#### Asset Browser (`asset_browser.cpp/h`)
```cpp
// Recognizes dataset files
enum class AssetType {
    Dataset,    // .csv, .json, .parquet, .h5, .arrow, .txt
    // ...
};

// Can detect dataset types
AssetType DetermineAssetType(const std::string& path);
```
- File tree navigation
- Double-click callbacks
- Context menus
- Search/filter

#### Dataset Manager (`dataset_panel.cpp/h`)
```cpp
// Supported formats
enum class DatasetType { None, CSV, Images, MNIST, CIFAR10 };

// Data storage
std::vector<std::vector<float>> raw_samples_;
std::vector<int> raw_labels_;

// Split management
std::vector<int> train_indices_, val_indices_, test_indices_;
```
- CSV, MNIST, CIFAR-10 loading (working)
- Image folder loading (stub)
- Train/val/test splitting
- Class distribution stats
- Local training (simulated)
- P2P job submission

#### Node Editor (`node_editor.cpp/h`)
```cpp
enum class NodeType {
    Input, Output,  // Data flow endpoints
    // ... layers
};
```
- Input/Output nodes exist
- No connection to Dataset Manager yet

### 2.2 Current Problems

| Problem | Impact |
|---------|--------|
| No integration between Asset Browser and Dataset Manager | User must manually copy paths |
| Dataset Manager is standalone | Can't feed data to Node Editor |
| No streaming | Large datasets crash the app |
| No augmentation | Limited data preprocessing |
| Image preview is text-only | Poor UX for visual data |

---

## 3. Integration Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CyxWiz Data Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐         ┌──────────────────┐         ┌─────────────┐ │
│  │   Asset Browser  │────────▶│  Dataset Manager │────────▶│ Node Editor │ │
│  │                  │         │                  │         │             │ │
│  │  - File tree     │ double  │  - Load & parse  │  data   │  - Input    │ │
│  │  - Preview pane  │  click  │  - Split         │  feed   │    node     │ │
│  │  - Context menu  │────────▶│  - Augment       │────────▶│  - Batch    │ │
│  │                  │         │  - Stats         │         │    node     │ │
│  └──────────────────┘         └──────────────────┘         └─────────────┘ │
│           │                            │                          │        │
│           │                            │                          │        │
│           ▼                            ▼                          ▼        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         DataRegistry (Singleton)                      │  │
│  │                                                                       │  │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │  │
│  │   │ Dataset A   │  │ Dataset B   │  │ Dataset C   │  │    ...     │  │  │
│  │   │ (loaded)    │  │ (streaming) │  │ (remote)    │  │            │  │  │
│  │   └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │  │
│  │                                                                       │  │
│  │   Methods:                                                            │  │
│  │   - RegisterDataset(name, config) → DatasetHandle                    │  │
│  │   - GetDataset(name) → DatasetHandle                                 │  │
│  │   - ListDatasets() → vector<DatasetInfo>                             │  │
│  │   - UnloadDataset(name)                                              │  │
│  │   - GetBatch(handle, indices) → Tensor                               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                      │                                     │
│                                      ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         Training Pipeline                             │  │
│  │                                                                       │  │
│  │   DataRegistry ──▶ DataLoader ──▶ Augmentation ──▶ Model ──▶ Loss    │  │
│  │                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Core Components

| Component | Responsibility | Location |
|-----------|----------------|----------|
| **DataRegistry** | Singleton managing all loaded datasets | `core/data_registry.h` |
| **DatasetHandle** | Reference to a loaded dataset | `core/dataset_handle.h` |
| **DataLoader** | Batching, shuffling, iteration | `core/data_loader.h` |
| **Augmentor** | Transform pipeline | `core/augmentation.h` |
| **AssetBrowser** | File discovery, preview trigger | `gui/panels/asset_browser.h` |
| **DatasetPanel** | Load UI, stats, config | `gui/panels/dataset_panel.h` |
| **InputNode** | Node Editor data source | `gui/node_editor.h` |

---

## 4. Asset Browser Integration

### 4.1 Enhanced Dataset Detection

```cpp
// Extended asset type detection
AssetType AssetBrowserPanel::DetermineAssetType(const std::string& path) {
    auto ext = GetExtension(path);

    // Dataset files
    if (ext == ".csv" || ext == ".tsv") return AssetType::TabularDataset;
    if (ext == ".parquet" || ext == ".arrow") return AssetType::TabularDataset;
    if (ext == ".json" || ext == ".jsonl") return AssetType::JSONDataset;
    if (ext == ".h5" || ext == ".hdf5") return AssetType::HDF5Dataset;

    // Image datasets (folders with images)
    if (IsImageFolder(path)) return AssetType::ImageDataset;

    // Standard ML datasets
    if (IsMNISTFolder(path)) return AssetType::MNISTDataset;
    if (IsCIFARFolder(path)) return AssetType::CIFARDataset;
    if (IsImageNetFolder(path)) return AssetType::ImageNetDataset;

    // ...
}

// Folder detection helpers
bool AssetBrowserPanel::IsMNISTFolder(const std::string& path) {
    // Check for train-images-idx3-ubyte, train-labels-idx1-ubyte
    return fs::exists(path + "/train-images-idx3-ubyte") ||
           fs::exists(path + "/train-images.idx3-ubyte");
}
```

### 4.2 Dataset Preview Pane

When a dataset file/folder is selected in Asset Browser, show a preview:

```cpp
void AssetBrowserPanel::RenderDatasetPreview(const AssetItem& item) {
    ImGui::BeginChild("DatasetPreview", ImVec2(0, 200));

    auto& registry = DataRegistry::Instance();
    auto preview = registry.GetPreview(item.absolute_path, /*max_rows=*/5);

    if (preview.type == PreviewType::Tabular) {
        // Show table with first 5 rows
        if (ImGui::BeginTable("Preview", preview.columns.size())) {
            for (const auto& col : preview.columns) {
                ImGui::TableSetupColumn(col.c_str());
            }
            ImGui::TableHeadersRow();

            for (const auto& row : preview.rows) {
                ImGui::TableNextRow();
                for (const auto& cell : row) {
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", cell.c_str());
                }
            }
            ImGui::EndTable();
        }
    }
    else if (preview.type == PreviewType::Image) {
        // Show thumbnail grid
        for (int i = 0; i < preview.thumbnails.size(); i++) {
            if (i > 0 && i % 4 != 0) ImGui::SameLine();
            ImGui::Image(preview.thumbnails[i].texture_id, ImVec2(64, 64));
        }
    }

    // Quick stats
    ImGui::Separator();
    ImGui::Text("Samples: %d | Classes: %d | Size: %s",
        preview.num_samples, preview.num_classes,
        FormatSize(preview.file_size).c_str());

    // Action button
    if (ImGui::Button("Load in Dataset Manager")) {
        LoadDatasetInManager(item.absolute_path);
    }

    ImGui::EndChild();
}
```

### 4.3 Double-Click Integration

```cpp
// In MainWindow setup
asset_browser_->SetOnAssetDoubleClick([this](const AssetItem& item) {
    switch (item.type) {
        case AssetType::Script:
            script_editor_->OpenFile(item.absolute_path);
            break;

        case AssetType::TabularDataset:
        case AssetType::ImageDataset:
        case AssetType::MNISTDataset:
        case AssetType::CIFARDataset:
            // Open in Dataset Manager
            dataset_panel_->LoadDataset(item.absolute_path);
            // Switch to Dataset Manager tab
            FocusPanel("Dataset Manager");
            break;

        case AssetType::Model:
            // Load model in Node Editor
            node_editor_->LoadModel(item.absolute_path);
            FocusPanel("Node Editor");
            break;
    }
});
```

### 4.4 Context Menu Extensions

```cpp
void AssetBrowserPanel::RenderDatasetContextMenu(const AssetItem& item) {
    if (ImGui::BeginPopupContextItem()) {
        if (ImGui::MenuItem(ICON_FA_DATABASE " Load Dataset")) {
            dataset_panel_->LoadDataset(item.absolute_path);
        }
        if (ImGui::MenuItem(ICON_FA_EYE " Preview Data")) {
            show_preview_popup_ = true;
            preview_item_ = &item;
        }
        if (ImGui::MenuItem(ICON_FA_CHART_BAR " Show Statistics")) {
            ShowDatasetStats(item.absolute_path);
        }

        ImGui::Separator();

        if (ImGui::BeginMenu(ICON_FA_DIAGRAM_PROJECT " Add to Node Editor")) {
            if (ImGui::MenuItem("As Input Node")) {
                node_editor_->CreateDatasetInputNode(item.absolute_path);
            }
            if (ImGui::MenuItem("As DataLoader Node")) {
                node_editor_->CreateDataLoaderNode(item.absolute_path);
            }
            ImGui::EndMenu();
        }

        ImGui::EndPopup();
    }
}
```

---

## 5. Node Editor Integration

### 5.1 Dataset Nodes

Add new node types for data handling:

```cpp
enum class NodeLabel {
    // ... existing nodes ...

    // Data Nodes
    DatasetInput,       // Load from DataRegistry
    DataLoader,         // Batch iterator
    Augmentation,       // Transform pipeline
    DataSplit,          // Train/val/test splitter

    // Utility
    TensorReshape,      // Reshape data
    Normalize,          // Normalize values
    OneHotEncode,       // Label encoding
};
```

### 5.2 DatasetInput Node

```cpp
struct DatasetInputNode : public Node {
    // Properties
    std::string dataset_name;      // Reference to DataRegistry
    std::string split = "train";   // train, val, test

    // Outputs
    Pin data_output;    // Tensor output
    Pin label_output;   // Labels output
    Pin shape_output;   // Shape info

    // Runtime
    DatasetHandle handle_;
};

// Node rendering
void NodeEditor::RenderDatasetInputNode(DatasetInputNode& node) {
    ImNodes::BeginNode(node.id);

    ImNodes::BeginNodeTitleBar();
    ImGui::Text(ICON_FA_DATABASE " Dataset: %s", node.dataset_name.c_str());
    ImNodes::EndNodeTitleBar();

    // Dataset selector dropdown
    auto& registry = DataRegistry::Instance();
    auto datasets = registry.ListDatasets();

    if (ImGui::BeginCombo("##dataset", node.dataset_name.c_str())) {
        for (const auto& ds : datasets) {
            if (ImGui::Selectable(ds.name.c_str(), ds.name == node.dataset_name)) {
                node.dataset_name = ds.name;
                node.handle_ = registry.GetDataset(ds.name);
            }
        }
        ImGui::EndCombo();
    }

    // Split selector
    const char* splits[] = {"train", "val", "test"};
    ImGui::Combo("Split", &node.split_index, splits, 3);

    // Show shape info
    if (node.handle_.IsValid()) {
        auto info = node.handle_.GetInfo();
        ImGui::TextDisabled("Shape: %s", FormatShape(info.shape).c_str());
        ImGui::TextDisabled("Samples: %d", info.num_samples);
    }

    // Output pins
    ImNodes::BeginOutputAttribute(node.data_output.id);
    ImGui::Text("Data");
    ImNodes::EndOutputAttribute();

    ImNodes::BeginOutputAttribute(node.label_output.id);
    ImGui::Text("Labels");
    ImNodes::EndOutputAttribute();

    ImNodes::EndNode();
}
```

### 5.3 DataLoader Node

```cpp
struct DataLoaderNode : public Node {
    // Properties
    int batch_size = 32;
    bool shuffle = true;
    bool drop_last = false;
    int num_workers = 4;

    // Inputs
    Pin dataset_input;   // From DatasetInput

    // Outputs
    Pin batch_output;    // Batched tensor
    Pin labels_output;   // Batched labels
    Pin epoch_output;    // Epoch signal
};
```

### 5.4 Augmentation Node

```cpp
struct AugmentationNode : public Node {
    // Configurable transforms
    std::vector<Transform> transforms;

    // Inputs
    Pin data_input;

    // Outputs
    Pin data_output;
};

// Available transforms
enum class TransformType {
    // Geometric
    RandomCrop,
    RandomFlip,
    RandomRotation,
    RandomAffine,
    Resize,
    CenterCrop,

    // Color
    ColorJitter,
    RandomGrayscale,
    Normalize,

    // Noise
    GaussianNoise,
    GaussianBlur,

    // Advanced
    Cutout,
    Mixup,
    CutMix,
    RandAugment,
    AutoAugment
};
```

### 5.5 Complete Training Graph Example

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│ DatasetInput    │────▶│ DataLoader   │────▶│ Augmentation│
│ (MNIST)         │     │ batch=32     │     │ RandomFlip  │
│ split=train     │     │ shuffle=true │     │ Normalize   │
└─────────────────┘     └──────────────┘     └──────┬──────┘
                                                     │
                        ┌────────────────────────────┘
                        ▼
              ┌─────────────────┐     ┌──────────┐     ┌──────────┐
              │ Conv2D(32)      │────▶│ ReLU     │────▶│ MaxPool  │
              └─────────────────┘     └──────────┘     └────┬─────┘
                                                            │
                        ┌───────────────────────────────────┘
                        ▼
              ┌─────────────────┐     ┌──────────┐     ┌──────────┐
              │ Conv2D(64)      │────▶│ ReLU     │────▶│ Flatten  │
              └─────────────────┘     └──────────┘     └────┬─────┘
                                                            │
                        ┌───────────────────────────────────┘
                        ▼
              ┌─────────────────┐     ┌──────────┐     ┌───────────────┐
              │ Dense(128)      │────▶│ ReLU     │────▶│ Dense(10)     │
              └─────────────────┘     └──────────┘     └───────┬───────┘
                                                               │
                        ┌──────────────────────────────────────┘
                        ▼
              ┌─────────────────┐     ┌──────────────────┐
              │ Softmax         │────▶│ CrossEntropyLoss │◀── Labels
              └─────────────────┘     └────────┬─────────┘
                                               │
                                               ▼
                                      ┌──────────────┐
                                      │ Adam         │
                                      │ lr=0.001     │
                                      └──────────────┘
```

---

## 6. Data Pipeline Architecture

### 6.1 DataRegistry Singleton

```cpp
namespace cyxwiz {

// Dataset information
struct DatasetInfo {
    std::string name;
    std::string path;
    DatasetType type;
    std::vector<size_t> shape;
    size_t num_samples;
    size_t num_classes;
    std::vector<std::string> class_names;

    // Split info
    size_t train_count;
    size_t val_count;
    size_t test_count;

    // Memory info
    size_t memory_usage;
    bool is_streaming;
};

// Handle to a loaded dataset
class DatasetHandle {
public:
    bool IsValid() const;
    DatasetInfo GetInfo() const;

    // Data access
    Tensor GetSample(size_t index) const;
    int GetLabel(size_t index) const;
    std::pair<Tensor, Tensor> GetBatch(const std::vector<size_t>& indices) const;

    // Split access
    std::vector<size_t> GetTrainIndices() const;
    std::vector<size_t> GetValIndices() const;
    std::vector<size_t> GetTestIndices() const;

private:
    friend class DataRegistry;
    std::shared_ptr<Dataset> dataset_;
};

// Central registry for all datasets
class DataRegistry {
public:
    static DataRegistry& Instance();

    // Registration
    DatasetHandle LoadDataset(const std::string& path, const std::string& name = "");
    DatasetHandle LoadDataset(const DatasetConfig& config);
    void UnloadDataset(const std::string& name);

    // Access
    DatasetHandle GetDataset(const std::string& name);
    std::vector<DatasetInfo> ListDatasets() const;
    bool HasDataset(const std::string& name) const;

    // Preview (lightweight, doesn't fully load)
    DatasetPreview GetPreview(const std::string& path, int max_samples = 5);

    // Memory management
    size_t GetTotalMemoryUsage() const;
    void SetMemoryLimit(size_t bytes);
    void EvictLRU();  // Evict least recently used

private:
    DataRegistry() = default;
    std::map<std::string, std::shared_ptr<Dataset>> datasets_;
    size_t memory_limit_ = 4ULL * 1024 * 1024 * 1024;  // 4GB default
    mutable std::mutex mutex_;
};

} // namespace cyxwiz
```

### 6.2 DataLoader Class

```cpp
namespace cyxwiz {

struct DataLoaderConfig {
    int batch_size = 32;
    bool shuffle = true;
    bool drop_last = false;
    int num_workers = 4;
    bool pin_memory = true;
    size_t prefetch_factor = 2;
};

class DataLoader {
public:
    DataLoader(DatasetHandle dataset, const DataLoaderConfig& config);

    // Iteration
    class Iterator {
    public:
        std::pair<Tensor, Tensor> operator*() const;
        Iterator& operator++();
        bool operator!=(const Iterator& other) const;
    };

    Iterator begin();
    Iterator end();

    // Info
    size_t NumBatches() const;
    size_t BatchSize() const;

    // Control
    void SetEpoch(int epoch);  // For deterministic shuffling

private:
    DatasetHandle dataset_;
    DataLoaderConfig config_;
    std::vector<size_t> indices_;

    // Prefetch
    std::queue<std::future<std::pair<Tensor, Tensor>>> prefetch_queue_;
    ThreadPool worker_pool_;
};

// Usage example
void TrainEpoch(DataLoader& loader, Model& model) {
    for (auto [data, labels] : loader) {
        auto output = model.Forward(data);
        auto loss = criterion(output, labels);
        loss.Backward();
        optimizer.Step();
    }
}

} // namespace cyxwiz
```

### 6.3 Transform Pipeline

```cpp
namespace cyxwiz {

// Base transform
class Transform {
public:
    virtual ~Transform() = default;
    virtual Tensor Apply(const Tensor& input) = 0;
    virtual std::string Name() const = 0;
};

// Compose multiple transforms
class Compose : public Transform {
public:
    Compose(std::vector<std::unique_ptr<Transform>> transforms);
    Tensor Apply(const Tensor& input) override;

private:
    std::vector<std::unique_ptr<Transform>> transforms_;
};

// Common transforms
class RandomHorizontalFlip : public Transform {
public:
    RandomHorizontalFlip(float p = 0.5f);
    Tensor Apply(const Tensor& input) override;
};

class RandomCrop : public Transform {
public:
    RandomCrop(std::vector<int> size, std::vector<int> padding = {});
    Tensor Apply(const Tensor& input) override;
};

class Normalize : public Transform {
public:
    Normalize(std::vector<float> mean, std::vector<float> std);
    Tensor Apply(const Tensor& input) override;
};

class ColorJitter : public Transform {
public:
    ColorJitter(float brightness, float contrast, float saturation, float hue);
    Tensor Apply(const Tensor& input) override;
};

// Factory
std::unique_ptr<Transform> CreateTransform(const std::string& name,
                                            const std::map<std::string, PropertyValue>& params);

} // namespace cyxwiz
```

---

## 7. Dataset Types & Loaders

### 7.1 Supported Formats

```cpp
enum class DatasetType {
    // Tabular
    CSV,
    TSV,
    Parquet,
    Arrow,
    Excel,

    // Image
    ImageFolder,        // folder/class_name/image.jpg
    ImageFile,          // Single image file
    COCO,               // COCO format annotations
    VOC,                // Pascal VOC format

    // Standard ML
    MNIST,
    FashionMNIST,
    CIFAR10,
    CIFAR100,
    ImageNet,

    // Text
    TextFile,
    JSONL,

    // Audio
    AudioFolder,
    LibriSpeech,

    // HuggingFace
    HuggingFace,

    // Custom
    Custom
};
```

### 7.2 Loader Implementations

```cpp
// Base dataset interface
class Dataset {
public:
    virtual ~Dataset() = default;

    virtual size_t Size() const = 0;
    virtual std::pair<Tensor, int> GetItem(size_t index) const = 0;
    virtual DatasetInfo GetInfo() const = 0;

    // Optional: streaming support
    virtual bool SupportsStreaming() const { return false; }
    virtual std::unique_ptr<Iterator> GetStreamIterator() { return nullptr; }
};

// CSV Dataset
class CSVDataset : public Dataset {
public:
    CSVDataset(const std::string& path, const CSVConfig& config);

    size_t Size() const override;
    std::pair<Tensor, int> GetItem(size_t index) const override;

private:
    std::vector<std::vector<float>> data_;
    std::vector<int> labels_;
    std::vector<std::string> columns_;
};

// Image Folder Dataset
class ImageFolderDataset : public Dataset {
public:
    ImageFolderDataset(const std::string& root, const ImageConfig& config);

    size_t Size() const override;
    std::pair<Tensor, int> GetItem(size_t index) const override;

private:
    std::vector<std::string> image_paths_;
    std::vector<int> labels_;
    std::vector<std::string> class_names_;
    std::unique_ptr<Transform> transform_;
};

// MNIST Dataset
class MNISTDataset : public Dataset {
public:
    MNISTDataset(const std::string& root, bool train = true);

    size_t Size() const override;
    std::pair<Tensor, int> GetItem(size_t index) const override;

private:
    std::vector<std::vector<uint8_t>> images_;
    std::vector<uint8_t> labels_;
};

// HuggingFace Dataset (remote)
class HuggingFaceDataset : public Dataset {
public:
    HuggingFaceDataset(const std::string& dataset_name,
                       const std::string& split = "train",
                       const std::string& config = "default");

    size_t Size() const override;
    std::pair<Tensor, int> GetItem(size_t index) const override;

    bool SupportsStreaming() const override { return true; }

private:
    std::string dataset_name_;
    std::string api_endpoint_;
    std::unique_ptr<HTTPClient> client_;
    mutable LRUCache<size_t, std::pair<Tensor, int>> cache_;
};
```

### 7.3 Format Detection

```cpp
DatasetType DetectDatasetType(const std::string& path) {
    namespace fs = std::filesystem;

    if (fs::is_directory(path)) {
        // Check for known structures
        if (fs::exists(path + "/train-images-idx3-ubyte")) return DatasetType::MNIST;
        if (fs::exists(path + "/data_batch_1.bin")) return DatasetType::CIFAR10;
        if (fs::exists(path + "/annotations/instances_train.json")) return DatasetType::COCO;

        // Check for image folder structure (class subfolders)
        bool has_image_subfolders = false;
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.is_directory()) {
                for (const auto& sub : fs::directory_iterator(entry)) {
                    if (IsImageFile(sub.path())) {
                        has_image_subfolders = true;
                        break;
                    }
                }
            }
            if (has_image_subfolders) break;
        }
        if (has_image_subfolders) return DatasetType::ImageFolder;
    }
    else {
        auto ext = fs::path(path).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".csv") return DatasetType::CSV;
        if (ext == ".tsv") return DatasetType::TSV;
        if (ext == ".parquet") return DatasetType::Parquet;
        if (ext == ".arrow") return DatasetType::Arrow;
        if (ext == ".jsonl" || ext == ".json") return DatasetType::JSONL;
        if (ext == ".txt") return DatasetType::TextFile;
    }

    return DatasetType::Custom;
}
```

---

## 8. Data Augmentation System

### 8.1 Transform Categories

```
Transforms
├── Geometric
│   ├── RandomCrop(size, padding)
│   ├── CenterCrop(size)
│   ├── Resize(size, interpolation)
│   ├── RandomHorizontalFlip(p)
│   ├── RandomVerticalFlip(p)
│   ├── RandomRotation(degrees)
│   ├── RandomAffine(degrees, translate, scale, shear)
│   ├── RandomPerspective(distortion, p)
│   └── RandomResizedCrop(size, scale, ratio)
│
├── Color
│   ├── ColorJitter(brightness, contrast, saturation, hue)
│   ├── RandomGrayscale(p)
│   ├── Normalize(mean, std)
│   ├── RandomInvert(p)
│   ├── RandomPosterize(bits, p)
│   ├── RandomSolarize(threshold, p)
│   └── RandomAdjustSharpness(factor, p)
│
├── Noise & Blur
│   ├── GaussianNoise(mean, std)
│   ├── GaussianBlur(kernel_size, sigma)
│   ├── RandomErasing(p, scale, ratio)
│   └── Cutout(n_holes, length)
│
├── Advanced
│   ├── Mixup(alpha)
│   ├── CutMix(alpha)
│   ├── RandAugment(n, m)
│   ├── AutoAugment(policy)
│   └── TrivialAugmentWide()
│
└── Utility
    ├── ToTensor()
    ├── ToPILImage()
    ├── Lambda(func)
    └── RandomChoice(transforms)
```

### 8.2 Augmentation Node UI

```cpp
void NodeEditor::RenderAugmentationNodeConfig(AugmentationNode& node) {
    ImGui::Text("Transform Pipeline");
    ImGui::Separator();

    // List current transforms with drag-reorder
    for (int i = 0; i < node.transforms.size(); i++) {
        auto& t = node.transforms[i];

        ImGui::PushID(i);

        // Drag handle
        ImGui::Button(ICON_FA_GRIP_VERTICAL);
        if (ImGui::BeginDragDropSource()) {
            ImGui::SetDragDropPayload("TRANSFORM_REORDER", &i, sizeof(int));
            ImGui::Text("Move %s", t.name.c_str());
            ImGui::EndDragDropSource();
        }

        ImGui::SameLine();

        // Transform name and toggle
        bool enabled = t.enabled;
        if (ImGui::Checkbox(t.name.c_str(), &enabled)) {
            t.enabled = enabled;
        }

        // Expand/collapse params
        ImGui::SameLine();
        if (ImGui::TreeNode("##params")) {
            RenderTransformParams(t);
            ImGui::TreePop();
        }

        // Remove button
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_TRASH)) {
            node.transforms.erase(node.transforms.begin() + i);
        }

        ImGui::PopID();
    }

    // Add new transform
    ImGui::Separator();
    if (ImGui::Button(ICON_FA_PLUS " Add Transform")) {
        ImGui::OpenPopup("AddTransform");
    }

    if (ImGui::BeginPopup("AddTransform")) {
        if (ImGui::BeginMenu("Geometric")) {
            if (ImGui::MenuItem("Random Crop")) AddTransform(node, "RandomCrop");
            if (ImGui::MenuItem("Random Flip")) AddTransform(node, "RandomHorizontalFlip");
            if (ImGui::MenuItem("Random Rotation")) AddTransform(node, "RandomRotation");
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Color")) {
            if (ImGui::MenuItem("Color Jitter")) AddTransform(node, "ColorJitter");
            if (ImGui::MenuItem("Normalize")) AddTransform(node, "Normalize");
            ImGui::EndMenu();
        }
        // ... more categories
        ImGui::EndPopup();
    }
}
```

---

## 9. Distributed Data Loading

### 9.1 P2P Data Distribution

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Distributed Training Setup                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐                                                       │
│   │   Engine    │  (Coordinator)                                        │
│   │             │                                                       │
│   │  Dataset:   │  Full dataset path: /data/imagenet/                   │
│   │  ImageNet   │                                                       │
│   └──────┬──────┘                                                       │
│          │                                                              │
│          │  Job submission with dataset_uri                             │
│          ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      Central Server                              │   │
│   │                                                                  │   │
│   │   1. Parse dataset_uri → determine sharding strategy            │   │
│   │   2. Assign shards to nodes: Node A gets shard 0-3              │   │
│   │                               Node B gets shard 4-7              │   │
│   │   3. Coordinate data transfer                                   │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│          │                                    │                         │
│          ▼                                    ▼                         │
│   ┌─────────────┐                      ┌─────────────┐                  │
│   │  Server     │                      │  Server     │                  │
│   │  Node A     │                      │  Node B     │                  │
│   │             │                      │             │                  │
│   │  Shard 0-3  │  ◀─── P2P Sync ───▶  │  Shard 4-7  │                  │
│   │  (25% data) │                      │  (25% data) │                  │
│   │             │                      │             │                  │
│   │  Gradient   │  ◀─── AllReduce ──▶  │  Gradient   │                  │
│   │  Sync       │                      │  Sync       │                  │
│   └─────────────┘                      └─────────────┘                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Dataset URI Format

```
// Dataset URI format for P2P training
dataset_uri := scheme "://" location ["?" options]

// Schemes:
file://     - Local filesystem
ipfs://     - IPFS content hash
http://     - Remote HTTP(S)
s3://       - AWS S3
hf://       - HuggingFace datasets

// Examples:
file:///data/mnist                           # Local MNIST
file://./project/data/train.csv              # Relative path
ipfs://QmXy123.../imagenet                   # IPFS stored dataset
hf://mnist?split=train                       # HuggingFace
s3://bucket/datasets/cifar10.tar.gz          # S3 bucket

// Options:
?split=train                                 # Data split
?shard=0&num_shards=4                       # Sharding
?cache=true                                 # Enable caching
?streaming=true                             # Stream mode
```

### 9.3 Sharding Strategy

```cpp
enum class ShardingStrategy {
    Random,         // Random assignment (default)
    Sequential,     // Sequential chunks
    ByClass,        // Each node gets complete classes
    Balanced        // Balance classes across nodes
};

struct ShardConfig {
    ShardingStrategy strategy = ShardingStrategy::Random;
    int num_shards = 0;     // 0 = auto (number of nodes)
    int shard_id = -1;      // -1 = auto assign
    int seed = 42;          // For reproducibility
};

class ShardedDataset : public Dataset {
public:
    ShardedDataset(std::shared_ptr<Dataset> base, const ShardConfig& config);

    size_t Size() const override {
        return shard_indices_.size();
    }

    std::pair<Tensor, int> GetItem(size_t index) const override {
        return base_->GetItem(shard_indices_[index]);
    }

private:
    std::shared_ptr<Dataset> base_;
    std::vector<size_t> shard_indices_;
};
```

---

## 10. UI/UX Design

### 10.1 Dataset Manager Redesign

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Dataset Manager                                                    [─][□][×]│
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─ Loaded Datasets ─────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  [▼] MNIST (train)        60,000 samples   28×28×1    [Unload] [→]   │  │
│  │  [▼] CIFAR-10 (train)     50,000 samples   32×32×3    [Unload] [→]   │  │
│  │  [ ] Custom CSV           1,234 samples    features:8  [Unload] [→]   │  │
│  │                                                                       │  │
│  │  [+ Load Dataset]  [+ From HuggingFace]  [+ From URL]                │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─ Selected: MNIST ─────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  Statistics                          │  Preview                       │  │
│  │  ─────────────────────────────────── │  ────────────────────────────  │  │
│  │  Type: MNIST                         │  Sample 1/60000  [◀] [▶]      │  │
│  │  Samples: 60,000                     │                                │  │
│  │  Classes: 10                         │  ┌────────────────────────┐   │  │
│  │  Shape: [28, 28, 1]                  │  │                        │   │  │
│  │  Memory: 45.6 MB                     │  │    ████████████        │   │  │
│  │                                      │  │    ██          ██      │   │  │
│  │  Class Distribution:                 │  │    ██          ██      │   │  │
│  │  0: ████████████ 5,923 (9.9%)       │  │    ██████████████      │   │  │
│  │  1: █████████████ 6,742 (11.2%)     │  │    ██          ██      │   │  │
│  │  2: ████████████ 5,958 (9.9%)       │  │    ██          ██      │   │  │
│  │  3: ████████████ 6,131 (10.2%)      │  │    ████████████        │   │  │
│  │  4: ███████████ 5,842 (9.7%)        │  │                        │   │  │
│  │  5: ██████████ 5,421 (9.0%)         │  └────────────────────────┘   │  │
│  │  6: ████████████ 5,918 (9.9%)       │  Label: 0                     │  │
│  │  7: ████████████ 6,265 (10.4%)      │                                │  │
│  │  8: ███████████ 5,851 (9.8%)        │  [ ] Show augmented           │  │
│  │  9: ███████████ 5,949 (9.9%)        │                                │  │
│  │                                      │                                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─ Data Splits ─────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  Train: [============================] 80%  48,000                    │  │
│  │  Val:   [===                         ] 10%   6,000                    │  │
│  │  Test:  [===                         ] 10%   6,000                    │  │
│  │                                                                       │  │
│  │  [Apply Split]  [ ] Stratified  [ ] Shuffle  Seed: [42    ]          │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─ Quick Actions ───────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  [Add to Node Editor]  [Export Splits]  [Show in Asset Browser]      │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Asset Browser Dataset Preview

```
┌─────────────────────────────────────────────────────────────────┐
│ Asset Browser                                          [─][□][×]│
├─────────────────────────────────────────────────────────────────┤
│ 🔍 [Search...                    ] [Filter: All ▼]              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📁 my_project/                                                 │
│  ├── 📁 data/                                                   │
│  │   ├── 📊 train.csv                    ← [Selected]          │
│  │   ├── 📊 test.csv                                           │
│  │   └── 📁 images/                                            │
│  │       ├── 📁 cats/                                          │
│  │       └── 📁 dogs/                                          │
│  ├── 📁 models/                                                 │
│  └── 📁 scripts/                                                │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Preview: train.csv                                              │
│ ─────────────────────────────────────────────────────────────── │
│ ┌─────────┬─────────┬─────────┬─────────┬─────────┐            │
│ │ feature1│ feature2│ feature3│ feature4│ label   │            │
│ ├─────────┼─────────┼─────────┼─────────┼─────────┤            │
│ │ 0.234   │ 1.567   │ -0.891  │ 0.123   │ 0       │            │
│ │ -0.456  │ 2.345   │ 0.678   │ -0.234  │ 1       │            │
│ │ 0.789   │ -1.234  │ 0.456   │ 0.567   │ 0       │            │
│ └─────────┴─────────┴─────────┴─────────┴─────────┘            │
│                                                                 │
│ Rows: 10,000 | Columns: 5 | Size: 1.2 MB                       │
│                                                                 │
│ [Load in Dataset Manager]  [Quick Stats]                        │
└─────────────────────────────────────────────────────────────────┘
```

### 10.3 Node Editor Dataset Node

```
┌────────────────────────────────────────┐
│  📊 Dataset Input                      │
├────────────────────────────────────────┤
│                                        │
│  Dataset: [MNIST            ▼]         │
│  Split:   [train ▼]                    │
│                                        │
│  Shape: [28, 28, 1]                    │
│  Samples: 48,000                       │
│                                        │
│  [Preview]  [Refresh]                  │
│                                        │
├────────────────────────────────────────┤
│                           ○ Data       │
│                           ○ Labels     │
│                           ○ Shape      │
└────────────────────────────────────────┘
```

---

## 11. Implementation Roadmap

### Phase 1: Integration Foundation (1-2 weeks)

```
Tasks:
├── [ ] Create DataRegistry singleton
├── [ ] Create DatasetHandle class
├── [ ] Refactor DatasetPanel to use DataRegistry
├── [ ] Add double-click handler in AssetBrowser
├── [ ] Add dataset preview pane to AssetBrowser
├── [ ] Add "Load in Dataset Manager" button
└── [ ] Wire up callbacks between components
```

### Phase 2: Node Editor Integration (2-3 weeks)

```
Tasks:
├── [ ] Add DatasetInput node type
├── [ ] Add DataLoader node type
├── [ ] Add Augmentation node type
├── [ ] Implement node property panels
├── [ ] Wire dataset nodes to training pipeline
├── [ ] Add visual feedback for data flow
└── [ ] Test end-to-end training with nodes
```

### Phase 3: Enhanced Data Loading (2 weeks)

```
Tasks:
├── [ ] Add image preview with OpenGL textures
├── [ ] Implement ImageFolder dataset
├── [ ] Implement HuggingFace dataset loader
├── [ ] Add streaming support for large datasets
├── [ ] Implement data caching (LRU)
└── [ ] Add memory usage monitoring
```

### Phase 4: Augmentation System (2 weeks)

```
Tasks:
├── [ ] Implement geometric transforms
├── [ ] Implement color transforms
├── [ ] Implement noise/blur transforms
├── [ ] Create transform composer
├── [ ] Add augmentation preview in UI
└── [ ] Implement Mixup/CutMix
```

### Phase 5: Distributed Data (2-3 weeks)

```
Tasks:
├── [ ] Define dataset_uri protocol
├── [ ] Implement sharding strategies
├── [ ] Add IPFS dataset support
├── [ ] Implement P2P data transfer
├── [ ] Add data prefetching
└── [ ] Test distributed training
```

### Phase 6: Polish & Performance (1-2 weeks)

```
Tasks:
├── [ ] Optimize memory usage
├── [ ] Add progress bars everywhere
├── [ ] Implement lazy loading
├── [ ] Add dataset versioning
├── [ ] Export/import dataset configs
└── [ ] Documentation and examples
```

---

## Appendix A: File Structure

```
cyxwiz-engine/src/
├── core/
│   ├── data_registry.h          # Singleton registry
│   ├── data_registry.cpp
│   ├── dataset_handle.h         # Handle to loaded dataset
│   ├── data_loader.h            # Batching iterator
│   ├── data_loader.cpp
│   └── transform.h              # Augmentation base
│
├── datasets/
│   ├── dataset.h                # Base dataset interface
│   ├── csv_dataset.cpp          # CSV loader
│   ├── image_folder_dataset.cpp # Image folder loader
│   ├── mnist_dataset.cpp        # MNIST loader
│   ├── cifar_dataset.cpp        # CIFAR loader
│   └── huggingface_dataset.cpp  # HF datasets loader
│
├── transforms/
│   ├── geometric.cpp            # Crop, flip, rotate, etc.
│   ├── color.cpp                # ColorJitter, normalize
│   ├── noise.cpp                # Gaussian noise, blur
│   └── advanced.cpp             # Mixup, CutMix, RandAugment
│
└── gui/panels/
    ├── asset_browser.cpp        # Enhanced with preview
    ├── dataset_panel.cpp        # Refactored to use registry
    └── node_editor.cpp          # Dataset nodes added
```

---

## Appendix B: Configuration Examples

### Dataset Configuration (JSON)

```json
{
  "name": "my_dataset",
  "type": "ImageFolder",
  "path": "./data/images",
  "config": {
    "image_size": [224, 224],
    "normalize": {
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225]
    }
  },
  "split": {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
    "stratified": true,
    "seed": 42
  },
  "augmentation": {
    "train": [
      {"type": "RandomResizedCrop", "size": 224, "scale": [0.8, 1.0]},
      {"type": "RandomHorizontalFlip", "p": 0.5},
      {"type": "ColorJitter", "brightness": 0.2, "contrast": 0.2},
      {"type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    ],
    "val": [
      {"type": "Resize", "size": 256},
      {"type": "CenterCrop", "size": 224},
      {"type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    ]
  }
}
```

### DataLoader Configuration

```json
{
  "batch_size": 32,
  "shuffle": true,
  "num_workers": 4,
  "pin_memory": true,
  "drop_last": true,
  "prefetch_factor": 2
}
```

---

*Document Version: 1.0*
*Last Updated: 2024*
*Author: CyxWiz Team*
