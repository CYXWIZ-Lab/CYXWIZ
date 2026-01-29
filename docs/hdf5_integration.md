# HDF5 Dataset Integration Architecture

> **Goal**: Integrate HDF5 file format into the CyxWiz Dataset Manager with full support for ML training workflows, maintaining compatibility with the existing DataRegistry architecture.

## Table of Contents

1. [Overview](#1-overview)
2. [HDF5 Format Background](#2-hdf5-format-background)
3. [Current State](#3-current-state)
4. [Architecture Design](#4-architecture-design)
5. [HDF5Dataset Class](#5-hdf5dataset-class)
6. [Supported Data Layouts](#6-supported-data-layouts)
7. [Memory Management](#7-memory-management)
8. [GUI Integration](#8-gui-integration)
9. [Implementation Plan](#9-implementation-plan)
10. [API Reference](#10-api-reference)

---

## 1. Overview

### 1.1 Why HDF5?

| Feature | Benefit for ML |
|---------|----------------|
| **Hierarchical structure** | Store images, labels, metadata in single file |
| **Chunked storage** | Efficient partial reads for large datasets |
| **Compression** | GZIP, LZF, SZIP reduce storage 2-10x |
| **Cross-platform** | Same file works on Windows/Linux/macOS |
| **Industry standard** | Used by PyTorch, TensorFlow, Keras, NumPy |
| **Lazy loading** | Read only what you need, when you need it |

### 1.2 Use Cases

1. **Image datasets** - Store preprocessed images as 4D tensors `[N, H, W, C]`
2. **Tabular data** - Features + labels in structured format
3. **Time series** - Sequential data with metadata
4. **Multi-modal** - Images + text + labels in one file
5. **Large datasets** - 100GB+ files with streaming access

---

## 2. HDF5 Format Background

### 2.1 Structure

```
my_dataset.h5
├── /images          # Dataset: float32 [10000, 224, 224, 3]
├── /labels          # Dataset: int32 [10000]
├── /metadata        # Group
│   ├── /class_names # Dataset: string [10]
│   └── /split_indices
│       ├── /train   # Dataset: int32 [8000]
│       ├── /val     # Dataset: int32 [1000]
│       └── /test    # Dataset: int32 [1000]
└── attrs
    ├── num_classes: 10
    ├── image_size: [224, 224]
    └── created: "2024-01-15"
```

### 2.2 Key Concepts

| Concept | Description |
|---------|-------------|
| **Group** | Folder-like container (like `/metadata/split_indices`) |
| **Dataset** | N-dimensional array with dtype |
| **Attribute** | Key-value metadata on groups or datasets |
| **Chunk** | Storage block for partial I/O (e.g., 256 samples per chunk) |
| **Compression** | Per-dataset (GZIP level 1-9, LZF, SZIP) |

### 2.3 Common ML Dataset Patterns

**Pattern A: Flat Structure (Simple)**
```
data.h5
├── /data       # [N, features] or [N, H, W, C]
└── /labels     # [N]
```

**Pattern B: Split Structure (Train/Val/Test)**
```
data.h5
├── /train
│   ├── /images  # [N_train, H, W, C]
│   └── /labels  # [N_train]
├── /val
│   ├── /images
│   └── /labels
└── /test
    ├── /images
    └── /labels
```

**Pattern C: PyTorch-style (Features + Targets)**
```
data.h5
├── /features   # or /X, /inputs
└── /targets    # or /y, /labels, /outputs
```

**Pattern D: Keras-style (x_train, y_train, etc.)**
```
data.h5
├── /x_train
├── /y_train
├── /x_test
└── /y_test
```

---

## 3. Current State

### 3.1 What Exists

| Component | File | Status |
|-----------|------|--------|
| HighFive library | vcpkg dependency | Installed |
| `DataTable::LoadFromHDF5()` | `data_table.cpp:272` | Working (table viewer) |
| `#ifdef CYXWIZ_HAS_HDF5` | CMake option | Configured |
| File dialog filter | `file_dialogs.cpp` | Includes `.h5`, `.hdf5` |

### 3.2 What's Missing

| Component | Description |
|-----------|-------------|
| `DatasetType::HDF5` | Enum value in `DatasetType` |
| `HDF5Dataset` class | Dataset implementation |
| `DataRegistry::LoadHDF5()` | Factory method |
| `DetectType()` update | Recognize `.h5`/`.hdf5` extensions |
| Schema detection | Auto-detect data/labels datasets |
| Lazy loading | Stream large files without full load |

---

## 4. Architecture Design

### 4.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DataRegistry                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ LoadHDF5(path, name, config) → DatasetHandle             │   │
│  │   1. Validate file exists                                │   │
│  │   2. Detect schema (auto or manual)                      │   │
│  │   3. Create HDF5Dataset instance                         │   │
│  │   4. Register in datasets_ map                           │   │
│  │   5. Return handle                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        HDF5Dataset                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Members:                                                 │   │
│  │   - file_path_: string                                   │   │
│  │   - data_path_: string (e.g., "/images")                │   │
│  │   - label_path_: string (e.g., "/labels")               │   │
│  │   - file_: HighFive::File (lazy opened)                 │   │
│  │   - shape_: vector<size_t>                              │   │
│  │   - dtype_: DataType enum                               │   │
│  │   - chunk_cache_: LRU<chunk_idx, vector<float>>         │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Methods:                                                 │   │
│  │   - Size() const → size_t                               │   │
│  │   - GetItem(idx) → pair<vector<float>, int>             │   │
│  │   - GetBatch(indices) → pair<vector<float>, vector<int>>│   │
│  │   - GetInfo() → DatasetInfo                             │   │
│  │   - GetShape() → vector<size_t>                         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      HighFive Library                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ - File::open(path, ReadOnly)                            │   │
│  │ - file.getDataSet("/images")                            │   │
│  │ - dataset.read<float>(buffer, slice)                    │   │
│  │ - dataset.getSpace().getDimensions()                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow

```
User Action                    System Response
────────────────────────────────────────────────────────────────
1. Load HDF5 file       →      DetectType() returns HDF5
                        →      Show HDF5 config dialog

2. Select data/labels   →      HDF5Dataset created with paths
   paths in dialog      →      Schema validated
                        →      Dataset registered

3. Create DataInput     →      Node shows dataset name
   node in editor       →      Shape inferred from HDF5

4. Start training       →      DatasetBatcher created
                        →      GetItem() reads from HDF5
                        →      Chunk caching reduces I/O

5. Memory pressure      →      LRU evicts old chunks
                        →      File handle remains open
```

---

## 5. HDF5Dataset Class

### 5.1 Class Definition

```cpp
// In data_registry.h

/**
 * Configuration for HDF5 dataset loading
 */
struct HDF5Config {
    std::string data_path = "";      // Path to data dataset (e.g., "/images")
    std::string label_path = "";     // Path to labels dataset (e.g., "/labels")
    bool auto_detect = true;         // Auto-detect data/label paths
    size_t chunk_cache_size = 100;   // Number of chunks to cache
    bool normalize = true;           // Normalize to [0, 1] if uint8
};

// In data_registry.cpp

class HDF5Dataset : public Dataset {
public:
    HDF5Dataset(const std::string& path, const HDF5Config& config = {});
    ~HDF5Dataset();

    // Dataset interface
    size_t Size() const override;
    std::pair<std::vector<float>, int> GetItem(size_t index) const override;
    DatasetInfo GetInfo() const override;

    // HDF5-specific
    std::vector<size_t> GetDataShape() const;
    std::vector<std::string> ListDatasets() const;
    bool HasDataset(const std::string& path) const;

private:
    bool DetectSchema();
    bool LoadMetadata();
    void ReadChunk(size_t chunk_idx) const;

    std::string file_path_;
    std::string data_path_;
    std::string label_path_;
    HDF5Config config_;

    // HighFive objects (lazy initialization)
    mutable std::unique_ptr<HighFive::File> file_;
    mutable std::unique_ptr<HighFive::DataSet> data_dataset_;
    mutable std::unique_ptr<HighFive::DataSet> label_dataset_;

    // Metadata
    std::vector<size_t> data_shape_;  // [N, ...] full shape
    std::vector<size_t> sample_shape_; // [...] single sample shape
    size_t num_samples_ = 0;
    size_t num_classes_ = 0;
    size_t bytes_per_sample_ = 0;

    // Chunk cache for efficient I/O
    size_t chunk_size_ = 256;  // Samples per chunk
    mutable std::map<size_t, std::vector<float>> chunk_cache_;
    mutable std::list<size_t> cache_order_;  // LRU tracking

    // Thread safety
    mutable std::mutex file_mutex_;
};
```

### 5.2 Key Methods

#### Constructor
```cpp
HDF5Dataset::HDF5Dataset(const std::string& path, const HDF5Config& config)
    : file_path_(path), config_(config)
{
    // Open file (read-only)
    file_ = std::make_unique<HighFive::File>(path, HighFive::File::ReadOnly);

    // Auto-detect or use provided paths
    if (config_.auto_detect) {
        if (!DetectSchema()) {
            throw std::runtime_error("Could not auto-detect HDF5 schema");
        }
    } else {
        data_path_ = config_.data_path;
        label_path_ = config_.label_path;
    }

    // Load metadata
    LoadMetadata();

    spdlog::info("HDF5Dataset loaded: {} samples, shape {}",
                 num_samples_, GetInfo().GetShapeString());
}
```

#### Schema Detection
```cpp
bool HDF5Dataset::DetectSchema() {
    // Common data path names (priority order)
    const std::vector<std::string> data_names = {
        "/images", "/data", "/features", "/X", "/x", "/inputs",
        "/x_train", "/train/images", "/train/data"
    };

    // Common label path names
    const std::vector<std::string> label_names = {
        "/labels", "/targets", "/y", "/Y", "/outputs",
        "/y_train", "/train/labels", "/train/targets"
    };

    // Try to find data dataset
    for (const auto& name : data_names) {
        if (file_->exist(name)) {
            auto ds = file_->getDataSet(name);
            auto dims = ds.getSpace().getDimensions();
            if (dims.size() >= 1) {  // At least 1D
                data_path_ = name;
                break;
            }
        }
    }

    // Try to find labels dataset
    for (const auto& name : label_names) {
        if (file_->exist(name)) {
            auto ds = file_->getDataSet(name);
            auto dims = ds.getSpace().getDimensions();
            if (dims.size() == 1) {  // Labels should be 1D
                label_path_ = name;
                break;
            }
        }
    }

    return !data_path_.empty();  // Labels are optional
}
```

#### GetItem (with caching)
```cpp
std::pair<std::vector<float>, int> HDF5Dataset::GetItem(size_t index) const {
    std::lock_guard<std::mutex> lock(file_mutex_);

    // Calculate chunk index
    size_t chunk_idx = index / chunk_size_;
    size_t local_idx = index % chunk_size_;

    // Check cache
    if (chunk_cache_.find(chunk_idx) == chunk_cache_.end()) {
        ReadChunk(chunk_idx);  // Load chunk into cache
    }

    // Update LRU order
    cache_order_.remove(chunk_idx);
    cache_order_.push_front(chunk_idx);

    // Evict if over limit
    while (chunk_cache_.size() > config_.chunk_cache_size) {
        size_t evict_idx = cache_order_.back();
        cache_order_.pop_back();
        chunk_cache_.erase(evict_idx);
    }

    // Extract sample from cached chunk
    const auto& chunk = chunk_cache_[chunk_idx];
    size_t sample_size = bytes_per_sample_ / sizeof(float);
    size_t offset = local_idx * sample_size;

    std::vector<float> sample(chunk.begin() + offset,
                               chunk.begin() + offset + sample_size);

    // Get label
    int label = 0;
    if (!label_path_.empty()) {
        // Labels are typically small, read directly
        label_dataset_->select({index}, {1}).read(&label);
    }

    return {sample, label};
}
```

---

## 6. Supported Data Layouts

### 6.1 Layout Detection Matrix

| HDF5 Structure | Detection Method | Data Path | Label Path |
|----------------|------------------|-----------|------------|
| `/images` + `/labels` | Name matching | `/images` | `/labels` |
| `/data` + `/targets` | Name matching | `/data` | `/targets` |
| `/X` + `/y` | Name matching | `/X` | `/y` |
| `/train/images` | Nested groups | `/train/images` | `/train/labels` |
| `/x_train` + `/y_train` | Keras style | `/x_train` | `/y_train` |
| Single dataset | Only one 2D+ dataset | Auto | None (unsupervised) |

### 6.2 Supported Data Types

| HDF5 Type | C++ Type | Normalization |
|-----------|----------|---------------|
| `H5T_NATIVE_FLOAT` | `float` | None |
| `H5T_NATIVE_DOUBLE` | `double` → `float` | Cast |
| `H5T_NATIVE_UINT8` | `uint8_t` → `float` | Divide by 255 |
| `H5T_NATIVE_INT32` | `int32_t` → `float` | Cast |
| `H5T_NATIVE_INT64` | `int64_t` → `float` | Cast |

### 6.3 Shape Interpretations

| Dimensions | Interpretation | Sample Shape |
|------------|----------------|--------------|
| `[N]` | 1D features | `[1]` |
| `[N, F]` | Tabular (N samples, F features) | `[F]` |
| `[N, H, W]` | Grayscale images | `[H, W, 1]` |
| `[N, H, W, C]` | Color images (channels last) | `[H, W, C]` |
| `[N, C, H, W]` | Color images (channels first) | `[H, W, C]` (transposed) |
| `[N, T, F]` | Time series (T steps, F features) | `[T, F]` |

---

## 7. Memory Management

### 7.1 Lazy Loading Strategy

```
File Open (fast)
    │
    ▼
Metadata Read (shape, dtype, attrs)
    │
    ▼
No data loaded yet ← Memory: ~0 bytes
    │
    ▼
GetItem(0) called
    │
    ▼
Load chunk 0 (samples 0-255) ← Memory: chunk_size * sample_size
    │
    ▼
GetItem(1-255) → Cache hit
    │
    ▼
GetItem(300) called
    │
    ▼
Load chunk 1 (samples 256-511) ← Memory: 2 * chunk_size * sample_size
    │
    ▼
... continue until cache limit
    │
    ▼
Cache full → LRU eviction
```

### 7.2 Memory Estimation

```cpp
size_t HDF5Dataset::EstimateMemoryUsage() const {
    // Metadata (always loaded)
    size_t meta_size = sizeof(*this);

    // Cache size (worst case: all chunks loaded)
    size_t max_cache_bytes = config_.chunk_cache_size * chunk_size_ * bytes_per_sample_;

    // Actual cache usage
    size_t current_cache_bytes = chunk_cache_.size() * chunk_size_ * bytes_per_sample_;

    return meta_size + current_cache_bytes;
}
```

### 7.3 Integration with DataRegistry Memory Limit

```cpp
// In DataRegistry::LoadHDF5()
DatasetHandle DataRegistry::LoadHDF5(const std::string& path,
                                      const std::string& name,
                                      const HDF5Config& config) {
    // Check memory before loading
    size_t estimated_size = EstimateHDF5Size(path, config);

    if (stats_.total_allocated + estimated_size > memory_limit_) {
        // Try to free memory
        TrimMemory(estimated_size);

        if (stats_.total_allocated + estimated_size > memory_limit_) {
            spdlog::warn("HDF5 dataset may exceed memory limit, using streaming mode");
            // Reduce chunk cache for streaming
            HDF5Config streaming_config = config;
            streaming_config.chunk_cache_size = 10;
            return LoadHDF5Internal(path, name, streaming_config);
        }
    }

    return LoadHDF5Internal(path, name, config);
}
```

---

## 8. GUI Integration

### 8.1 Dataset Panel Updates

```cpp
// In dataset_panel.cpp

void DatasetPanel::OnLoadHDF5(const std::string& path) {
    // Show HDF5 configuration dialog
    HDF5ConfigDialog dialog;
    dialog.SetFilePath(path);

    // List available datasets in file
    auto datasets = HDF5Utils::ListDatasets(path);
    dialog.SetAvailableDatasets(datasets);

    if (dialog.Show()) {
        HDF5Config config = dialog.GetConfig();

        // Load async
        loading_task_ = async_manager_->RunAsync([=]() {
            auto& registry = DataRegistry::Instance();
            return registry.LoadHDF5(path, GenerateName(path), config);
        });
    }
}
```

### 8.2 HDF5 Configuration Dialog

```
┌─────────────────────────────────────────────────────────────┐
│ Load HDF5 Dataset                                      [X] │
├─────────────────────────────────────────────────────────────┤
│ File: /path/to/dataset.h5                                  │
│                                                             │
│ Available Datasets:                                         │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [▼] /                                                   │ │
│ │     [▼] images  [10000, 224, 224, 3] float32           │ │
│ │     [▼] labels  [10000] int32                          │ │
│ │     [▶] metadata                                        │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Data Configuration:                                         │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Data Path:    [/images            ▼] [Auto-detect ☑]   │ │
│ │ Label Path:   [/labels            ▼]                   │ │
│ │                                                         │ │
│ │ Options:                                                │ │
│ │ [ ] Normalize uint8 to [0, 1]                          │ │
│ │ [ ] Transpose NCHW → NHWC                              │ │
│ │ Cache Size:   [100    ] chunks                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Preview (first 5 samples):                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [img0] [img1] [img2] [img3] [img4]                     │ │
│ │  cat    dog    cat    bird   dog                       │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│                              [Cancel]  [Load Dataset]       │
└─────────────────────────────────────────────────────────────┘
```

### 8.3 DetectType Update

```cpp
// In data_registry.cpp

DatasetType DataRegistry::DetectType(const std::string& path) {
    // ... existing detection logic ...

    // Check file extension for HDF5
    std::string ext = GetExtension(path);
    if (ext == ".h5" || ext == ".hdf5" || ext == ".hdf") {
        return DatasetType::HDF5;
    }

    // ... rest of detection ...
}
```

---

## 9. Implementation Plan

### Phase 1: Core HDF5Dataset (Priority: High)

| Task | File | Effort |
|------|------|--------|
| Add `DatasetType::HDF5` | `data_registry.h` | 1 line |
| Add `HDF5Config` struct | `data_registry.h` | 15 lines |
| Implement `HDF5Dataset` class | `data_registry.cpp` | ~300 lines |
| Add `LoadHDF5()` to DataRegistry | `data_registry.cpp` | ~50 lines |
| Update `DetectType()` | `data_registry.cpp` | 5 lines |
| Update `TypeToString()` | `data_registry.cpp` | 1 line |

### Phase 2: GUI Integration (Priority: Medium)

| Task | File | Effort |
|------|------|--------|
| HDF5 config dialog | `dataset_panel.cpp` | ~200 lines |
| Tree view for HDF5 structure | `dataset_panel.cpp` | ~100 lines |
| Preview rendering | `dataset_panel.cpp` | ~50 lines |

### Phase 3: Advanced Features (Priority: Low)

| Task | Description |
|------|-------------|
| NCHW → NHWC transpose | Auto-detect and convert |
| Compression support | GZIP, LZF decompression |
| Virtual datasets | Support HDF5 virtual datasets |
| Parallel I/O | Multi-threaded chunk loading |
| Write support | Export datasets to HDF5 |

---

## 10. API Reference

### 10.1 Public API

```cpp
// Load HDF5 dataset with auto-detection
DatasetHandle handle = DataRegistry::Instance().LoadHDF5(
    "path/to/data.h5",
    "my_dataset"
);

// Load with manual configuration
HDF5Config config;
config.data_path = "/train/images";
config.label_path = "/train/labels";
config.normalize = true;
config.chunk_cache_size = 200;

DatasetHandle handle = DataRegistry::Instance().LoadHDF5(
    "path/to/data.h5",
    "my_dataset",
    config
);

// Access data
Dataset* ds = handle.Get();
auto [sample, label] = ds->GetItem(0);
DatasetInfo info = ds->GetInfo();
```

### 10.2 HDF5 Utility Functions

```cpp
namespace HDF5Utils {
    // List all datasets in file
    std::vector<DatasetDesc> ListDatasets(const std::string& path);

    // Get dataset info without loading
    DatasetDesc GetDatasetInfo(const std::string& path, const std::string& dataset_path);

    // Validate file is valid HDF5
    bool IsValidHDF5(const std::string& path);

    // Check if file has expected structure
    bool HasExpectedSchema(const std::string& path, const HDF5Config& config);
}

struct DatasetDesc {
    std::string path;           // e.g., "/images"
    std::string dtype;          // e.g., "float32"
    std::vector<size_t> shape;  // e.g., [10000, 224, 224, 3]
    bool is_chunked;
    size_t chunk_size;
    std::string compression;    // e.g., "gzip", "none"
};
```

---

## Appendix A: Example HDF5 Files

### A.1 Creating Test HDF5 (Python)

```python
import h5py
import numpy as np

# Create simple image dataset
with h5py.File('test_images.h5', 'w') as f:
    # Random images [1000, 64, 64, 3]
    images = np.random.randint(0, 255, (1000, 64, 64, 3), dtype=np.uint8)
    labels = np.random.randint(0, 10, 1000, dtype=np.int32)

    f.create_dataset('images', data=images, chunks=(100, 64, 64, 3), compression='gzip')
    f.create_dataset('labels', data=labels)

    # Metadata
    f.attrs['num_classes'] = 10
    f.attrs['class_names'] = ['cat', 'dog', 'bird', 'car', 'plane',
                              'ship', 'truck', 'horse', 'deer', 'frog']
```

### A.2 Reading with HighFive (C++)

```cpp
#include <highfive/H5File.hpp>

HighFive::File file("test_images.h5", HighFive::File::ReadOnly);

// Get dataset
auto dataset = file.getDataSet("/images");
auto dims = dataset.getSpace().getDimensions();  // [1000, 64, 64, 3]

// Read single sample
std::vector<uint8_t> sample(64 * 64 * 3);
dataset.select({0, 0, 0, 0}, {1, 64, 64, 3}).read(sample.data());

// Read batch
std::vector<uint8_t> batch(100 * 64 * 64 * 3);
dataset.select({0, 0, 0, 0}, {100, 64, 64, 3}).read(batch.data());
```

---

## Appendix B: Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| `HDF5 file not found` | Invalid path | Check file exists |
| `Cannot detect schema` | No recognized datasets | Use manual config |
| `Shape mismatch` | Data/labels different N | Validate file |
| `Unsupported dtype` | Complex numbers, etc. | Convert to float |
| `Out of memory` | File too large | Reduce cache size |
| `Chunk read failed` | Corrupted file | Re-download/recreate |

---

## Appendix C: Performance Benchmarks (Expected)

| Operation | Time (10K samples, 224x224x3) |
|-----------|-------------------------------|
| Open file | < 10ms |
| Read metadata | < 5ms |
| First GetItem (cold) | ~50ms (includes chunk load) |
| Subsequent GetItem (cached) | < 0.1ms |
| Full epoch (cached) | ~2s |
| Full epoch (uncached) | ~30s |

*Note: Actual performance depends on storage speed (SSD vs HDD) and compression.*
