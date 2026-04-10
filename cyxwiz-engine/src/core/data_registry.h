#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <functional>
#include <optional>
#include <chrono>

#include "dataset_base.h"

// Forward declarations for Arrow
namespace arrow {
    class Table;
}

// Forward declarations for preprocessing and augmentation
namespace cyxwiz {
    struct PreprocessingConfig;
    class AnnotationManager;
    class ArrowDataset;

    namespace transforms {
        class Compose;
    }
}

namespace cyxwiz {

// Forward declarations
class DatasetHandle;

/**
 * Memory statistics for monitoring
 */
struct MemoryStats {
    size_t total_allocated = 0;          // Total bytes allocated by all datasets
    size_t total_cached = 0;             // Bytes currently in cache
    size_t peak_usage = 0;               // Peak memory usage
    size_t memory_limit = 0;             // Configured memory limit
    size_t datasets_count = 0;           // Number of loaded datasets

    // Cache statistics
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    size_t cache_evictions = 0;

    // Texture memory
    size_t texture_memory = 0;
    size_t texture_count = 0;

    // Helper methods
    float GetCacheHitRate() const {
        size_t total = cache_hits + cache_misses;
        return total > 0 ? static_cast<float>(cache_hits) / total * 100.0f : 0.0f;
    }

    float GetUsagePercent() const {
        return memory_limit > 0 ? static_cast<float>(total_allocated) / memory_limit * 100.0f : 0.0f;
    }

    std::string FormatBytes(size_t bytes) const {
        const char* units[] = {"B", "KB", "MB", "GB"};
        int unit_index = 0;
        double size = static_cast<double>(bytes);
        while (size >= 1024.0 && unit_index < 3) {
            size /= 1024.0;
            unit_index++;
        }
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unit_index]);
        return buffer;
    }
};

// DatasetInfo moved to dataset_types.h

/**
 * Preview data for quick display
 */
struct DatasetPreview {
    DatasetType type = DatasetType::None;
    size_t num_samples = 0;
    size_t num_classes = 0;
    std::vector<size_t> shape;
    size_t file_size = 0;

    // For tabular data
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;

    // For image data
    std::vector<std::vector<float>> thumbnails;
    std::vector<int> thumbnail_labels;
    int thumbnail_width = 0;
    int thumbnail_height = 0;
    int thumbnail_channels = 0;
};

// SplitConfig moved to dataset_types.h

/**
 * Streaming configuration for large datasets
 */
struct StreamingConfig {
    bool enabled = false;
    size_t buffer_size = 1000;         // Number of samples to prefetch
    size_t chunk_size = 100;           // Samples per chunk
    bool shuffle_buffer = true;
    int prefetch_threads = 2;
};

/**
 * HDF5 dataset configuration
 */
struct HDF5Config {
    std::string data_path = "";      // Path to data dataset (e.g., "/images")
    std::string label_path = "";     // Path to labels dataset (e.g., "/labels")
    bool auto_detect = true;         // Auto-detect data/label paths
    bool normalize = true;           // Normalize uint8 to [0,1]

    // Lazy loading options
    bool lazy_loading = true;        // Enable lazy loading (default: true for large files)
    size_t chunk_size = 256;         // Number of samples per chunk
    size_t chunk_cache_size = 32;    // Max number of chunks to keep in memory
    size_t lazy_threshold = 10000;   // Enable lazy loading if num_samples > this

    // Layout options
    bool auto_detect_nchw = true;    // Auto-detect NCHW format and transpose to NHWC
    bool force_nchw = false;         // Force NCHW->NHWC transpose (overrides auto-detect)

    // Parallel I/O options
    bool prefetch_enabled = true;    // Enable background prefetching
    size_t prefetch_threads = 2;     // Number of prefetch worker threads
    size_t prefetch_ahead = 4;       // Number of chunks to prefetch ahead
};

/**
 * HDF5 export configuration
 */
struct HDF5ExportConfig {
    std::string data_path = "/images";   // Path to store data dataset
    std::string label_path = "/labels";  // Path to store labels dataset

    // Compression options
    bool compress = true;                // Enable GZIP compression
    int compression_level = 4;           // GZIP level (1-9, higher = more compression)

    // Chunking options
    bool chunked = true;                 // Enable chunked storage
    size_t chunk_samples = 256;          // Number of samples per chunk

    // Data type options
    bool store_as_uint8 = false;         // Store as uint8 (assumes data is [0,1], multiplies by 255)
    bool store_as_nchw = false;          // Transpose NHWC to NCHW for PyTorch compatibility

    // Metadata
    bool include_metadata = true;        // Include class names, split indices, etc.
    std::vector<std::string> class_names; // Optional class names to store
};

/**
 * HuggingFace dataset configuration
 */
struct HuggingFaceConfig {
    std::string dataset_name;          // e.g., "mnist", "cifar10", "imdb"
    std::string subset;                // Optional subset name
    std::string split = "train";       // "train", "validation", "test"
    std::string cache_dir;             // Local cache directory
    bool streaming = false;            // Use streaming mode
    std::string auth_token;            // Optional HF auth token
};

/**
 * Kaggle dataset configuration
 */
struct KaggleConfig {
    std::string dataset_slug;          // e.g., "zalando-research/fashionmnist", "uciml/iris"
    std::string competition;           // Competition name (alternative to dataset_slug)
    std::string file_name;             // Specific file to load (optional, loads all if empty)
    std::string cache_dir;             // Local cache directory
    std::string username;              // Kaggle username (optional, uses ~/.kaggle/kaggle.json)
    std::string api_key;               // Kaggle API key (optional)
    bool unzip = true;                 // Auto-unzip downloaded files
};

/**
 * Custom dataset configuration
 * Supports loading from various file formats with user-defined schema
 */
struct CustomConfig {
    std::string data_path;             // Path to data file or directory
    std::string labels_path;           // Optional separate labels file
    std::string format;                // File format: "json", "npy", "npz", "binary", "text", "folder"

    // Schema configuration
    std::string data_key;              // JSON/NPZ key for data (e.g., "images", "X", "features")
    std::string labels_key;            // JSON/NPZ key for labels (e.g., "labels", "y", "targets")
    std::vector<size_t> shape;         // Expected shape per sample (e.g., {28, 28, 1} for images)
    size_t num_classes = 0;            // Number of classes (0 = auto-detect)
    std::vector<std::string> class_names; // Optional class names

    // Data type and normalization
    std::string dtype = "float32";     // Data type: "float32", "float64", "uint8", "int32"
    bool normalize = true;             // Normalize to [0, 1] range
    float scale = 1.0f;                // Scale factor (e.g., 1/255 for images)

    // For text/folder formats
    std::string delimiter = ",";       // Delimiter for text files
    bool has_header = false;           // First row is header
    int label_column = -1;             // Label column index (-1 = last column)
};

/**
 * Handle to a loaded dataset
 * Provides safe access to dataset data
 */
class DatasetHandle {
public:
    DatasetHandle() = default;
    DatasetHandle(std::shared_ptr<Dataset> dataset, const std::string& name);

    // Validity check
    bool IsValid() const { return dataset_ != nullptr; }
    explicit operator bool() const { return IsValid(); }

    // Info access
    DatasetInfo GetInfo() const;
    std::string GetName() const { return name_; }

    // Data access
    size_t Size() const;
    size_t Size(DatasetSplit split) const;
    std::pair<std::vector<float>, int> GetSample(size_t index) const;
    std::pair<std::vector<std::vector<float>>, std::vector<int>>
        GetBatch(const std::vector<size_t>& indices) const;

    // Split access
    const std::vector<size_t>& GetTrainIndices() const;
    const std::vector<size_t>& GetValIndices() const;
    const std::vector<size_t>& GetTestIndices() const;
    const std::vector<size_t>& GetSplitIndices(DatasetSplit split) const;

    // Apply split configuration
    void ApplySplit(const SplitConfig& config);

    // Access underlying dataset for type-specific operations (preview)
    Dataset* GetUnderlyingDataset() const { return dataset_.get(); }

private:
    std::shared_ptr<Dataset> dataset_;
    std::string name_;
};

/**
 * DataRegistry - Singleton managing all loaded datasets
 *
 * Central registry for dataset management. Provides:
 * - Dataset loading and unloading
 * - Preview generation (lightweight)
 * - Memory management
 * - Dataset discovery
 */
class DataRegistry {
public:
    // Singleton access
    static DataRegistry& Instance();

    // Prevent copying
    DataRegistry(const DataRegistry&) = delete;
    DataRegistry& operator=(const DataRegistry&) = delete;

    // Dataset loading
    DatasetHandle LoadDataset(const std::string& path, const std::string& name = "");
    DatasetHandle LoadMNIST(const std::string& path, const std::string& name = "mnist");
    DatasetHandle LoadCIFAR10(const std::string& path, const std::string& name = "cifar10");
    DatasetHandle LoadCSV(const std::string& path, const std::string& name = "");
    DatasetHandle LoadTSV(const std::string& path, const std::string& name = "");
    DatasetHandle LoadJSON(const std::string& path, const std::string& name = "");
    DatasetHandle LoadTXT(const std::string& path, const std::string& name = "");
    DatasetHandle LoadImageFolder(const std::string& path, const std::string& name = "");
    DatasetHandle LoadImageCSV(const std::string& image_folder, const std::string& csv_path,
                                const std::string& name = "", int target_width = 224, int target_height = 224,
                                size_t cache_size = 100);
    DatasetHandle LoadHuggingFace(const HuggingFaceConfig& config, const std::string& name = "");
    DatasetHandle LoadKaggle(const KaggleConfig& config, const std::string& name = "");
    DatasetHandle LoadCustom(const CustomConfig& config, const std::string& name = "");
    DatasetHandle LoadHDF5(const std::string& path, const std::string& name = "",
                           const HDF5Config& config = {});

    // Streaming dataset loading
    DatasetHandle LoadStreamingDataset(const std::string& path, const StreamingConfig& config, const std::string& name = "");

    // Apache Arrow columnar data support (Data Studio foundation)
    std::shared_ptr<class ArrowDataset> LoadArrowTable(const std::string& path, const std::string& name = "");
    std::shared_ptr<class ArrowDataset> RegisterArrowTable(std::shared_ptr<arrow::Table> table, const std::string& name);
    std::shared_ptr<class ArrowDataset> GetArrowDataset(const std::string& name);
    bool IsArrowDataset(const std::string& name) const;

    // Disk-backed Parquet dataset support — used automatically for files that
    // are larger than available RAM. See LoadTabularCSV below for the
    // dispatcher that picks between in-memory Arrow and disk-backed Parquet.
    // The backing file is a Snappy-compressed Parquet written to the system
    // temp directory (see ParquetBackedDataset::GetCacheFilePath).
    std::shared_ptr<class ParquetBackedDataset> GetParquetBackedDataset(const std::string& name) const;
    bool IsParquetBackedDataset(const std::string& name) const;
    void RegisterParquetBacked(const std::string& name,
                               std::shared_ptr<class ParquetBackedDataset> dataset);

    // Which backing store a successful LoadTabularCSV call ended up using.
    enum class TabularLoadBackend {
        Failed,      // load failed; nothing is registered
        InMemory,    // ArrowDataset registered in arrow_datasets_
        DiskBacked   // ParquetBackedDataset registered in parquet_backed_datasets_
    };

    /**
     * High-level CSV loader with automatic in-memory vs disk-backed dispatch.
     *
     * Compares the source file size to the available RAM. If the file comfortably
     * fits (file_size < 0.75 * available RAM) or force_disk_backed is false and
     * the file is under a safety threshold, uses LoadCSVToArrow (in-memory Arrow
     * table). Otherwise converts the CSV to a Snappy Parquet cache in the system
     * temp dir and opens that via memory-mapped reads.
     *
     * The force_disk_backed flag is the Advanced-tab escape hatch: users can
     * force the disk-backed path even for small datasets (for testing, or to
     * simulate the experience on limited hardware).
     *
     * Returns the backend that was used so the caller (DataInput dialog, etc.)
     * can show an honest status line. On success the dataset is queryable via
     * the matching GetArrowDataset / GetParquetBackedDataset accessor.
     */
    TabularLoadBackend LoadTabularCSV(const std::string& path,
                                       const std::string& name,
                                       bool has_header = true,
                                       char delimiter = ',',
                                       int skip_rows = 0,
                                       int64_t max_rows = 0,
                                       bool force_disk_backed = false);

    // Arrow format-specific loaders (for DataInput node)
    //
    // LoadCSVToArrow auto-detects the right Arrow CSV block_size from the
    // file size so the whole table loads as a single Arrow chunk whenever
    // possible. This means the ArrowDatasetBatcher hits its fast
    // raw_values() direct-pointer path, and integer columns are then
    // downcast to their natural width (e.g. MNIST pixels int64 -> uint8,
    // an 8x memory saving). Users don't need to tune anything.
    //
    // max_rows: if > 0, keep only the first max_rows rows after load (via
    // table->Slice). Useful for training on a subset of a large dataset.
    // 0 means "load all rows". Note that this slices after the full parse,
    // so load-time memory is still proportional to the full file size —
    // for a true lazy load with memory cap, see the deferred Parquet+mmap
    // design (not yet implemented).
    std::shared_ptr<class ArrowDataset> LoadCSVToArrow(const std::string& path, const std::string& name,
                                                        bool has_header = true, char delimiter = ',', int skip_rows = 0,
                                                        int64_t max_rows = 0);
    std::shared_ptr<class ArrowDataset> LoadParquetToArrow(const std::string& path, const std::string& name);
    std::shared_ptr<class ArrowDataset> LoadJSONToArrow(const std::string& path, const std::string& name, bool json_lines = false);
    std::shared_ptr<class ArrowDataset> LoadExcelToArrow(const std::string& path, const std::string& name, int sheet_idx = 0);
    std::shared_ptr<class ArrowDataset> LoadImageFolderToArrow(const std::string& path, const std::string& name);
    std::shared_ptr<class ArrowDataset> LoadMLDatasetToArrow(const std::string& ml_type, const std::string& name);

    // Arrow export methods (for DataOutput node)
    bool ExportArrowToCSV(const std::string& dataset_name, const std::string& output_path);
    bool ExportArrowToParquet(const std::string& dataset_name, const std::string& output_path);
    bool ExportArrowToJSON(const std::string& dataset_name, const std::string& output_path);

    // Dataset unloading
    void UnloadDataset(const std::string& name);
    void UnloadAll();

    // Dataset access
    DatasetHandle GetDataset(const std::string& name);
    bool HasDataset(const std::string& name) const;
    std::vector<DatasetInfo> ListDatasets() const;
    std::vector<std::string> GetDatasetNames() const;

    // Preview (lightweight, doesn't fully load)
    DatasetPreview GetPreview(const std::string& path, int max_samples = 5);

    // Type detection
    static DatasetType DetectType(const std::string& path);
    static std::string TypeToString(DatasetType type);

    // Memory management
    size_t GetTotalMemoryUsage() const;
    void SetMemoryLimit(size_t bytes);
    size_t GetMemoryLimit() const { return memory_limit_; }
    MemoryStats GetMemoryStats() const;
    void ResetCacheStats();

    // Memory optimization
    void TrimMemory(size_t target_bytes = 0);  // Evict least-used datasets until under limit
    void EvictOldest();                         // Evict the least recently used dataset
    bool IsMemoryPressure() const;              // Check if approaching memory limit
    void EnableAutoEviction(bool enable) { auto_eviction_enabled_ = enable; }
    bool IsAutoEvictionEnabled() const { return auto_eviction_enabled_; }

    // Memory pressure callback (called when over limit)
    using MemoryPressureCallback = std::function<void(size_t current, size_t limit)>;
    void SetOnMemoryPressure(MemoryPressureCallback callback) { on_memory_pressure_ = std::move(callback); }

    // Callbacks
    using DatasetLoadedCallback = std::function<void(const std::string& name, const DatasetInfo& info)>;
    using DatasetUnloadedCallback = std::function<void(const std::string& name)>;
    using LoadProgressCallback = std::function<void(float progress, const std::string& status)>;

    void SetOnDatasetLoaded(DatasetLoadedCallback callback) { on_loaded_ = std::move(callback); }
    void SetOnDatasetUnloaded(DatasetUnloadedCallback callback) { on_unloaded_ = std::move(callback); }
    void SetOnLoadProgress(LoadProgressCallback callback) { on_progress_ = std::move(callback); }

    // Dataset configuration export/import
    bool ExportConfig(const std::string& name, const std::string& filepath) const;
    bool ExportConfig(const std::string& name, const std::string& filepath, const SplitConfig& split) const;
    bool ImportConfig(const std::string& filepath, std::string& out_name);
    bool ImportConfig(const std::string& filepath, std::string& out_name, SplitConfig& out_split);
    static std::string SerializeConfig(const DatasetInfo& info, const SplitConfig& split);
    static bool DeserializeConfig(const std::string& json_str, DatasetInfo& info, SplitConfig& split);

    // HDF5 export
    bool ExportHDF5(const std::string& name, const std::string& filepath,
                    const HDF5ExportConfig& config = {});
    bool ExportHDF5(DatasetHandle handle, const std::string& filepath,
                    const HDF5ExportConfig& config = {});

    // Dataset versioning
    struct DatasetVersion {
        std::string version_id;
        std::string timestamp;
        std::string description;
        size_t num_samples;
        std::string checksum;
    };
    std::vector<DatasetVersion> GetVersionHistory(const std::string& name) const;
    bool SaveVersion(const std::string& name, const std::string& description = "");

    // Preprocessing configuration management
    void SetPreprocessingConfig(const std::string& dataset_id, const PreprocessingConfig& config);
    PreprocessingConfig GetPreprocessingConfig(const std::string& dataset_id) const;
    bool HasPreprocessingConfig(const std::string& dataset_id) const;
    void ClearPreprocessingConfig(const std::string& dataset_id);

    // Augmentation pipeline management
    void SetAugmentationPipeline(const std::string& dataset_id,
                                  std::shared_ptr<transforms::Compose> pipeline);
    std::shared_ptr<transforms::Compose> GetAugmentationPipeline(
        const std::string& dataset_id) const;
    bool HasAugmentationPipeline(const std::string& dataset_id) const;
    void ClearAugmentationPipeline(const std::string& dataset_id);

    // Annotation management (for labeling and export)
    AnnotationManager& GetAnnotationManager();
    const AnnotationManager& GetAnnotationManager() const;

private:
    DataRegistry() = default;

    // Generate unique name if not provided
    std::string GenerateUniqueName(const std::string& base_name);

    // Dataset storage
    std::map<std::string, std::shared_ptr<Dataset>> datasets_;

    // Arrow dataset storage (separate for Data Studio columnar data)
    std::map<std::string, std::shared_ptr<class ArrowDataset>> arrow_datasets_;

    // Disk-backed Parquet datasets — populated by LoadTabularCSV when a file
    // is too large to fit comfortably in RAM. Lookups by name fall through
    // to this map when not found in arrow_datasets_.
    std::map<std::string, std::shared_ptr<class ParquetBackedDataset>> parquet_backed_datasets_;

    mutable std::mutex mutex_;

    // Memory management
    size_t memory_limit_ = 4ULL * 1024 * 1024 * 1024;  // 4GB default
    mutable size_t peak_usage_ = 0;
    bool auto_eviction_enabled_ = false;
    float memory_pressure_threshold_ = 0.9f;  // 90% triggers pressure warning

    // LRU tracking - maps dataset name to last access time
    mutable std::map<std::string, std::chrono::steady_clock::time_point> last_access_times_;

    // Global cache statistics
    mutable size_t total_cache_hits_ = 0;
    mutable size_t total_cache_misses_ = 0;
    mutable size_t total_cache_evictions_ = 0;

    // Callbacks
    DatasetLoadedCallback on_loaded_;
    DatasetUnloadedCallback on_unloaded_;
    LoadProgressCallback on_progress_;
    MemoryPressureCallback on_memory_pressure_;

    // Name generation
    int name_counter_ = 0;

    // Version history storage (in-memory, could be persisted)
    std::map<std::string, std::vector<DatasetVersion>> version_history_;

    // Preprocessing configurations per dataset
    mutable std::map<std::string, PreprocessingConfig> preprocessing_configs_;

    // Augmentation pipelines per dataset
    mutable std::map<std::string, std::shared_ptr<transforms::Compose>> augmentation_pipelines_;

    // Annotation manager
    mutable std::unique_ptr<AnnotationManager> annotation_manager_;
};

} // namespace cyxwiz
