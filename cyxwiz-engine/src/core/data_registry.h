#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <functional>
#include <optional>
#include <chrono>

namespace cyxwiz {

// Forward declarations
class Dataset;
class DatasetHandle;

/**
 * Dataset type enumeration
 */
enum class DatasetType {
    None,
    CSV,
    TSV,
    ImageFolder,
    ImageCSV,           // Images in folder + labels in CSV file
    MNIST,
    FashionMNIST,
    CIFAR10,
    CIFAR100,
    HuggingFace,
    Kaggle,
    Custom
};

/**
 * Dataset split enumeration
 */
enum class DatasetSplit {
    Train,
    Validation,
    Test,
    All
};

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

/**
 * Dataset information structure
 */
struct DatasetInfo {
    std::string name;                    // Unique identifier
    std::string path;                    // Source path
    DatasetType type = DatasetType::None;
    std::vector<size_t> shape;           // Sample shape (e.g., [28, 28, 1])
    size_t num_samples = 0;
    size_t num_classes = 0;
    std::vector<std::string> class_names;

    // Split information
    size_t train_count = 0;
    size_t val_count = 0;
    size_t test_count = 0;
    float train_ratio = 0.8f;
    float val_ratio = 0.1f;
    float test_ratio = 0.1f;

    // Memory information
    size_t memory_usage = 0;             // Bytes
    size_t cache_usage = 0;              // Bytes in cache
    bool is_loaded = false;
    bool is_streaming = false;

    // Cache stats per dataset
    size_t cache_hits = 0;
    size_t cache_misses = 0;

    // Get formatted shape string
    std::string GetShapeString() const {
        if (shape.empty()) return "[]";
        std::string result = "[";
        for (size_t i = 0; i < shape.size(); i++) {
            if (i > 0) result += ", ";
            result += std::to_string(shape[i]);
        }
        result += "]";
        return result;
    }

    // Get formatted memory usage string
    std::string GetMemoryString() const {
        const char* units[] = {"B", "KB", "MB", "GB"};
        int unit_index = 0;
        double size = static_cast<double>(memory_usage);
        while (size >= 1024.0 && unit_index < 3) {
            size /= 1024.0;
            unit_index++;
        }
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unit_index]);
        return buffer;
    }
};

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

/**
 * Split configuration
 */
struct SplitConfig {
    float train_ratio = 0.8f;
    float val_ratio = 0.1f;
    float test_ratio = 0.1f;
    bool stratified = true;
    bool shuffle = true;
    int seed = 42;
};

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
 * Base Dataset interface
 */
class Dataset {
public:
    virtual ~Dataset() = default;

    // Core interface
    virtual size_t Size() const = 0;
    virtual std::pair<std::vector<float>, int> GetItem(size_t index) const = 0;
    virtual DatasetInfo GetInfo() const = 0;

    // Batch access
    virtual std::pair<std::vector<std::vector<float>>, std::vector<int>>
        GetBatch(const std::vector<size_t>& indices) const;

    // Split management
    virtual void SetSplit(const SplitConfig& config);
    virtual const std::vector<size_t>& GetTrainIndices() const { return train_indices_; }
    virtual const std::vector<size_t>& GetValIndices() const { return val_indices_; }
    virtual const std::vector<size_t>& GetTestIndices() const { return test_indices_; }

    // Get indices for a specific split
    virtual const std::vector<size_t>& GetSplitIndices(DatasetSplit split) const;

    // Streaming support
    virtual bool IsStreaming() const { return false; }
    virtual bool HasNext() const { return false; }
    virtual std::pair<std::vector<float>, int> GetNext() { return {{}, -1}; }
    virtual void ResetStream() {}

protected:
    std::vector<size_t> train_indices_;
    std::vector<size_t> val_indices_;
    std::vector<size_t> test_indices_;
    std::vector<size_t> all_indices_;
    SplitConfig split_config_;
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
    DatasetHandle LoadImageFolder(const std::string& path, const std::string& name = "");
    DatasetHandle LoadImageCSV(const std::string& image_folder, const std::string& csv_path,
                                const std::string& name = "", int target_width = 224, int target_height = 224,
                                size_t cache_size = 100);
    DatasetHandle LoadHuggingFace(const HuggingFaceConfig& config, const std::string& name = "");
    DatasetHandle LoadKaggle(const KaggleConfig& config, const std::string& name = "");
    DatasetHandle LoadCustom(const CustomConfig& config, const std::string& name = "");

    // Streaming dataset loading
    DatasetHandle LoadStreamingDataset(const std::string& path, const StreamingConfig& config, const std::string& name = "");

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

private:
    DataRegistry() = default;

    // Generate unique name if not provided
    std::string GenerateUniqueName(const std::string& base_name);

    // Dataset storage
    std::map<std::string, std::shared_ptr<Dataset>> datasets_;
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
};

} // namespace cyxwiz
