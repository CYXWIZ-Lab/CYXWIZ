#include "data_registry.h"
#include "annotation_manager.h"
#include "image_utils.h"
#include "../preprocessing/preprocessing_config.h"
#include "../transforms/transform.h"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <random>
#include <sstream>
#include <cstring>
#include <set>
#include <list>
#include <optional>
#include <unordered_map>
#include <numeric>
#include <chrono>
#include <ctime>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <nlohmann/json.hpp>

// stb_image for image loading (implementation in stb_image_impl.cpp)
#include <stb_image.h>

// HDF5 support (optional - only if HighFive is available)
#ifdef CYXWIZ_HAS_HDF5
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#endif

namespace fs = std::filesystem;

namespace cyxwiz {

// =============================================================================
// Dataset Base Class Implementation
// =============================================================================

std::pair<std::vector<std::vector<float>>, std::vector<int>>
Dataset::GetBatch(const std::vector<size_t>& indices) const {
    std::vector<std::vector<float>> samples;
    std::vector<int> labels;
    samples.reserve(indices.size());
    labels.reserve(indices.size());

    for (size_t idx : indices) {
        auto [sample, label] = GetItem(idx);
        samples.push_back(std::move(sample));
        labels.push_back(label);
    }

    return {std::move(samples), std::move(labels)};
}

void Dataset::SetSplit(const SplitConfig& config) {
    split_config_ = config;

    // Create all indices
    all_indices_.resize(Size());
    for (size_t i = 0; i < Size(); i++) {
        all_indices_[i] = i;
    }

    // Shuffle if requested
    if (config.shuffle) {
        std::mt19937 rng(config.seed);
        std::shuffle(all_indices_.begin(), all_indices_.end(), rng);
    }

    // Calculate split sizes
    size_t total = Size();
    size_t train_size = static_cast<size_t>(total * config.train_ratio);
    size_t val_size = static_cast<size_t>(total * config.val_ratio);
    size_t test_size = total - train_size - val_size;

    // Assign indices to splits
    train_indices_.clear();
    val_indices_.clear();
    test_indices_.clear();

    train_indices_.reserve(train_size);
    val_indices_.reserve(val_size);
    test_indices_.reserve(test_size);

    for (size_t i = 0; i < train_size; i++) {
        train_indices_.push_back(all_indices_[i]);
    }
    for (size_t i = train_size; i < train_size + val_size; i++) {
        val_indices_.push_back(all_indices_[i]);
    }
    for (size_t i = train_size + val_size; i < total; i++) {
        test_indices_.push_back(all_indices_[i]);
    }

    spdlog::info("Dataset split: train={}, val={}, test={}",
        train_indices_.size(), val_indices_.size(), test_indices_.size());
}

const std::vector<size_t>& Dataset::GetSplitIndices(DatasetSplit split) const {
    switch (split) {
        case DatasetSplit::Train: return train_indices_;
        case DatasetSplit::Validation: return val_indices_;
        case DatasetSplit::Test: return test_indices_;
        case DatasetSplit::All: return all_indices_;
        default: return all_indices_;
    }
}

// =============================================================================
// DatasetHandle Implementation
// =============================================================================

DatasetHandle::DatasetHandle(std::shared_ptr<Dataset> dataset, const std::string& name)
    : dataset_(std::move(dataset)), name_(name) {}

DatasetInfo DatasetHandle::GetInfo() const {
    if (!IsValid()) return DatasetInfo{};
    return dataset_->GetInfo();
}

size_t DatasetHandle::Size() const {
    if (!IsValid()) return 0;
    return dataset_->Size();
}

size_t DatasetHandle::Size(DatasetSplit split) const {
    if (!IsValid()) return 0;
    return dataset_->GetSplitIndices(split).size();
}

std::pair<std::vector<float>, int> DatasetHandle::GetSample(size_t index) const {
    if (!IsValid()) return {{}, -1};
    return dataset_->GetItem(index);
}

std::pair<std::vector<std::vector<float>>, std::vector<int>>
DatasetHandle::GetBatch(const std::vector<size_t>& indices) const {
    if (!IsValid()) return {{}, {}};
    return dataset_->GetBatch(indices);
}

const std::vector<size_t>& DatasetHandle::GetTrainIndices() const {
    static std::vector<size_t> empty;
    if (!IsValid()) return empty;
    return dataset_->GetTrainIndices();
}

const std::vector<size_t>& DatasetHandle::GetValIndices() const {
    static std::vector<size_t> empty;
    if (!IsValid()) return empty;
    return dataset_->GetValIndices();
}

const std::vector<size_t>& DatasetHandle::GetTestIndices() const {
    static std::vector<size_t> empty;
    if (!IsValid()) return empty;
    return dataset_->GetTestIndices();
}

const std::vector<size_t>& DatasetHandle::GetSplitIndices(DatasetSplit split) const {
    static std::vector<size_t> empty;
    if (!IsValid()) return empty;
    return dataset_->GetSplitIndices(split);
}

void DatasetHandle::ApplySplit(const SplitConfig& config) {
    if (IsValid()) {
        dataset_->SetSplit(config);
    }
}

// =============================================================================
// Concrete Dataset Implementations
// =============================================================================

/**
 * MNIST Dataset Implementation
 */
class MNISTDataset : public Dataset {
public:
    MNISTDataset(const std::string& path) : path_(path) {
        LoadData();
    }

    size_t Size() const override { return images_.size(); }

    std::pair<std::vector<float>, int> GetItem(size_t index) const override {
        if (index >= images_.size()) return {{}, -1};
        return {images_[index], static_cast<int>(labels_[index])};
    }

    DatasetInfo GetInfo() const override {
        DatasetInfo info;
        info.name = "mnist";
        info.path = path_;
        info.type = DatasetType::MNIST;
        info.shape = {28, 28, 1};
        info.num_samples = images_.size();
        info.num_classes = 10;
        info.class_names = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
        info.train_count = train_indices_.size();
        info.val_count = val_indices_.size();
        info.test_count = test_indices_.size();
        info.memory_usage = images_.size() * 28 * 28 * sizeof(float);
        info.is_loaded = !images_.empty();
        return info;
    }

private:
    void LoadData() {
        // Try to find MNIST files
        std::string images_file = path_ + "/train-images-idx3-ubyte";
        std::string labels_file = path_ + "/train-labels-idx1-ubyte";

        // Alternative naming
        if (!fs::exists(images_file)) {
            images_file = path_ + "/train-images.idx3-ubyte";
            labels_file = path_ + "/train-labels.idx1-ubyte";
        }

        if (!fs::exists(images_file) || !fs::exists(labels_file)) {
            spdlog::error("MNIST files not found in: {}", path_);
            return;
        }

        // Load images
        std::ifstream img_stream(images_file, std::ios::binary);
        if (!img_stream) {
            spdlog::error("Failed to open MNIST images file");
            return;
        }

        // Read header (big-endian)
        auto read_int = [&img_stream]() -> uint32_t {
            uint32_t val;
            img_stream.read(reinterpret_cast<char*>(&val), 4);
            // Convert from big-endian
            return ((val & 0xFF) << 24) | ((val & 0xFF00) << 8) |
                   ((val & 0xFF0000) >> 8) | ((val & 0xFF000000) >> 24);
        };

        uint32_t magic = read_int();
        uint32_t num_images = read_int();
        uint32_t rows = read_int();
        uint32_t cols = read_int();

        if (magic != 2051) {
            spdlog::error("Invalid MNIST images file magic number");
            return;
        }

        // Read image data
        images_.reserve(num_images);
        for (uint32_t i = 0; i < num_images; i++) {
            std::vector<float> image(rows * cols);
            for (uint32_t j = 0; j < rows * cols; j++) {
                uint8_t pixel;
                img_stream.read(reinterpret_cast<char*>(&pixel), 1);
                image[j] = pixel / 255.0f;  // Normalize to [0, 1]
            }
            images_.push_back(std::move(image));
        }

        // Load labels
        std::ifstream lbl_stream(labels_file, std::ios::binary);
        if (!lbl_stream) {
            spdlog::error("Failed to open MNIST labels file");
            images_.clear();
            return;
        }

        // Skip header
        lbl_stream.seekg(8);

        labels_.reserve(num_images);
        for (uint32_t i = 0; i < num_images; i++) {
            uint8_t label;
            lbl_stream.read(reinterpret_cast<char*>(&label), 1);
            labels_.push_back(label);
        }

        spdlog::info("Loaded MNIST dataset: {} images", images_.size());

        // Apply default split
        SetSplit(SplitConfig{});
    }

    std::string path_;
    std::vector<std::vector<float>> images_;
    std::vector<uint8_t> labels_;
};

/**
 * CIFAR-10 Dataset Implementation
 */
class CIFAR10Dataset : public Dataset {
public:
    CIFAR10Dataset(const std::string& path) : path_(path) {
        class_names_ = {"airplane", "automobile", "bird", "cat", "deer",
                        "dog", "frog", "horse", "ship", "truck"};
        LoadData();
    }

    size_t Size() const override { return images_.size(); }

    std::pair<std::vector<float>, int> GetItem(size_t index) const override {
        if (index >= images_.size()) return {{}, -1};
        return {images_[index], static_cast<int>(labels_[index])};
    }

    DatasetInfo GetInfo() const override {
        DatasetInfo info;
        info.name = "cifar10";
        info.path = path_;
        info.type = DatasetType::CIFAR10;
        info.shape = {32, 32, 3};
        info.num_samples = images_.size();
        info.num_classes = 10;
        info.class_names = class_names_;
        info.train_count = train_indices_.size();
        info.val_count = val_indices_.size();
        info.test_count = test_indices_.size();
        info.memory_usage = images_.size() * 32 * 32 * 3 * sizeof(float);
        info.is_loaded = !images_.empty();
        return info;
    }

private:
    void LoadData() {
        // Load all batch files
        for (int batch = 1; batch <= 5; batch++) {
            std::string batch_file = path_ + "/data_batch_" + std::to_string(batch) + ".bin";
            if (!fs::exists(batch_file)) {
                spdlog::warn("CIFAR-10 batch file not found: {}", batch_file);
                continue;
            }

            std::ifstream stream(batch_file, std::ios::binary);
            if (!stream) continue;

            // Each sample: 1 byte label + 3072 bytes image (32*32*3)
            const int samples_per_batch = 10000;
            const int image_size = 32 * 32 * 3;

            for (int i = 0; i < samples_per_batch; i++) {
                uint8_t label;
                stream.read(reinterpret_cast<char*>(&label), 1);
                labels_.push_back(label);

                std::vector<float> image(image_size);
                std::vector<uint8_t> raw(image_size);
                stream.read(reinterpret_cast<char*>(raw.data()), image_size);

                // Convert to float and normalize
                for (int j = 0; j < image_size; j++) {
                    image[j] = raw[j] / 255.0f;
                }
                images_.push_back(std::move(image));
            }
        }

        if (!images_.empty()) {
            spdlog::info("Loaded CIFAR-10 dataset: {} images", images_.size());
            SetSplit(SplitConfig{});
        } else {
            spdlog::error("Failed to load CIFAR-10 dataset from: {}", path_);
        }
    }

    std::string path_;
    std::vector<std::vector<float>> images_;
    std::vector<uint8_t> labels_;
    std::vector<std::string> class_names_;
};

/**
 * CSV Dataset Implementation
 */
class CSVDataset : public Dataset {
public:
    CSVDataset(const std::string& path) : path_(path) {
        LoadData();
    }

    size_t Size() const override { return samples_.size(); }

    std::pair<std::vector<float>, int> GetItem(size_t index) const override {
        if (index >= samples_.size()) return {{}, -1};
        return {samples_[index], labels_[index]};
    }

    DatasetInfo GetInfo() const override {
        DatasetInfo info;
        info.name = fs::path(path_).stem().string();
        info.path = path_;
        info.type = DatasetType::CSV;
        info.shape = num_features_ > 0 ? std::vector<size_t>{static_cast<size_t>(num_features_)} : std::vector<size_t>{};
        info.num_samples = samples_.size();
        info.num_classes = num_classes_;
        info.train_count = train_indices_.size();
        info.val_count = val_indices_.size();
        info.test_count = test_indices_.size();
        info.memory_usage = samples_.size() * num_features_ * sizeof(float);
        info.is_loaded = !samples_.empty();
        return info;
    }

    std::vector<std::string> GetColumnNames() const override { return column_names_; }

private:
    void LoadData() {
        std::ifstream file(path_);
        if (!file) {
            spdlog::error("Failed to open CSV file: {}", path_);
            return;
        }

        std::string line;
        bool first_line = true;
        std::set<int> unique_labels;

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            std::vector<std::string> tokens;
            std::stringstream ss(line);
            std::string token;

            while (std::getline(ss, token, ',')) {
                // Trim whitespace
                token.erase(0, token.find_first_not_of(" \t\r\n"));
                token.erase(token.find_last_not_of(" \t\r\n") + 1);
                tokens.push_back(token);
            }

            if (tokens.empty()) continue;

            // Check if first line is header
            if (first_line) {
                first_line = false;
                try {
                    (void)std::stof(tokens[0]);  // Just checking if it's numeric
                } catch (...) {
                    // First line is header
                    column_names_ = tokens;
                    continue;
                }
            }

            // Parse values (last column is label)
            std::vector<float> sample;
            sample.reserve(tokens.size() - 1);

            for (size_t i = 0; i < tokens.size() - 1; i++) {
                try {
                    sample.push_back(std::stof(tokens[i]));
                } catch (...) {
                    sample.push_back(0.0f);
                }
            }

            int label = 0;
            try {
                label = std::stoi(tokens.back());
            } catch (...) {
                label = 0;
            }

            samples_.push_back(std::move(sample));
            labels_.push_back(label);
            unique_labels.insert(label);
        }

        num_features_ = samples_.empty() ? 0 : static_cast<int>(samples_[0].size());
        num_classes_ = static_cast<int>(unique_labels.size());

        if (!samples_.empty()) {
            spdlog::info("Loaded CSV dataset: {} samples, {} features, {} classes",
                samples_.size(), num_features_, num_classes_);
            SetSplit(SplitConfig{});
        }
    }

    std::string path_;
    std::vector<std::vector<float>> samples_;
    std::vector<int> labels_;
    std::vector<std::string> column_names_;
    int num_features_ = 0;
    int num_classes_ = 0;
};

// =============================================================================
// HDF5 Dataset Implementation
// =============================================================================
#ifdef CYXWIZ_HAS_HDF5

/**
 * HDF5 Dataset Implementation with Lazy Loading
 *
 * Features:
 * - Lazy loading: Only loads chunks on demand
 * - LRU cache: Keeps recently accessed chunks in memory
 * - Memory efficient: Suitable for large datasets (100GB+)
 * - Thread-safe: Mutex-protected file access and cache
 *
 * Supports common HDF5 layouts: /images+/labels, /data+/labels, /X+/y, etc.
 */
class HDF5Dataset : public Dataset {
public:
    HDF5Dataset(const std::string& path, const HDF5Config& config = {})
        : path_(path), config_(config) {
        Initialize();
    }

    ~HDF5Dataset() {
        // Stop prefetch threads
        StopPrefetch();
        // File is closed automatically when unique_ptr is destroyed
    }

    size_t Size() const override { return num_samples_; }

    std::pair<std::vector<float>, int> GetItem(size_t index) const override {
        if (index >= num_samples_) return {{}, -1};

        // Get label (labels are always in memory - they're small)
        int label = labels_.empty() ? 0 : labels_[index];

        // If not using lazy loading, return directly from memory
        if (!use_lazy_loading_) {
            return {eager_samples_[index], label};
        }

        // Trigger prefetch for upcoming chunks
        if (config_.prefetch_enabled && !prefetch_threads_.empty()) {
            TriggerPrefetch(index);
        }

        // Lazy loading path: get sample from chunk cache
        std::vector<float> sample = GetSampleFromCache(index);
        return {std::move(sample), label};
    }

    DatasetInfo GetInfo() const override {
        DatasetInfo info;
        info.name = fs::path(path_).stem().string();
        info.path = path_;
        info.type = DatasetType::HDF5;
        info.shape = sample_shape_;
        info.num_samples = num_samples_;
        info.num_classes = num_classes_;
        info.train_count = train_indices_.size();
        info.val_count = val_indices_.size();
        info.test_count = test_indices_.size();

        // Memory usage depends on loading mode
        if (use_lazy_loading_) {
            // Only cache is in memory
            std::lock_guard<std::mutex> lock(cache_mutex_);
            info.memory_usage = chunk_cache_.size() * config_.chunk_size * sample_size_ * sizeof(float);
            info.cache_usage = info.memory_usage;
            info.is_streaming = true;
        } else {
            info.memory_usage = num_samples_ * sample_size_ * sizeof(float);
            info.cache_usage = 0;
            info.is_streaming = false;
        }
        info.is_loaded = true;

        // Cache statistics
        info.cache_hits = cache_hits_;
        info.cache_misses = cache_misses_;

        return info;
    }

    // Check if using lazy loading
    bool IsLazyLoading() const { return use_lazy_loading_; }

    // Get cache statistics
    size_t GetCacheHits() const { return cache_hits_; }
    size_t GetCacheMisses() const { return cache_misses_; }
    size_t GetCachedChunks() const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return chunk_cache_.size();
    }

    // Clear cache (useful for memory pressure)
    void ClearCache() const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        chunk_cache_.clear();
        lru_order_.clear();
        cache_hits_ = 0;
        cache_misses_ = 0;
    }

private:
    // Chunk data structure
    struct Chunk {
        std::vector<std::vector<float>> samples;  // samples in this chunk
        size_t start_index;                        // first sample index
        size_t count;                              // number of samples
    };

    bool DetectSchema(HighFive::File& file) {
        // Common data path names (priority order)
        const std::vector<std::string> data_names = {
            "images", "data", "features", "X", "x", "inputs",
            "x_train", "train/images", "train/data"
        };

        // Common label path names
        const std::vector<std::string> label_names = {
            "labels", "targets", "y", "Y", "outputs",
            "y_train", "train/labels", "train/targets"
        };

        // Try to find data dataset
        for (const auto& name : data_names) {
            if (file.exist(name)) {
                try {
                    auto ds = file.getDataSet(name);
                    auto dims = ds.getDimensions();
                    if (dims.size() >= 1) {
                        data_path_ = name;
                        break;
                    }
                } catch (...) {}
            }
        }

        // Try to find labels dataset
        for (const auto& name : label_names) {
            if (file.exist(name)) {
                try {
                    auto ds = file.getDataSet(name);
                    auto dims = ds.getDimensions();
                    if (dims.size() == 1) {
                        label_path_ = name;
                        break;
                    }
                } catch (...) {}
            }
        }

        return !data_path_.empty();
    }

    void Initialize() {
        try {
            HighFive::File file(path_, HighFive::File::ReadOnly);

            // Use config paths or auto-detect
            if (config_.auto_detect || config_.data_path.empty()) {
                if (!DetectSchema(file)) {
                    spdlog::error("HDF5: Could not auto-detect schema in {}", path_);
                    return;
                }
            } else {
                data_path_ = config_.data_path;
                label_path_ = config_.label_path;
            }

            // Get data dataset info
            auto data_ds = file.getDataSet(data_path_);
            dims_ = data_ds.getDimensions();
            dtype_is_uint8_ = (data_ds.getDataType() == HighFive::AtomicType<uint8_t>());
            dtype_is_float_ = (data_ds.getDataType() == HighFive::AtomicType<float>());
            dtype_is_double_ = (data_ds.getDataType() == HighFive::AtomicType<double>());
            dtype_is_int32_ = (data_ds.getDataType() == HighFive::AtomicType<int32_t>());

            if (dims_.empty()) {
                spdlog::error("HDF5: Empty dataset at {}", data_path_);
                return;
            }

            num_samples_ = dims_[0];

            // Calculate sample shape and size
            sample_shape_.clear();
            sample_size_ = 1;
            for (size_t i = 1; i < dims_.size(); i++) {
                sample_shape_.push_back(dims_[i]);
                sample_size_ *= dims_[i];
            }

            // Handle 1D data (tabular)
            if (dims_.size() == 1) {
                sample_shape_ = {1};
                sample_size_ = 1;
            }

            // Detect NCHW format for 4D image data
            is_nchw_ = false;
            if (dims_.size() == 4) {
                // dims_ = [N, dim1, dim2, dim3]
                // NCHW: [N, C, H, W] where C is small (1, 3, 4) and H, W are larger
                // NHWC: [N, H, W, C] where C is small (1, 3, 4) and H, W are larger
                size_t dim1 = dims_[1];  // C in NCHW, H in NHWC
                size_t dim2 = dims_[2];  // H in NCHW, W in NHWC
                size_t dim3 = dims_[3];  // W in NCHW, C in NHWC

                bool dim1_is_channel = (dim1 == 1 || dim1 == 3 || dim1 == 4) && dim2 > 4 && dim3 > 4;
                bool dim3_is_channel = (dim3 == 1 || dim3 == 3 || dim3 == 4) && dim1 > 4 && dim2 > 4;

                if (config_.force_nchw) {
                    // User explicitly said it's NCHW
                    is_nchw_ = true;
                    spdlog::info("HDF5: Forcing NCHW->NHWC transpose");
                } else if (config_.auto_detect_nchw && dim1_is_channel && !dim3_is_channel) {
                    // Auto-detected NCHW format
                    is_nchw_ = true;
                    spdlog::info("HDF5: Auto-detected NCHW format [N,{},{},{}], will transpose to NHWC",
                                 dim1, dim2, dim3);
                }

                // Adjust sample_shape_ if NCHW -> NHWC
                if (is_nchw_) {
                    // Original: [C, H, W] -> Transposed: [H, W, C]
                    sample_shape_ = {dim2, dim3, dim1};  // [H, W, C]
                }
            }

            // Load labels (always load into memory - they're small)
            LoadLabels(file);

            // Decide whether to use lazy loading
            use_lazy_loading_ = config_.lazy_loading &&
                                (num_samples_ > config_.lazy_threshold);

            if (use_lazy_loading_) {
                // Lazy loading: just store metadata, don't load data
                spdlog::info("HDF5 dataset initialized (LAZY): {} samples, shape [{}], {} classes, "
                             "chunk_size={}, cache_size={}",
                    num_samples_, GetShapeString(), num_classes_,
                    config_.chunk_size, config_.chunk_cache_size);

                // Start prefetch threads if enabled
                if (config_.prefetch_enabled && config_.prefetch_threads > 0) {
                    StartPrefetch();
                }
            } else {
                // Eager loading: load all data into memory
                LoadAllData(file);
                spdlog::info("HDF5 dataset loaded (EAGER): {} samples, shape [{}], {} classes",
                    num_samples_, GetShapeString(), num_classes_);
            }

            if (num_samples_ > 0) {
                SetSplit(SplitConfig{});
            }

        } catch (const std::exception& e) {
            spdlog::error("HDF5 initialization error: {}", e.what());
        }
    }

    void LoadLabels(HighFive::File& file) {
        std::set<int> unique_labels;

        if (!label_path_.empty() && file.exist(label_path_)) {
            auto label_ds = file.getDataSet(label_path_);
            auto label_dtype = label_ds.getDataType();

            if (label_dtype == HighFive::AtomicType<int32_t>()) {
                label_ds.read(labels_);
            } else if (label_dtype == HighFive::AtomicType<int64_t>()) {
                std::vector<int64_t> labels64;
                label_ds.read(labels64);
                labels_.resize(labels64.size());
                for (size_t i = 0; i < labels64.size(); i++) {
                    labels_[i] = static_cast<int>(labels64[i]);
                }
            } else if (label_dtype == HighFive::AtomicType<uint8_t>()) {
                std::vector<uint8_t> labels8;
                label_ds.read(labels8);
                labels_.resize(labels8.size());
                for (size_t i = 0; i < labels8.size(); i++) {
                    labels_[i] = static_cast<int>(labels8[i]);
                }
            } else if (label_dtype == HighFive::AtomicType<float>()) {
                std::vector<float> labelsf;
                label_ds.read(labelsf);
                labels_.resize(labelsf.size());
                for (size_t i = 0; i < labelsf.size(); i++) {
                    labels_[i] = static_cast<int>(labelsf[i]);
                }
            }

            for (int label : labels_) {
                unique_labels.insert(label);
            }
        }

        num_classes_ = unique_labels.empty() ? 0 : static_cast<int>(unique_labels.size());
    }

    void LoadAllData(HighFive::File& file) {
        auto data_ds = file.getDataSet(data_path_);
        eager_samples_.resize(num_samples_);

        float scale = (config_.normalize && dtype_is_uint8_) ? 1.0f / 255.0f : 1.0f;

        // Build selection parameters
        std::vector<size_t> offset(dims_.size(), 0);
        std::vector<size_t> count(dims_.size(), 1);
        for (size_t d = 1; d < dims_.size(); d++) {
            count[d] = dims_[d];
        }

        for (size_t i = 0; i < num_samples_; i++) {
            offset[0] = i;
            eager_samples_[i].resize(sample_size_);
            ReadSampleInto(data_ds, offset, count, eager_samples_[i], scale);
        }
    }

    void ReadSampleInto(HighFive::DataSet& data_ds,
                        const std::vector<size_t>& offset,
                        const std::vector<size_t>& count,
                        std::vector<float>& out,
                        float scale) const {
        auto selection = data_ds.select(offset, count);

        if (dims_.size() == 2) {
            // 2D: [N, features]
            if (dtype_is_float_) {
                selection.read(out);
            } else if (dtype_is_uint8_) {
                std::vector<uint8_t> raw(sample_size_);
                selection.read(raw);
                for (size_t j = 0; j < sample_size_; j++) {
                    out[j] = static_cast<float>(raw[j]) * scale;
                }
            } else if (dtype_is_double_) {
                std::vector<double> raw(sample_size_);
                selection.read(raw);
                for (size_t j = 0; j < sample_size_; j++) {
                    out[j] = static_cast<float>(raw[j]);
                }
            } else if (dtype_is_int32_) {
                std::vector<int32_t> raw(sample_size_);
                selection.read(raw);
                for (size_t j = 0; j < sample_size_; j++) {
                    out[j] = static_cast<float>(raw[j]);
                }
            }
        } else if (dims_.size() == 3) {
            // 3D: [N, H, W]
            ReadAndFlatten3D(selection, out, scale);
        } else if (dims_.size() == 4) {
            // 4D: [N, H, W, C]
            ReadAndFlatten4D(selection, out, scale);
        }
    }

    void ReadAndFlatten3D(HighFive::Selection& selection, std::vector<float>& out, float scale) const {
        size_t k = 0;
        if (dtype_is_uint8_) {
            std::vector<std::vector<std::vector<uint8_t>>> data_3d;
            selection.read(data_3d);
            for (const auto& plane : data_3d) {
                for (const auto& row : plane) {
                    for (auto val : row) {
                        out[k++] = static_cast<float>(val) * scale;
                    }
                }
            }
        } else if (dtype_is_float_) {
            std::vector<std::vector<std::vector<float>>> data_3d;
            selection.read(data_3d);
            for (const auto& plane : data_3d) {
                for (const auto& row : plane) {
                    for (auto val : row) {
                        out[k++] = val;
                    }
                }
            }
        } else if (dtype_is_double_) {
            std::vector<std::vector<std::vector<double>>> data_3d;
            selection.read(data_3d);
            for (const auto& plane : data_3d) {
                for (const auto& row : plane) {
                    for (auto val : row) {
                        out[k++] = static_cast<float>(val);
                    }
                }
            }
        } else if (dtype_is_int32_) {
            std::vector<std::vector<std::vector<int32_t>>> data_3d;
            selection.read(data_3d);
            for (const auto& plane : data_3d) {
                for (const auto& row : plane) {
                    for (auto val : row) {
                        out[k++] = static_cast<float>(val);
                    }
                }
            }
        }
    }

    void ReadAndFlatten4D(HighFive::Selection& selection, std::vector<float>& out, float scale) const {
        size_t k = 0;

        // Helper lambda to flatten with optional NCHW->NHWC transpose
        auto flatten_4d = [&](auto& data_4d, float s) {
            for (const auto& vol : data_4d) {
                if (is_nchw_) {
                    // NCHW -> NHWC transpose: vol is [C, H, W], output as [H, W, C]
                    size_t C = vol.size();
                    size_t H = C > 0 ? vol[0].size() : 0;
                    size_t W = (C > 0 && H > 0) ? vol[0][0].size() : 0;
                    for (size_t h = 0; h < H; h++) {
                        for (size_t w = 0; w < W; w++) {
                            for (size_t c = 0; c < C; c++) {
                                out[k++] = static_cast<float>(vol[c][h][w]) * s;
                            }
                        }
                    }
                } else {
                    // NHWC: vol is [H, W, C], output as-is
                    for (const auto& plane : vol) {
                        for (const auto& row : plane) {
                            for (auto val : row) {
                                out[k++] = static_cast<float>(val) * s;
                            }
                        }
                    }
                }
            }
        };

        if (dtype_is_uint8_) {
            std::vector<std::vector<std::vector<std::vector<uint8_t>>>> data_4d;
            selection.read(data_4d);
            flatten_4d(data_4d, scale);
        } else if (dtype_is_float_) {
            std::vector<std::vector<std::vector<std::vector<float>>>> data_4d;
            selection.read(data_4d);
            flatten_4d(data_4d, 1.0f);
        } else if (dtype_is_double_) {
            std::vector<std::vector<std::vector<std::vector<double>>>> data_4d;
            selection.read(data_4d);
            flatten_4d(data_4d, 1.0f);
        } else if (dtype_is_int32_) {
            std::vector<std::vector<std::vector<std::vector<int32_t>>>> data_4d;
            selection.read(data_4d);
            flatten_4d(data_4d, 1.0f);
        }
    }

    std::vector<float> GetSampleFromCache(size_t index) const {
        size_t chunk_idx = index / config_.chunk_size;
        size_t local_idx = index % config_.chunk_size;

        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Check if chunk is in cache
        auto it = chunk_cache_.find(chunk_idx);
        if (it != chunk_cache_.end()) {
            // Cache hit - update LRU order
            cache_hits_++;
            lru_order_.remove(chunk_idx);
            lru_order_.push_front(chunk_idx);
            return it->second.samples[local_idx];
        }

        // Cache miss - need to load chunk
        cache_misses_++;

        // Evict oldest chunks if cache is full
        while (chunk_cache_.size() >= config_.chunk_cache_size && !lru_order_.empty()) {
            size_t evict_idx = lru_order_.back();
            lru_order_.pop_back();
            chunk_cache_.erase(evict_idx);
        }

        // Load chunk from file
        Chunk chunk = LoadChunk(chunk_idx);
        chunk_cache_[chunk_idx] = std::move(chunk);
        lru_order_.push_front(chunk_idx);

        return chunk_cache_[chunk_idx].samples[local_idx];
    }

    Chunk LoadChunk(size_t chunk_idx) const {
        Chunk chunk;
        chunk.start_index = chunk_idx * config_.chunk_size;
        chunk.count = std::min(config_.chunk_size, num_samples_ - chunk.start_index);

        float scale = (config_.normalize && dtype_is_uint8_) ? 1.0f / 255.0f : 1.0f;

        try {
            // HDF5 is not thread-safe, so we need to serialize file access
            std::lock_guard<std::mutex> lock(file_mutex_);

            // Open file for reading
            HighFive::File file(path_, HighFive::File::ReadOnly);
            auto data_ds = file.getDataSet(data_path_);

            chunk.samples.resize(chunk.count);

            // Build selection parameters for the chunk
            std::vector<size_t> offset(dims_.size(), 0);
            std::vector<size_t> count(dims_.size(), 1);
            for (size_t d = 1; d < dims_.size(); d++) {
                count[d] = dims_[d];
            }

            for (size_t i = 0; i < chunk.count; i++) {
                offset[0] = chunk.start_index + i;
                chunk.samples[i].resize(sample_size_);
                ReadSampleInto(data_ds, offset, count, chunk.samples[i], scale);
            }

        } catch (const std::exception& e) {
            spdlog::error("HDF5: Failed to load chunk {}: {}", chunk_idx, e.what());
        }

        return chunk;
    }

    std::string GetShapeString() const {
        std::string s;
        for (size_t i = 0; i < sample_shape_.size(); i++) {
            if (i > 0) s += ", ";
            s += std::to_string(sample_shape_[i]);
        }
        return s;
    }

    // ========== Prefetch Methods ==========

    void StartPrefetch() const {
        prefetch_stop_ = false;
        size_t num_threads = config_.prefetch_threads;
        spdlog::info("HDF5: Starting {} prefetch threads", num_threads);

        for (size_t i = 0; i < num_threads; i++) {
            prefetch_threads_.emplace_back([this]() { PrefetchWorker(); });
        }
    }

    void StopPrefetch() const {
        if (prefetch_threads_.empty()) return;

        // Signal threads to stop
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex_);
            prefetch_stop_ = true;
        }
        prefetch_cv_.notify_all();

        // Wait for threads to finish
        for (auto& t : prefetch_threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
        prefetch_threads_.clear();
    }

    void TriggerPrefetch(size_t current_index) const {
        size_t current_chunk = current_index / config_.chunk_size;
        size_t total_chunks = (num_samples_ + config_.chunk_size - 1) / config_.chunk_size;

        std::lock_guard<std::mutex> lock(prefetch_mutex_);

        // Queue chunks ahead of current position
        for (size_t i = 1; i <= config_.prefetch_ahead; i++) {
            size_t chunk_to_prefetch = current_chunk + i;
            if (chunk_to_prefetch >= total_chunks) break;

            // Skip if already cached or being loaded
            {
                std::lock_guard<std::mutex> cache_lock(cache_mutex_);
                if (chunk_cache_.find(chunk_to_prefetch) != chunk_cache_.end()) continue;
            }
            if (prefetch_in_progress_.find(chunk_to_prefetch) != prefetch_in_progress_.end()) continue;

            prefetch_queue_.push(chunk_to_prefetch);
            prefetch_in_progress_.insert(chunk_to_prefetch);
        }

        if (!prefetch_queue_.empty()) {
            prefetch_cv_.notify_one();
        }
    }

    void PrefetchWorker() const {
        while (true) {
            size_t chunk_idx;

            // Wait for work
            {
                std::unique_lock<std::mutex> lock(prefetch_mutex_);
                prefetch_cv_.wait(lock, [this]() {
                    return prefetch_stop_ || !prefetch_queue_.empty();
                });

                if (prefetch_stop_ && prefetch_queue_.empty()) {
                    return;
                }

                if (prefetch_queue_.empty()) continue;

                chunk_idx = prefetch_queue_.front();
                prefetch_queue_.pop();
            }

            // Load chunk (outside of lock)
            Chunk chunk = LoadChunk(chunk_idx);

            // Store in cache
            {
                std::lock_guard<std::mutex> cache_lock(cache_mutex_);

                // Evict if cache is full
                while (chunk_cache_.size() >= config_.chunk_cache_size && !lru_order_.empty()) {
                    size_t evict_idx = lru_order_.back();
                    lru_order_.pop_back();
                    chunk_cache_.erase(evict_idx);
                }

                chunk_cache_[chunk_idx] = std::move(chunk);
                lru_order_.push_front(chunk_idx);
            }

            // Remove from in-progress set
            {
                std::lock_guard<std::mutex> lock(prefetch_mutex_);
                prefetch_in_progress_.erase(chunk_idx);
            }
        }
    }

    // Configuration
    std::string path_;
    HDF5Config config_;
    std::string data_path_;
    std::string label_path_;

    // Dataset metadata
    std::vector<size_t> dims_;
    std::vector<size_t> sample_shape_;
    size_t num_samples_ = 0;
    size_t sample_size_ = 0;
    int num_classes_ = 0;

    // Data type flags
    bool dtype_is_uint8_ = false;
    bool dtype_is_float_ = false;
    bool dtype_is_double_ = false;
    bool dtype_is_int32_ = false;

    // Loading mode
    bool use_lazy_loading_ = false;

    // NCHW transpose flag (for PyTorch-style datasets)
    bool is_nchw_ = false;

    // Eager loading storage (used when lazy_loading is disabled)
    std::vector<std::vector<float>> eager_samples_;

    // Labels (always loaded into memory)
    std::vector<int> labels_;

    // Lazy loading: chunk cache with LRU eviction
    mutable std::mutex cache_mutex_;
    mutable std::mutex file_mutex_;  // Serialize HDF5 file access (not thread-safe)
    mutable std::unordered_map<size_t, Chunk> chunk_cache_;
    mutable std::list<size_t> lru_order_;  // front = most recent, back = oldest
    mutable size_t cache_hits_ = 0;
    mutable size_t cache_misses_ = 0;

    // Prefetch thread pool for parallel I/O
    mutable std::vector<std::thread> prefetch_threads_;
    mutable std::queue<size_t> prefetch_queue_;
    mutable std::mutex prefetch_mutex_;
    mutable std::condition_variable prefetch_cv_;
    mutable std::atomic<bool> prefetch_stop_{false};
    mutable std::set<size_t> prefetch_in_progress_;  // Chunks currently being loaded
};

#endif // CYXWIZ_HAS_HDF5

/**
 * TSV Dataset Implementation (Tab-Separated Values)
 */
class TSVDataset : public Dataset {
public:
    TSVDataset(const std::string& path) : path_(path) {
        LoadData();
    }

    size_t Size() const override { return samples_.size(); }

    std::pair<std::vector<float>, int> GetItem(size_t index) const override {
        if (index >= samples_.size()) return {{}, -1};
        return {samples_[index], labels_[index]};
    }

    DatasetInfo GetInfo() const override {
        DatasetInfo info;
        info.name = fs::path(path_).stem().string();
        info.path = path_;
        info.type = DatasetType::TSV;
        info.shape = num_features_ > 0 ? std::vector<size_t>{static_cast<size_t>(num_features_)} : std::vector<size_t>{};
        info.num_samples = samples_.size();
        info.num_classes = num_classes_;
        info.train_count = train_indices_.size();
        info.val_count = val_indices_.size();
        info.test_count = test_indices_.size();
        info.memory_usage = samples_.size() * num_features_ * sizeof(float);
        info.is_loaded = !samples_.empty();
        return info;
    }

    std::vector<std::string> GetColumnNames() const override { return column_names_; }

private:
    void LoadData() {
        std::ifstream file(path_);
        if (!file) {
            spdlog::error("Failed to open TSV file: {}", path_);
            return;
        }

        std::string line;
        bool first_line = true;
        std::set<int> unique_labels;

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            std::vector<std::string> tokens;
            std::stringstream ss(line);
            std::string token;

            while (std::getline(ss, token, '\t')) {
                token.erase(0, token.find_first_not_of(" \r\n"));
                token.erase(token.find_last_not_of(" \r\n") + 1);
                tokens.push_back(token);
            }

            if (tokens.empty()) continue;

            if (first_line) {
                first_line = false;
                try {
                    (void)std::stof(tokens[0]);  // Just checking if it's numeric
                } catch (...) {
                    column_names_ = tokens;
                    continue;
                }
            }

            std::vector<float> sample;
            sample.reserve(tokens.size() - 1);

            for (size_t i = 0; i < tokens.size() - 1; i++) {
                try {
                    sample.push_back(std::stof(tokens[i]));
                } catch (...) {
                    sample.push_back(0.0f);
                }
            }

            int label = 0;
            try {
                label = std::stoi(tokens.back());
            } catch (...) {
                label = 0;
            }

            samples_.push_back(std::move(sample));
            labels_.push_back(label);
            unique_labels.insert(label);
        }

        num_features_ = samples_.empty() ? 0 : static_cast<int>(samples_[0].size());
        num_classes_ = static_cast<int>(unique_labels.size());

        if (!samples_.empty()) {
            spdlog::info("Loaded TSV dataset: {} samples, {} features, {} classes",
                samples_.size(), num_features_, num_classes_);
            SetSplit(SplitConfig{});
        }
    }

    std::string path_;
    std::vector<std::vector<float>> samples_;
    std::vector<int> labels_;
    std::vector<std::string> column_names_;
    int num_features_ = 0;
    int num_classes_ = 0;
};

/**
 * JSON Dataset Implementation
 * Supports JSON files with data/labels arrays or line-delimited JSON
 */
class JSONDataset : public Dataset {
public:
    JSONDataset(const std::string& path) : path_(path) {
        LoadData();
    }

    size_t Size() const override { return samples_.size(); }

    std::pair<std::vector<float>, int> GetItem(size_t index) const override {
        if (index >= samples_.size()) return {{}, -1};
        return {samples_[index], labels_[index]};
    }

    DatasetInfo GetInfo() const override {
        DatasetInfo info;
        info.name = fs::path(path_).stem().string();
        info.path = path_;
        info.type = DatasetType::JSON;
        info.shape = num_features_ > 0 ? std::vector<size_t>{static_cast<size_t>(num_features_)} : std::vector<size_t>{};
        info.num_samples = samples_.size();
        info.num_classes = num_classes_;
        info.train_count = train_indices_.size();
        info.val_count = val_indices_.size();
        info.test_count = test_indices_.size();
        info.memory_usage = samples_.size() * num_features_ * sizeof(float);
        info.is_loaded = !samples_.empty();
        return info;
    }

    const void* GetRawJSON() const override { return raw_json_ ? &(*raw_json_) : nullptr; }

private:
    void LoadData() {
        std::ifstream file(path_);
        if (!file) {
            spdlog::error("Failed to open JSON file: {}", path_);
            return;
        }

        try {
            using json = nlohmann::json;
            json j;
            file >> j;

            // Store raw JSON for preview
            raw_json_ = j;

            std::set<int> unique_labels;

            // Try different JSON structures
            // Structure 1: {"data": [[...], [...]], "labels": [0, 1, ...]}
            if (j.contains("data") && j.contains("labels")) {
                auto& data = j["data"];
                auto& labels = j["labels"];

                for (size_t i = 0; i < data.size() && i < labels.size(); i++) {
                    std::vector<float> sample;
                    for (const auto& val : data[i]) {
                        if (val.is_number()) {
                            sample.push_back(val.get<float>());
                        }
                    }
                    int label = labels[i].is_number() ? labels[i].get<int>() : 0;
                    samples_.push_back(std::move(sample));
                    labels_.push_back(label);
                    unique_labels.insert(label);
                }
            }
            // Structure 2: [{"features": [...], "label": 0}, ...]
            else if (j.is_array()) {
                for (const auto& item : j) {
                    std::vector<float> sample;
                    int label = 0;

                    if (item.contains("features")) {
                        for (const auto& val : item["features"]) {
                            if (val.is_number()) {
                                sample.push_back(val.get<float>());
                            }
                        }
                    } else if (item.contains("data")) {
                        for (const auto& val : item["data"]) {
                            if (val.is_number()) {
                                sample.push_back(val.get<float>());
                            }
                        }
                    }

                    if (item.contains("label")) {
                        label = item["label"].is_number() ? item["label"].get<int>() : 0;
                    } else if (item.contains("target")) {
                        label = item["target"].is_number() ? item["target"].get<int>() : 0;
                    }

                    if (!sample.empty()) {
                        samples_.push_back(std::move(sample));
                        labels_.push_back(label);
                        unique_labels.insert(label);
                    }
                }
            }

            num_features_ = samples_.empty() ? 0 : static_cast<int>(samples_[0].size());
            num_classes_ = static_cast<int>(unique_labels.size());

            if (!samples_.empty()) {
                spdlog::info("Loaded JSON dataset: {} samples, {} features, {} classes",
                    samples_.size(), num_features_, num_classes_);
                SetSplit(SplitConfig{});
            } else {
                spdlog::info("Loaded JSON file (configuration or metadata): {}", path_);
            }

        } catch (const std::exception& e) {
            spdlog::error("Error parsing JSON file: {}", e.what());
        }
    }

    std::string path_;
    std::vector<std::vector<float>> samples_;
    std::vector<int> labels_;
    int num_features_ = 0;
    int num_classes_ = 0;
    std::optional<nlohmann::json> raw_json_;
};

/**
 * TXT Dataset Implementation
 * Loads plain text files with one sample per line (space/comma separated features, last value is label)
 */
class TXTDataset : public Dataset {
public:
    TXTDataset(const std::string& path) : path_(path) {
        LoadData();
    }

    size_t Size() const override { return samples_.size(); }

    std::pair<std::vector<float>, int> GetItem(size_t index) const override {
        if (index >= samples_.size()) return {{}, -1};
        return {samples_[index], labels_[index]};
    }

    DatasetInfo GetInfo() const override {
        DatasetInfo info;
        info.name = fs::path(path_).stem().string();
        info.path = path_;
        info.type = DatasetType::TXT;
        info.shape = num_features_ > 0 ? std::vector<size_t>{static_cast<size_t>(num_features_)} : std::vector<size_t>{};
        info.num_samples = samples_.size();
        info.num_classes = num_classes_;
        info.train_count = train_indices_.size();
        info.val_count = val_indices_.size();
        info.test_count = test_indices_.size();
        info.memory_usage = samples_.size() * num_features_ * sizeof(float);
        info.is_loaded = !samples_.empty();
        return info;
    }

    // Get raw lines for non-numeric text files
    const std::vector<std::string>& GetLines() const { return raw_lines_; }
    const std::vector<std::string>& GetTextLines() const override { return raw_lines_; }

private:
    void LoadData() {
        std::ifstream file(path_);
        if (!file) {
            spdlog::error("Failed to open TXT file: {}", path_);
            return;
        }

        std::string line;
        std::set<int> unique_labels;
        bool is_numeric = true;

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            raw_lines_.push_back(line);

            // Try to parse as numeric data (space or comma separated)
            std::vector<std::string> tokens;
            std::stringstream ss(line);
            std::string token;

            // Try space first, then comma
            char delimiter = (line.find(',') != std::string::npos) ? ',' : ' ';
            while (std::getline(ss, token, delimiter)) {
                token.erase(0, token.find_first_not_of(" \t\r\n"));
                token.erase(token.find_last_not_of(" \t\r\n") + 1);
                if (!token.empty()) {
                    tokens.push_back(token);
                }
            }

            if (tokens.size() >= 2) {
                std::vector<float> sample;
                sample.reserve(tokens.size() - 1);
                bool valid = true;

                for (size_t i = 0; i < tokens.size() - 1; i++) {
                    try {
                        sample.push_back(std::stof(tokens[i]));
                    } catch (...) {
                        valid = false;
                        is_numeric = false;
                        break;
                    }
                }

                if (valid) {
                    int label = 0;
                    try {
                        label = std::stoi(tokens.back());
                    } catch (...) {
                        is_numeric = false;
                    }

                    if (is_numeric) {
                        samples_.push_back(std::move(sample));
                        labels_.push_back(label);
                        unique_labels.insert(label);
                    }
                }
            }
        }

        num_features_ = samples_.empty() ? 0 : static_cast<int>(samples_[0].size());
        num_classes_ = static_cast<int>(unique_labels.size());

        if (!samples_.empty()) {
            spdlog::info("Loaded TXT dataset: {} samples, {} features, {} classes",
                samples_.size(), num_features_, num_classes_);
            SetSplit(SplitConfig{});
        } else {
            spdlog::info("Loaded TXT file as text: {} lines", raw_lines_.size());
        }
    }

    std::string path_;
    std::vector<std::vector<float>> samples_;
    std::vector<int> labels_;
    std::vector<std::string> raw_lines_;
    int num_features_ = 0;
    int num_classes_ = 0;
};

/**
 * LRU Cache for images
 * Thread-safe cache with configurable maximum size and statistics tracking
 */
template<typename K, typename V>
class LRUCache {
public:
    LRUCache(size_t max_size = 100) : max_size_(max_size) {}

    std::optional<V> Get(const K& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            misses_++;
            return std::nullopt;
        }
        hits_++;
        // Move to front (most recently used)
        access_order_.erase(it->second.second);
        access_order_.push_front(key);
        it->second.second = access_order_.begin();
        return it->second.first;
    }

    void Put(const K& key, const V& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            // Update existing
            it->second.first = value;
            access_order_.erase(it->second.second);
            access_order_.push_front(key);
            it->second.second = access_order_.begin();
            return;
        }

        // Evict if full
        while (cache_map_.size() >= max_size_ && !access_order_.empty()) {
            auto lru_key = access_order_.back();
            access_order_.pop_back();
            cache_map_.erase(lru_key);
            evictions_++;
        }

        // Insert new
        access_order_.push_front(key);
        cache_map_[key] = {value, access_order_.begin()};
    }

    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_map_.clear();
        access_order_.clear();
    }

    size_t Size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_map_.size();
    }

    void SetMaxSize(size_t max_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        max_size_ = max_size;
        // Evict excess
        while (cache_map_.size() > max_size_ && !access_order_.empty()) {
            auto lru_key = access_order_.back();
            access_order_.pop_back();
            cache_map_.erase(lru_key);
            evictions_++;
        }
    }

    // Statistics
    size_t GetHits() const { std::lock_guard<std::mutex> lock(mutex_); return hits_; }
    size_t GetMisses() const { std::lock_guard<std::mutex> lock(mutex_); return misses_; }
    size_t GetEvictions() const { std::lock_guard<std::mutex> lock(mutex_); return evictions_; }

    float GetHitRate() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t total = hits_ + misses_;
        return total > 0 ? static_cast<float>(hits_) / total * 100.0f : 0.0f;
    }

    void ResetStats() {
        std::lock_guard<std::mutex> lock(mutex_);
        hits_ = 0;
        misses_ = 0;
        evictions_ = 0;
    }

private:
    size_t max_size_;
    std::list<K> access_order_;
    std::unordered_map<K, std::pair<V, typename std::list<K>::iterator>> cache_map_;
    mutable std::mutex mutex_;

    // Statistics
    size_t hits_ = 0;
    size_t misses_ = 0;
    size_t evictions_ = 0;
};

/**
 * ImageFolder Dataset Implementation
 * Expects directory structure: root/class_name/image.jpg
 * Loads images using stb_image, resizes to consistent dimensions
 * Includes LRU cache for loaded images
 */
class ImageFolderDataset : public Dataset {
public:
    ImageFolderDataset(const std::string& path, int target_width = 224, int target_height = 224, size_t cache_size = 100)
        : path_(path), target_width_(target_width), target_height_(target_height), image_cache_(cache_size) {
        LoadData();
    }

    size_t Size() const override { return image_paths_.size(); }

    std::pair<std::vector<float>, int> GetItem(size_t index) const override {
        if (index >= image_paths_.size()) return {{}, -1};

        // Check cache first
        auto cached = image_cache_.Get(index);
        if (cached.has_value()) {
            return {cached.value(), labels_[index]};
        }

        // Lazy load the image
        std::vector<float> image = LoadImage(image_paths_[index]);

        // Store in cache
        image_cache_.Put(index, image);

        return {std::move(image), labels_[index]};
    }

    DatasetInfo GetInfo() const override {
        DatasetInfo info;
        info.name = fs::path(path_).filename().string();
        info.path = path_;
        info.type = DatasetType::ImageFolder;
        info.shape = {static_cast<size_t>(target_width_),
                      static_cast<size_t>(target_height_),
                      static_cast<size_t>(channels_)};
        info.num_samples = image_paths_.size();
        info.num_classes = class_names_.size();
        info.class_names = class_names_;
        info.train_count = train_indices_.size();
        info.val_count = val_indices_.size();
        info.test_count = test_indices_.size();
        // Estimate memory (lazy loading, but estimate full load)
        info.memory_usage = image_paths_.size() * target_width_ * target_height_ * channels_ * sizeof(float);
        info.is_loaded = !image_paths_.empty();
        return info;
    }

private:
    void LoadData() {
        if (!fs::exists(path_) || !fs::is_directory(path_)) {
            spdlog::error("ImageFolder path does not exist or is not a directory: {}", path_);
            return;
        }

        // Scan for class directories
        std::vector<std::string> class_dirs;
        for (const auto& entry : fs::directory_iterator(path_)) {
            if (entry.is_directory()) {
                class_dirs.push_back(entry.path().filename().string());
            }
        }

        if (class_dirs.empty()) {
            spdlog::error("No class directories found in: {}", path_);
            return;
        }

        // Sort class names for consistent label assignment
        std::sort(class_dirs.begin(), class_dirs.end());
        class_names_ = class_dirs;

        // Build class name to label mapping
        std::map<std::string, int> class_to_label;
        for (size_t i = 0; i < class_names_.size(); i++) {
            class_to_label[class_names_[i]] = static_cast<int>(i);
        }

        // Scan for images in each class directory
        std::vector<std::string> valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tga"};

        for (const auto& class_name : class_names_) {
            fs::path class_path = fs::path(path_) / class_name;
            int label = class_to_label[class_name];

            for (const auto& entry : fs::directory_iterator(class_path)) {
                if (!entry.is_regular_file()) continue;

                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (std::find(valid_extensions.begin(), valid_extensions.end(), ext) != valid_extensions.end()) {
                    image_paths_.push_back(entry.path().string());
                    labels_.push_back(label);
                }
            }
        }

        if (image_paths_.empty()) {
            spdlog::error("No valid images found in: {}", path_);
            return;
        }

        // Detect channels from first image using OpenCV
        std::vector<float> temp_data;
        int width, height, channels;
        if (!ImageUtils::LoadImage(image_paths_[0], temp_data, width, height, channels)) {
            spdlog::warn("Could not get info for first image, defaulting to 3 channels");
            channels_ = 3;
        } else {
            channels_ = channels;
        }

        spdlog::info("Loaded ImageFolder dataset: {} images, {} classes from {}",
            image_paths_.size(), class_names_.size(), path_);

        // Apply default split
        SetSplit(SplitConfig{});
    }

    std::vector<float> LoadImage(const std::string& path) const {
        // Load image using OpenCV (via ImageUtils)
        std::vector<float> result;
        int width, height, channels;

        if (!ImageUtils::LoadImage(path, result, width, height, channels)) {
            spdlog::warn("Failed to load image: {}", path);
            return std::vector<float>(target_width_ * target_height_ * channels_, 0.0f);
        }

        // Resize if needed using OpenCV high-quality resize
        if (width != target_width_ || height != target_height_) {
            // Use Area method for downscaling (best quality), Lanczos for upscaling
            ImageUtils::ResizeMethod method = (target_width_ < width || target_height_ < height)
                ? ImageUtils::ResizeMethod::Area
                : ImageUtils::ResizeMethod::Lanczos;

            if (!ImageUtils::ResizeImage(result, width, height, channels,
                                          target_width_, target_height_, method)) {
                spdlog::warn("Failed to resize image: {}", path);
                return std::vector<float>(target_width_ * target_height_ * channels_, 0.0f);
            }
        }

        return result;
    }

    std::string path_;
    int target_width_;
    int target_height_;
    int channels_ = 3;
    std::vector<std::string> image_paths_;
    std::vector<int> labels_;
    std::vector<std::string> class_names_;
    mutable LRUCache<size_t, std::vector<float>> image_cache_;
};

/**
 * ImageCSV Dataset Implementation
 * Loads images from a folder with labels from a CSV file
 * CSV format: filename,label (or filename,label_name)
 * Supports both numeric labels and string class names
 */
class ImageCSVDataset : public Dataset {
public:
    ImageCSVDataset(const std::string& image_folder, const std::string& csv_path,
                    int target_width = 224, int target_height = 224, size_t cache_size = 100,
                    const std::string& filename_col = "", const std::string& label_col = "")
        : image_folder_(image_folder), csv_path_(csv_path),
          target_width_(target_width), target_height_(target_height),
          filename_col_(filename_col), label_col_(label_col),
          image_cache_(cache_size) {
        LoadData();
    }

    size_t Size() const override { return image_paths_.size(); }

    std::pair<std::vector<float>, int> GetItem(size_t index) const override {
        if (index >= image_paths_.size()) return {{}, -1};

        // Check cache first
        auto cached = image_cache_.Get(index);
        if (cached.has_value()) {
            return {cached.value(), labels_[index]};
        }

        // Lazy load the image
        std::vector<float> image = LoadImage(image_paths_[index]);

        // Store in cache
        image_cache_.Put(index, image);

        return {std::move(image), labels_[index]};
    }

    DatasetInfo GetInfo() const override {
        DatasetInfo info;
        info.name = fs::path(image_folder_).filename().string() + (csv_path_.empty() ? " (Folder)" : " (CSV)");
        info.path = image_folder_;
        info.type = DatasetType::ImageCSV;
        info.shape = {static_cast<size_t>(target_width_),
                      static_cast<size_t>(target_height_),
                      static_cast<size_t>(channels_)};
        info.num_samples = image_paths_.size();
        info.num_classes = class_names_.size();
        info.class_names = class_names_;
        info.train_count = train_indices_.size();
        info.val_count = val_indices_.size();
        info.test_count = test_indices_.size();

        // Memory usage is the cache size, not total dataset size (lazy loading)
        size_t image_bytes = target_width_ * target_height_ * channels_ * sizeof(float);
        size_t cached_images = image_cache_.Size();
        info.memory_usage = cached_images * image_bytes;  // Actual memory in use
        info.cache_usage = cached_images * image_bytes;
        info.is_loaded = !image_paths_.empty();
        info.is_streaming = true;  // Mark as streaming/lazy loading

        // Track cache stats
        info.cache_hits = image_cache_.GetHits();
        info.cache_misses = image_cache_.GetMisses();

        return info;
    }

private:
    void LoadData() {
        // Validate image folder
        if (!fs::exists(image_folder_) || !fs::is_directory(image_folder_)) {
            spdlog::error("ImageDataset: Image folder does not exist: {}", image_folder_);
            return;
        }

        // Valid image extensions
        std::vector<std::string> valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tga", ".tiff", ".webp"};
        auto is_image = [&](const std::string& ext) {
            std::string lower_ext = ext;
            std::transform(lower_ext.begin(), lower_ext.end(), lower_ext.begin(), ::tolower);
            return std::find(valid_extensions.begin(), valid_extensions.end(), lower_ext) != valid_extensions.end();
        };

        // Check if CSV is provided and exists
        bool use_csv = !csv_path_.empty() && fs::exists(csv_path_);

        if (use_csv) {
            // CSV mode: load images with labels from CSV file
            LoadFromCSV(valid_extensions);
        } else {
            // Folder mode: scan subfolders, use subfolder names as class labels
            LoadFromFolder(is_image);
        }

        if (image_paths_.empty()) {
            spdlog::error("ImageDataset: No valid images found");
            return;
        }

        // Detect channels from first image using OpenCV
        std::vector<float> temp_data;
        int width, height, channels;
        if (!ImageUtils::LoadImage(image_paths_[0], temp_data, width, height, channels)) {
            spdlog::warn("ImageDataset: Could not get info for first image, defaulting to 3 channels");
            channels_ = 3;
        } else {
            channels_ = channels;
        }

        if (use_csv) {
            spdlog::info("ImageDataset: Loaded {} images, {} classes from {} + {}",
                image_paths_.size(), class_names_.size(), image_folder_, csv_path_);
        } else {
            spdlog::info("ImageDataset: Loaded {} images, {} classes from folder {}",
                image_paths_.size(), class_names_.size(), image_folder_);
        }

        // Apply default split
        SetSplit(SplitConfig{});
    }

    void LoadFromCSV(const std::vector<std::string>& valid_extensions) {
        std::ifstream file(csv_path_);
        if (!file.is_open()) {
            spdlog::error("ImageDataset: Failed to open CSV file: {}", csv_path_);
            return;
        }

        std::string line;
        std::vector<std::string> headers;
        int filename_idx = 0;
        int label_idx = 1;

        // Read first line to detect headers
        if (std::getline(file, line)) {
            std::vector<std::string> first_row = ParseCSVLine(line);

            // Check if first line looks like a header
            // Use multiple heuristics: exact match, suffix match, substring match
            bool likely_header = false;
            for (const auto& cell : first_row) {
                std::string lower = cell;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

                // Exact matches for common header names
                if (lower == "filename" || lower == "file" || lower == "image" ||
                    lower == "label" || lower == "class" || lower == "target" ||
                    lower == "id" || lower == "name" || lower == "path" ||
                    lower == "dx" || lower == "category" || lower == "diagnosis") {
                    likely_header = true;
                    break;
                }

                // Suffix matches (e.g., "image_id", "file_name", "class_name")
                if (lower.length() > 3) {
                    if (lower.substr(lower.length() - 3) == "_id" ||
                        lower.substr(lower.length() - 5 > 0 ? lower.length() - 5 : 0) == "_name" ||
                        lower.substr(lower.length() - 5 > 0 ? lower.length() - 5 : 0) == "_path" ||
                        lower.substr(lower.length() - 5 > 0 ? lower.length() - 5 : 0) == "_file" ||
                        lower.substr(lower.length() - 6 > 0 ? lower.length() - 6 : 0) == "_class" ||
                        lower.substr(lower.length() - 6 > 0 ? lower.length() - 6 : 0) == "_label") {
                        likely_header = true;
                        break;
                    }
                }

                // Substring matches for common header keywords
                if (lower.find("image") != std::string::npos ||
                    lower.find("file") != std::string::npos ||
                    lower.find("label") != std::string::npos ||
                    lower.find("class") != std::string::npos ||
                    lower.find("name") != std::string::npos) {
                    likely_header = true;
                    break;
                }

                // Check if cell is non-numeric text (headers are usually text)
                // If all cells look like text (not numbers or file paths), it's likely a header
                bool is_numeric = !lower.empty();
                for (char c : lower) {
                    if (!std::isdigit(c) && c != '.' && c != '-' && c != '+' && c != 'e') {
                        is_numeric = false;
                        break;
                    }
                }
                // If it's a short non-numeric string without path separators, likely a header
                if (!is_numeric && lower.length() < 30 &&
                    lower.find('/') == std::string::npos &&
                    lower.find('\\') == std::string::npos &&
                    lower.find('.') == std::string::npos) {
                    // Check first row for typical header patterns (age, sex, etc.)
                    if (lower == "age" || lower == "sex" || lower == "type" ||
                        lower == "date" || lower == "time" || lower == "index" ||
                        lower == "row" || lower == "col" || lower == "value") {
                        likely_header = true;
                        break;
                    }
                }
            }

            if (likely_header) {
                headers = first_row;

                // Find column indices
                if (!filename_col_.empty()) {
                    for (size_t i = 0; i < headers.size(); i++) {
                        if (headers[i] == filename_col_) {
                            filename_idx = static_cast<int>(i);
                            break;
                        }
                    }
                } else {
                    // Auto-detect filename column (prioritize image_id over other columns)
                    // Priority list: image_id > filename > image > file > path > img
                    std::vector<std::string> filename_priorities = {
                        "image_id", "imageid", "image_name", "imagename",
                        "filename", "file_name", "file", "image",
                        "path", "image_path", "img", "photo", "picture"
                    };
                    for (const auto& prio : filename_priorities) {
                        bool found = false;
                        for (size_t i = 0; i < headers.size(); i++) {
                            std::string lower = headers[i];
                            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                            if (lower == prio) {
                                filename_idx = static_cast<int>(i);
                                found = true;
                                break;
                            }
                        }
                        if (found) break;
                    }
                }

                if (!label_col_.empty()) {
                    for (size_t i = 0; i < headers.size(); i++) {
                        if (headers[i] == label_col_) {
                            label_idx = static_cast<int>(i);
                            break;
                        }
                    }
                } else {
                    // Auto-detect label column
                    // Priority list for label column detection
                    std::vector<std::string> label_priorities = {
                        "label", "labels", "class", "class_name", "classname",
                        "target", "category", "dx", "diagnosis", "y"
                    };
                    for (const auto& prio : label_priorities) {
                        bool found = false;
                        for (size_t i = 0; i < headers.size(); i++) {
                            std::string lower = headers[i];
                            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                            if (lower == prio) {
                                label_idx = static_cast<int>(i);
                                found = true;
                                break;
                            }
                        }
                        if (found) break;
                    }
                }

                spdlog::info("ImageDataset: Using column {} for filenames, column {} for labels",
                    filename_idx < static_cast<int>(headers.size()) ? headers[filename_idx] : std::to_string(filename_idx),
                    label_idx < static_cast<int>(headers.size()) ? headers[label_idx] : std::to_string(label_idx));
            } else {
                // No header, process first line as data
                file.seekg(0);
            }
        }

        // Build label mapping
        std::map<std::string, int> label_map;
        std::vector<std::pair<std::string, std::string>> raw_data;

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            std::vector<std::string> row = ParseCSVLine(line);
            if (row.size() <= static_cast<size_t>(std::max(filename_idx, label_idx))) {
                continue;
            }

            std::string filename = row[filename_idx];
            std::string label_str = row[label_idx];

            // Trim whitespace
            filename.erase(0, filename.find_first_not_of(" \t\r\n"));
            filename.erase(filename.find_last_not_of(" \t\r\n") + 1);
            label_str.erase(0, label_str.find_first_not_of(" \t\r\n"));
            label_str.erase(label_str.find_last_not_of(" \t\r\n") + 1);

            raw_data.emplace_back(filename, label_str);

            if (label_map.find(label_str) == label_map.end()) {
                label_map[label_str] = static_cast<int>(label_map.size());
            }
        }

        file.close();

        // Build class names
        class_names_.resize(label_map.size());
        for (const auto& [name, idx] : label_map) {
            class_names_[idx] = name;
        }

        // Build a map of filename -> full path by scanning the folder (including subfolders)
        // This handles cases where images are in nested folders
        std::map<std::string, std::string> filename_to_path;
        auto scan_dir = [&](const fs::path& dir, auto& self) -> void {
            for (const auto& entry : fs::directory_iterator(dir)) {
                if (entry.is_directory()) {
                    self(entry.path(), self);  // Recurse into subdirectory
                } else if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::string lower_ext = ext;
                    std::transform(lower_ext.begin(), lower_ext.end(), lower_ext.begin(), ::tolower);
                    if (std::find(valid_extensions.begin(), valid_extensions.end(), lower_ext) != valid_extensions.end()) {
                        // Store both with and without extension for matching
                        std::string full_path = entry.path().string();
                        std::string stem = entry.path().stem().string();
                        std::string name_with_ext = entry.path().filename().string();

                        filename_to_path[stem] = full_path;
                        filename_to_path[name_with_ext] = full_path;
                    }
                }
            }
        };
        scan_dir(image_folder_, scan_dir);

        spdlog::info("ImageDataset: Scanned {} image files in folder tree", filename_to_path.size() / 2);

        // Process each row
        for (const auto& [filename, label_str] : raw_data) {
            std::string resolved_path;

            // First try direct lookup in our map
            auto it = filename_to_path.find(filename);
            if (it != filename_to_path.end()) {
                resolved_path = it->second;
            } else {
                // Try with direct path construction
                fs::path image_path = fs::path(image_folder_) / filename;
                if (fs::exists(image_path)) {
                    resolved_path = image_path.string();
                } else {
                    // Try adding extensions
                    for (const auto& ext : valid_extensions) {
                        fs::path test_path = fs::path(image_folder_) / (filename + ext);
                        if (fs::exists(test_path)) {
                            resolved_path = test_path.string();
                            break;
                        }
                    }
                }
            }

            if (resolved_path.empty()) {
                spdlog::warn("ImageDataset: Image not found: {}", filename);
                continue;
            }

            image_paths_.push_back(resolved_path);
            labels_.push_back(label_map[label_str]);
        }
    }

    void LoadFromFolder(std::function<bool(const std::string&)> is_image) {
        // Scan for subfolders (class folders) or direct images
        std::map<std::string, int> label_map;
        bool has_subfolders = false;

        // First pass: check if we have class subfolders
        for (const auto& entry : fs::directory_iterator(image_folder_)) {
            if (entry.is_directory()) {
                has_subfolders = true;
                break;
            }
        }

        if (has_subfolders) {
            // Class subfolder mode: folder/class_name/images
            for (const auto& class_dir : fs::directory_iterator(image_folder_)) {
                if (!class_dir.is_directory()) continue;

                std::string class_name = class_dir.path().filename().string();

                // Skip hidden folders
                if (class_name.empty() || class_name[0] == '.') continue;

                // Assign class index
                if (label_map.find(class_name) == label_map.end()) {
                    label_map[class_name] = static_cast<int>(label_map.size());
                }
                int class_idx = label_map[class_name];

                // Scan images in class folder
                for (const auto& img_entry : fs::directory_iterator(class_dir.path())) {
                    if (!img_entry.is_regular_file()) continue;

                    std::string ext = img_entry.path().extension().string();
                    if (!is_image(ext)) continue;

                    image_paths_.push_back(img_entry.path().string());
                    labels_.push_back(class_idx);
                }
            }

            // Build class names
            class_names_.resize(label_map.size());
            for (const auto& [name, idx] : label_map) {
                class_names_[idx] = name;
            }

            spdlog::info("ImageDataset: Found {} class subfolders", class_names_.size());
        } else {
            // Flat folder mode: all images in one folder, no labels (single class)
            class_names_.push_back("default");

            for (const auto& entry : fs::directory_iterator(image_folder_)) {
                if (!entry.is_regular_file()) continue;

                std::string ext = entry.path().extension().string();
                if (!is_image(ext)) continue;

                image_paths_.push_back(entry.path().string());
                labels_.push_back(0);  // All same class
            }

            spdlog::info("ImageDataset: Flat folder mode, {} images with single class", image_paths_.size());
        }
    }

    std::vector<std::string> ParseCSVLine(const std::string& line) {
        std::vector<std::string> result;
        std::string current;
        bool in_quotes = false;

        for (size_t i = 0; i < line.size(); i++) {
            char c = line[i];

            if (c == '"') {
                if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                    // Escaped quote
                    current += '"';
                    i++;
                } else {
                    in_quotes = !in_quotes;
                }
            } else if (c == ',' && !in_quotes) {
                result.push_back(current);
                current.clear();
            } else {
                current += c;
            }
        }
        result.push_back(current);

        return result;
    }

    std::vector<float> LoadImage(const std::string& path) const {
        // Load image using OpenCV (via ImageUtils)
        std::vector<float> result;
        int width, height, channels;

        if (!ImageUtils::LoadImage(path, result, width, height, channels)) {
            spdlog::warn("ImageCSV: Failed to load image: {}", path);
            return std::vector<float>(target_width_ * target_height_ * channels_, 0.0f);
        }

        // Resize if needed using OpenCV high-quality resize
        if (width != target_width_ || height != target_height_) {
            // Use Area method for downscaling (best quality), Lanczos for upscaling
            ImageUtils::ResizeMethod method = (target_width_ < width || target_height_ < height)
                ? ImageUtils::ResizeMethod::Area
                : ImageUtils::ResizeMethod::Lanczos;

            if (!ImageUtils::ResizeImage(result, width, height, channels,
                                          target_width_, target_height_, method)) {
                spdlog::warn("ImageCSV: Failed to resize image: {}", path);
                return std::vector<float>(target_width_ * target_height_ * channels_, 0.0f);
            }
        }

        return result;
    }

    std::string image_folder_;
    std::string csv_path_;
    std::string filename_col_;
    std::string label_col_;
    int target_width_;
    int target_height_;
    int channels_ = 3;
    std::vector<std::string> image_paths_;
    std::vector<int> labels_;
    std::vector<std::string> class_names_;
    mutable LRUCache<size_t, std::vector<float>> image_cache_;
};

/**
 * HuggingFace Dataset Implementation
 * Uses Python's datasets library via embedded interpreter
 * Caches data locally after first download
 */
class HuggingFaceDataset : public Dataset {
public:
    HuggingFaceDataset(const HuggingFaceConfig& config) : config_(config) {
        LoadData();
    }

    size_t Size() const override { return samples_.size(); }

    std::pair<std::vector<float>, int> GetItem(size_t index) const override {
        if (index >= samples_.size()) return {{}, -1};
        return {samples_[index], labels_[index]};
    }

    DatasetInfo GetInfo() const override {
        DatasetInfo info;
        info.name = config_.dataset_name;
        info.path = "huggingface://" + config_.dataset_name;
        info.type = DatasetType::HuggingFace;
        info.shape = shape_;
        info.num_samples = samples_.size();
        info.num_classes = num_classes_;
        info.class_names = class_names_;
        info.train_count = train_indices_.size();
        info.val_count = val_indices_.size();
        info.test_count = test_indices_.size();
        info.memory_usage = samples_.size() * (shape_.empty() ? 0 : shape_[0]) * sizeof(float);
        info.is_loaded = !samples_.empty();
        info.is_streaming = config_.streaming;
        return info;
    }

    // Streaming interface
    bool IsStreaming() const override { return config_.streaming; }

    bool HasNext() const override {
        return config_.streaming && stream_position_ < estimated_size_;
    }

    std::pair<std::vector<float>, int> GetNext() override {
        if (!config_.streaming || stream_position_ >= estimated_size_) {
            return {{}, -1};
        }
        // In streaming mode, fetch next sample
        // This would call Python to get next item from iterator
        stream_position_++;
        if (stream_position_ <= samples_.size()) {
            return {samples_[stream_position_ - 1], labels_[stream_position_ - 1]};
        }
        return {{}, -1};
    }

    void ResetStream() override {
        stream_position_ = 0;
    }

private:
    void LoadData() {
        spdlog::info("Loading HuggingFace dataset: {}", config_.dataset_name);

        // Determine cache directory
        std::string cache_dir = config_.cache_dir;
        if (cache_dir.empty()) {
            cache_dir = "./data/huggingface_cache";
        }
        fs::create_directories(cache_dir);

        // Check for cached data first
        std::string cache_file = cache_dir + "/" + config_.dataset_name + "_" + config_.split + ".bin";
        if (fs::exists(cache_file)) {
            if (LoadFromCache(cache_file)) {
                spdlog::info("Loaded HuggingFace dataset from cache: {} samples", samples_.size());
                SetSplit(SplitConfig{});
                return;
            }
        }

        // Map common dataset names to their configurations
        if (!LoadPredefinedDataset()) {
            spdlog::error("Dataset '{}' not available. Install 'datasets' Python package and ensure it's downloaded.",
                         config_.dataset_name);
            return;
        }

        // Save to cache
        SaveToCache(cache_file);

        if (!samples_.empty()) {
            SetSplit(SplitConfig{});
        }
    }

    bool LoadPredefinedDataset() {
        // Map of predefined HuggingFace datasets with sample data
        // In production, this would use Python embedding to call datasets library

        std::string name = config_.dataset_name;
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);

        if (name == "mnist" || name == "fashion_mnist") {
            // Generate MNIST-like sample data for demonstration
            shape_ = {784};  // 28x28 flattened
            num_classes_ = 10;
            class_names_ = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

            // Create sample data (in real implementation, this calls Python)
            size_t num_samples = 1000;  // Demo size
            samples_.reserve(num_samples);
            labels_.reserve(num_samples);

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> label_dist(0, 9);
            std::normal_distribution<float> pixel_dist(0.5f, 0.3f);

            for (size_t i = 0; i < num_samples; i++) {
                std::vector<float> sample(784);
                for (auto& pixel : sample) {
                    pixel = std::clamp(pixel_dist(gen), 0.0f, 1.0f);
                }
                samples_.push_back(std::move(sample));
                labels_.push_back(label_dist(gen));
            }

            spdlog::info("Created HuggingFace MNIST placeholder: {} samples", samples_.size());
            spdlog::warn("For real HuggingFace data, install: pip install datasets");
            return true;
        }
        else if (name == "cifar10" || name == "cifar100") {
            shape_ = {3072};  // 32x32x3 flattened
            num_classes_ = (name == "cifar10") ? 10 : 100;

            if (name == "cifar10") {
                class_names_ = {"airplane", "automobile", "bird", "cat", "deer",
                               "dog", "frog", "horse", "ship", "truck"};
            }

            size_t num_samples = 1000;
            samples_.reserve(num_samples);
            labels_.reserve(num_samples);

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> label_dist(0, num_classes_ - 1);
            std::uniform_real_distribution<float> pixel_dist(0.0f, 1.0f);

            for (size_t i = 0; i < num_samples; i++) {
                std::vector<float> sample(3072);
                for (auto& pixel : sample) {
                    pixel = pixel_dist(gen);
                }
                samples_.push_back(std::move(sample));
                labels_.push_back(label_dist(gen));
            }

            spdlog::info("Created HuggingFace CIFAR placeholder: {} samples", samples_.size());
            return true;
        }
        else if (name == "imdb") {
            // Text classification dataset
            shape_ = {512};  // Embedding size
            num_classes_ = 2;
            class_names_ = {"negative", "positive"};

            size_t num_samples = 500;
            samples_.reserve(num_samples);
            labels_.reserve(num_samples);

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> label_dist(0, 1);
            std::normal_distribution<float> embed_dist(0.0f, 1.0f);

            for (size_t i = 0; i < num_samples; i++) {
                std::vector<float> sample(512);
                for (auto& val : sample) {
                    val = embed_dist(gen);
                }
                samples_.push_back(std::move(sample));
                labels_.push_back(label_dist(gen));
            }

            spdlog::info("Created HuggingFace IMDB placeholder: {} samples", samples_.size());
            return true;
        }

        spdlog::warn("Unknown HuggingFace dataset: {}", config_.dataset_name);
        return false;
    }

    bool LoadFromCache(const std::string& cache_file) {
        std::ifstream file(cache_file, std::ios::binary);
        if (!file) return false;

        try {
            // Read header
            size_t num_samples, feature_size, num_classes;
            file.read(reinterpret_cast<char*>(&num_samples), sizeof(num_samples));
            file.read(reinterpret_cast<char*>(&feature_size), sizeof(feature_size));
            file.read(reinterpret_cast<char*>(&num_classes), sizeof(num_classes));

            shape_ = {feature_size};
            num_classes_ = num_classes;

            // Read samples
            samples_.resize(num_samples);
            labels_.resize(num_samples);

            for (size_t i = 0; i < num_samples; i++) {
                samples_[i].resize(feature_size);
                file.read(reinterpret_cast<char*>(samples_[i].data()), feature_size * sizeof(float));
                file.read(reinterpret_cast<char*>(&labels_[i]), sizeof(int));
            }

            return file.good();
        } catch (...) {
            return false;
        }
    }

    void SaveToCache(const std::string& cache_file) {
        std::ofstream file(cache_file, std::ios::binary);
        if (!file) return;

        size_t num_samples = samples_.size();
        size_t feature_size = shape_.empty() ? 0 : shape_[0];
        size_t num_classes = num_classes_;

        file.write(reinterpret_cast<const char*>(&num_samples), sizeof(num_samples));
        file.write(reinterpret_cast<const char*>(&feature_size), sizeof(feature_size));
        file.write(reinterpret_cast<const char*>(&num_classes), sizeof(num_classes));

        for (size_t i = 0; i < num_samples; i++) {
            file.write(reinterpret_cast<const char*>(samples_[i].data()), feature_size * sizeof(float));
            file.write(reinterpret_cast<const char*>(&labels_[i]), sizeof(int));
        }

        spdlog::info("Cached HuggingFace dataset to: {}", cache_file);
    }

    HuggingFaceConfig config_;
    std::vector<std::vector<float>> samples_;
    std::vector<int> labels_;
    std::vector<size_t> shape_;
    size_t num_classes_ = 0;
    std::vector<std::string> class_names_;

    // Streaming state
    size_t stream_position_ = 0;
    size_t estimated_size_ = 0;
};

/**
 * Streaming Dataset Implementation
 * Loads data in chunks for memory-efficient processing of large datasets
 */
class StreamingDataset : public Dataset {
public:
    StreamingDataset(const std::string& path, const StreamingConfig& config)
        : path_(path), config_(config) {
        Initialize();
    }

    size_t Size() const override {
        return estimated_total_size_;
    }

    std::pair<std::vector<float>, int> GetItem(size_t index) const override {
        // For streaming, we need to ensure the chunk containing this index is loaded
        size_t chunk_idx = index / config_.chunk_size;
        size_t local_idx = index % config_.chunk_size;

        // Check if chunk is in buffer
        auto it = chunk_buffer_.find(chunk_idx);
        if (it != chunk_buffer_.end()) {
            const auto& chunk = it->second;
            if (local_idx < chunk.samples.size()) {
                return {chunk.samples[local_idx], chunk.labels[local_idx]};
            }
        }

        // Load chunk
        LoadChunk(chunk_idx);

        it = chunk_buffer_.find(chunk_idx);
        if (it != chunk_buffer_.end() && local_idx < it->second.samples.size()) {
            return {it->second.samples[local_idx], it->second.labels[local_idx]};
        }

        return {{}, -1};
    }

    DatasetInfo GetInfo() const override {
        DatasetInfo info;
        info.name = fs::path(path_).stem().string();
        info.path = path_;
        info.type = detected_type_;
        info.shape = shape_;
        info.num_samples = estimated_total_size_;
        info.num_classes = num_classes_;
        info.train_count = train_indices_.size();
        info.val_count = val_indices_.size();
        info.test_count = test_indices_.size();
        info.memory_usage = GetBufferMemoryUsage();
        info.is_loaded = true;
        info.is_streaming = true;
        return info;
    }

    // Streaming interface
    bool IsStreaming() const override { return true; }

    bool HasNext() const override {
        return current_position_ < estimated_total_size_;
    }

    std::pair<std::vector<float>, int> GetNext() override {
        if (current_position_ >= estimated_total_size_) {
            return {{}, -1};
        }
        auto result = GetItem(current_position_);
        current_position_++;
        return result;
    }

    void ResetStream() override {
        current_position_ = 0;
        chunk_buffer_.clear();
    }

private:
    struct DataChunk {
        std::vector<std::vector<float>> samples;
        std::vector<int> labels;
    };

    void Initialize() {
        // Detect dataset type and estimate size
        detected_type_ = DataRegistry::DetectType(path_);

        if (!fs::exists(path_)) {
            spdlog::error("Streaming dataset path does not exist: {}", path_);
            return;
        }

        // Estimate total size based on file/directory
        EstimateSize();

        spdlog::info("Initialized streaming dataset: {} (est. {} samples, {} chunks)",
                    path_, estimated_total_size_, (estimated_total_size_ + config_.chunk_size - 1) / config_.chunk_size);
    }

    void EstimateSize() {
        if (detected_type_ == DatasetType::CSV) {
            // Count lines in CSV
            std::ifstream file(path_);
            estimated_total_size_ = std::count(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(), '\n');
            if (estimated_total_size_ > 0) estimated_total_size_--;  // Subtract header
        }
        else if (detected_type_ == DatasetType::ImageFolder) {
            // Count images
            for (const auto& entry : fs::recursive_directory_iterator(path_)) {
                if (entry.is_regular_file()) {
                    auto ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                        estimated_total_size_++;
                    }
                }
            }
        }
        else {
            estimated_total_size_ = 10000;  // Default estimate
        }
    }

    void LoadChunk(size_t chunk_idx) const {
        // Evict old chunks if buffer is full
        while (chunk_buffer_.size() >= max_chunks_in_buffer_) {
            // Remove oldest chunk
            if (!chunk_access_order_.empty()) {
                size_t oldest = chunk_access_order_.front();
                chunk_access_order_.erase(chunk_access_order_.begin());
                chunk_buffer_.erase(oldest);
            }
        }

        // Load the chunk
        DataChunk chunk;
        size_t start_idx = chunk_idx * config_.chunk_size;
        size_t end_idx = std::min(start_idx + config_.chunk_size, estimated_total_size_);

        if (detected_type_ == DatasetType::CSV) {
            LoadCSVChunk(chunk, start_idx, end_idx);
        }
        else if (detected_type_ == DatasetType::ImageFolder) {
            LoadImageChunk(chunk, start_idx, end_idx);
        }

        chunk_buffer_[chunk_idx] = std::move(chunk);
        chunk_access_order_.push_back(chunk_idx);
    }

    void LoadCSVChunk(DataChunk& chunk, size_t start_idx, size_t end_idx) const {
        std::ifstream file(path_);
        if (!file) return;

        std::string line;
        size_t current_line = 0;

        // Skip header
        std::getline(file, line);

        while (std::getline(file, line) && current_line < end_idx) {
            if (current_line >= start_idx) {
                std::vector<std::string> tokens;
                std::stringstream ss(line);
                std::string token;

                while (std::getline(ss, token, ',')) {
                    token.erase(0, token.find_first_not_of(" \t\r\n"));
                    token.erase(token.find_last_not_of(" \t\r\n") + 1);
                    tokens.push_back(token);
                }

                if (!tokens.empty()) {
                    std::vector<float> sample;
                    for (size_t i = 0; i < tokens.size() - 1; i++) {
                        try {
                            sample.push_back(std::stof(tokens[i]));
                        } catch (...) {
                            sample.push_back(0.0f);
                        }
                    }

                    int label = 0;
                    try {
                        label = std::stoi(tokens.back());
                    } catch (...) {}

                    chunk.samples.push_back(std::move(sample));
                    chunk.labels.push_back(label);
                }
            }
            current_line++;
        }

        // Update shape if this is first chunk
        if (shape_.empty() && !chunk.samples.empty()) {
            shape_ = {chunk.samples[0].size()};
        }
    }

    void LoadImageChunk(DataChunk& chunk, size_t start_idx, size_t end_idx) const {
        // Collect image paths first (if not already done)
        if (image_paths_.empty()) {
            CollectImagePaths();
        }

        for (size_t i = start_idx; i < end_idx && i < image_paths_.size(); i++) {
            const auto& [path, label] = image_paths_[i];

            int width, height, channels;
            unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 0);

            if (data) {
                std::vector<float> sample(width * height * channels);
                for (int j = 0; j < width * height * channels; j++) {
                    sample[j] = data[j] / 255.0f;
                }
                stbi_image_free(data);

                chunk.samples.push_back(std::move(sample));
                chunk.labels.push_back(label);

                // Update shape
                if (shape_.empty()) {
                    shape_ = {static_cast<size_t>(width), static_cast<size_t>(height), static_cast<size_t>(channels)};
                }
            }
        }
    }

    void CollectImagePaths() const {
        if (!fs::is_directory(path_)) return;

        std::vector<std::string> class_dirs;
        for (const auto& entry : fs::directory_iterator(path_)) {
            if (entry.is_directory()) {
                class_dirs.push_back(entry.path().filename().string());
            }
        }
        std::sort(class_dirs.begin(), class_dirs.end());

        std::map<std::string, int> class_to_label;
        for (size_t i = 0; i < class_dirs.size(); i++) {
            class_to_label[class_dirs[i]] = static_cast<int>(i);
        }

        for (const auto& class_name : class_dirs) {
            fs::path class_path = fs::path(path_) / class_name;
            int label = class_to_label[class_name];

            for (const auto& entry : fs::directory_iterator(class_path)) {
                if (!entry.is_regular_file()) continue;

                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    image_paths_.push_back({entry.path().string(), label});
                }
            }
        }

        num_classes_ = class_dirs.size();
    }

    size_t GetBufferMemoryUsage() const {
        size_t total = 0;
        for (const auto& [_, chunk] : chunk_buffer_) {
            for (const auto& sample : chunk.samples) {
                total += sample.size() * sizeof(float);
            }
            total += chunk.labels.size() * sizeof(int);
        }
        return total;
    }

    std::string path_;
    StreamingConfig config_;
    DatasetType detected_type_ = DatasetType::None;
    mutable std::vector<size_t> shape_;
    mutable size_t num_classes_ = 0;
    size_t estimated_total_size_ = 0;
    mutable size_t current_position_ = 0;

    // Chunk buffer (LRU cache of chunks)
    mutable std::map<size_t, DataChunk> chunk_buffer_;
    mutable std::vector<size_t> chunk_access_order_;
    size_t max_chunks_in_buffer_ = 10;

    // Image paths for image folder streaming
    mutable std::vector<std::pair<std::string, int>> image_paths_;
};

/**
 * Kaggle Dataset Implementation
 * Downloads and loads datasets from Kaggle using local caching
 */
class KaggleDataset : public Dataset {
public:
    KaggleDataset(const KaggleConfig& config) : config_(config) {
        Initialize();
    }

    size_t Size() const override { return samples_.size(); }

    std::pair<std::vector<float>, int> GetItem(size_t index) const override {
        if (index >= samples_.size()) {
            return {{}, -1};
        }
        return {samples_[index], labels_[index]};
    }

    DatasetInfo GetInfo() const override {
        DatasetInfo info;
        info.name = GetDatasetName();
        info.path = GetCacheDir();
        info.type = DatasetType::Kaggle;
        info.shape = shape_;
        info.num_samples = samples_.size();
        info.num_classes = num_classes_;
        info.class_names = class_names_;
        info.train_count = train_indices_.size();
        info.val_count = val_indices_.size();
        info.test_count = test_indices_.size();
        info.memory_usage = CalculateMemoryUsage();
        info.is_loaded = true;
        return info;
    }

private:
    void Initialize() {
        // Determine cache directory
        std::string cache_dir = GetCacheDir();
        fs::create_directories(cache_dir);

        std::string dataset_name = GetDatasetName();
        spdlog::info("Initializing Kaggle dataset: {}", dataset_name);

        // Check for cached data
        std::string cache_file = cache_dir + "/" + dataset_name + ".cache";
        if (fs::exists(cache_file)) {
            if (LoadFromCache(cache_file)) {
                spdlog::info("Loaded Kaggle dataset from cache: {}", cache_file);
                return;
            }
        }

        // Try to load from downloaded files in cache directory
        if (LoadFromDownloadedFiles(cache_dir)) {
            SaveToCache(cache_file);
            return;
        }

        // Simulate well-known Kaggle datasets with predefined data
        if (LoadPredefinedDataset(dataset_name)) {
            SaveToCache(cache_file);
            spdlog::info("Loaded predefined Kaggle dataset: {}", dataset_name);
            return;
        }

        spdlog::warn("Kaggle dataset '{}' not found. Please download it using the Kaggle CLI:", dataset_name);
        spdlog::warn("  kaggle datasets download -d {}", config_.dataset_slug);
        spdlog::warn("Or for competitions:");
        spdlog::warn("  kaggle competitions download -c {}", config_.competition);
    }

    std::string GetDatasetName() const {
        if (!config_.dataset_slug.empty()) {
            // Extract name from slug (e.g., "zalando-research/fashionmnist" -> "fashionmnist")
            size_t pos = config_.dataset_slug.find_last_of('/');
            if (pos != std::string::npos) {
                return config_.dataset_slug.substr(pos + 1);
            }
            return config_.dataset_slug;
        }
        return config_.competition;
    }

    std::string GetCacheDir() const {
        if (!config_.cache_dir.empty()) {
            return config_.cache_dir;
        }
        return "./data/kaggle_cache/" + GetDatasetName();
    }

    bool LoadFromCache(const std::string& cache_file) {
        std::ifstream file(cache_file, std::ios::binary);
        if (!file.is_open()) return false;

        size_t num_samples, feature_size;
        file.read(reinterpret_cast<char*>(&num_samples), sizeof(num_samples));
        file.read(reinterpret_cast<char*>(&feature_size), sizeof(feature_size));
        file.read(reinterpret_cast<char*>(&num_classes_), sizeof(num_classes_));

        samples_.resize(num_samples);
        labels_.resize(num_samples);

        for (size_t i = 0; i < num_samples; i++) {
            samples_[i].resize(feature_size);
            file.read(reinterpret_cast<char*>(samples_[i].data()), feature_size * sizeof(float));
            file.read(reinterpret_cast<char*>(&labels_[i]), sizeof(int));
        }

        // Read shape
        size_t shape_size;
        file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
        shape_.resize(shape_size);
        file.read(reinterpret_cast<char*>(shape_.data()), shape_size * sizeof(size_t));

        SetupSplits();
        return true;
    }

    void SaveToCache(const std::string& cache_file) {
        if (samples_.empty()) return;

        std::ofstream file(cache_file, std::ios::binary);
        if (!file.is_open()) return;

        size_t num_samples = samples_.size();
        size_t feature_size = samples_[0].size();

        file.write(reinterpret_cast<const char*>(&num_samples), sizeof(num_samples));
        file.write(reinterpret_cast<const char*>(&feature_size), sizeof(feature_size));
        file.write(reinterpret_cast<const char*>(&num_classes_), sizeof(num_classes_));

        for (size_t i = 0; i < num_samples; i++) {
            file.write(reinterpret_cast<const char*>(samples_[i].data()), feature_size * sizeof(float));
            file.write(reinterpret_cast<const char*>(&labels_[i]), sizeof(int));
        }

        // Write shape
        size_t shape_size = shape_.size();
        file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
        file.write(reinterpret_cast<const char*>(shape_.data()), shape_size * sizeof(size_t));

        spdlog::info("Cached Kaggle dataset to: {}", cache_file);
    }

    bool LoadFromDownloadedFiles(const std::string& cache_dir) {
        // Look for CSV files in the cache directory
        for (const auto& entry : fs::directory_iterator(cache_dir)) {
            if (entry.path().extension() == ".csv") {
                std::string csv_path = entry.path().string();

                // Check if this is the file we want
                if (!config_.file_name.empty() &&
                    entry.path().filename().string() != config_.file_name) {
                    continue;
                }

                spdlog::info("Loading Kaggle dataset from CSV: {}", csv_path);
                return LoadCSVFile(csv_path);
            }
        }

        // Look for image folders
        for (const auto& entry : fs::directory_iterator(cache_dir)) {
            if (entry.is_directory()) {
                // Check for class subdirectories (image classification structure)
                bool has_subdirs = false;
                for (const auto& subentry : fs::directory_iterator(entry.path())) {
                    if (subentry.is_directory()) {
                        has_subdirs = true;
                        break;
                    }
                }
                if (has_subdirs) {
                    spdlog::info("Loading Kaggle dataset from image folder: {}", entry.path().string());
                    return LoadImageFolder(entry.path().string());
                }
            }
        }

        return false;
    }

    bool LoadCSVFile(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return false;

        std::string line;
        std::getline(file, line); // Skip header

        std::vector<std::vector<float>> features;
        std::vector<int> labels;
        std::map<std::string, int> label_map;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;
            std::vector<float> row;
            std::string label_str;

            // Parse CSV (assume last column is label)
            while (std::getline(ss, value, ',')) {
                label_str = value;
                if (row.empty() || ss.peek() != EOF) {
                    try {
                        row.push_back(std::stof(value));
                    } catch (...) {
                        // Non-numeric value, use as label
                    }
                }
            }

            // Map label to integer
            if (label_map.find(label_str) == label_map.end()) {
                label_map[label_str] = static_cast<int>(label_map.size());
                class_names_.push_back(label_str);
            }
            labels.push_back(label_map[label_str]);

            if (!row.empty()) {
                row.pop_back(); // Remove label from features
                features.push_back(row);
            }
        }

        if (features.empty()) return false;

        samples_ = std::move(features);
        labels_ = std::move(labels);
        num_classes_ = label_map.size();
        shape_ = {samples_[0].size()};

        SetupSplits();
        return true;
    }

    bool LoadImageFolder(const std::string& path) {
        std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp"};
        int label = 0;

        for (const auto& class_dir : fs::directory_iterator(path)) {
            if (!class_dir.is_directory()) continue;

            class_names_.push_back(class_dir.path().filename().string());

            for (const auto& img_entry : fs::directory_iterator(class_dir.path())) {
                std::string ext = img_entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                    // Load image using stb_image
                    int width, height, channels;
                    unsigned char* data = stbi_load(img_entry.path().string().c_str(),
                                                     &width, &height, &channels, 0);
                    if (data) {
                        std::vector<float> sample(width * height * channels);
                        for (int i = 0; i < width * height * channels; i++) {
                            sample[i] = data[i] / 255.0f;
                        }
                        stbi_image_free(data);

                        samples_.push_back(std::move(sample));
                        labels_.push_back(label);

                        if (shape_.empty()) {
                            shape_ = {static_cast<size_t>(height),
                                     static_cast<size_t>(width),
                                     static_cast<size_t>(channels)};
                        }
                    }
                }
            }
            label++;
        }

        num_classes_ = class_names_.size();
        SetupSplits();
        return !samples_.empty();
    }

    bool LoadPredefinedDataset(const std::string& name) {
        // Normalize dataset name
        std::string normalized = name;
        std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);

        // Support for well-known Kaggle datasets
        if (normalized == "titanic" || normalized.find("titanic") != std::string::npos) {
            return CreateTitanicDataset();
        }
        else if (normalized == "iris" || normalized.find("iris") != std::string::npos) {
            return CreateIrisDataset();
        }
        else if (normalized == "fashionmnist" || normalized.find("fashion") != std::string::npos) {
            return CreateFashionMNISTDataset();
        }
        else if (normalized == "digits" || normalized.find("digit") != std::string::npos) {
            return CreateDigitsDataset();
        }

        return false;
    }

    bool CreateTitanicDataset() {
        // Simplified Titanic dataset (Pclass, Sex, Age, SibSp, Parch, Fare -> Survived)
        class_names_ = {"Died", "Survived"};
        num_classes_ = 2;
        shape_ = {6}; // 6 features

        // Generate synthetic data based on Titanic patterns
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        for (int i = 0; i < 891; i++) { // 891 samples like original
            std::vector<float> features(6);
            features[0] = static_cast<float>((i % 3) + 1) / 3.0f;  // Pclass (1-3)
            features[1] = (i % 2 == 0) ? 1.0f : 0.0f;               // Sex
            features[2] = (20.0f + dis(gen) * 50.0f) / 80.0f;       // Age
            features[3] = static_cast<float>(i % 4) / 4.0f;         // SibSp
            features[4] = static_cast<float>(i % 3) / 3.0f;         // Parch
            features[5] = dis(gen);                                  // Fare (normalized)

            samples_.push_back(features);
            // Survival probability based on class and sex
            int survived = (features[0] < 0.5f && features[1] > 0.5f) ? 1 : 0;
            if (dis(gen) < 0.3f) survived = 1 - survived; // Add noise
            labels_.push_back(survived);
        }

        SetupSplits();
        return true;
    }

    bool CreateIrisDataset() {
        class_names_ = {"Setosa", "Versicolor", "Virginica"};
        num_classes_ = 3;
        shape_ = {4}; // 4 features

        // Classic Iris dataset measurements
        float iris_data[][5] = {
            // Sepal L, Sepal W, Petal L, Petal W, Class
            {5.1f, 3.5f, 1.4f, 0.2f, 0}, {4.9f, 3.0f, 1.4f, 0.2f, 0}, {4.7f, 3.2f, 1.3f, 0.2f, 0},
            {4.6f, 3.1f, 1.5f, 0.2f, 0}, {5.0f, 3.6f, 1.4f, 0.2f, 0}, {5.4f, 3.9f, 1.7f, 0.4f, 0},
            {4.6f, 3.4f, 1.4f, 0.3f, 0}, {5.0f, 3.4f, 1.5f, 0.2f, 0}, {4.4f, 2.9f, 1.4f, 0.2f, 0},
            {4.9f, 3.1f, 1.5f, 0.1f, 0}, {7.0f, 3.2f, 4.7f, 1.4f, 1}, {6.4f, 3.2f, 4.5f, 1.5f, 1},
            {6.9f, 3.1f, 4.9f, 1.5f, 1}, {5.5f, 2.3f, 4.0f, 1.3f, 1}, {6.5f, 2.8f, 4.6f, 1.5f, 1},
            {5.7f, 2.8f, 4.5f, 1.3f, 1}, {6.3f, 3.3f, 4.7f, 1.6f, 1}, {4.9f, 2.4f, 3.3f, 1.0f, 1},
            {6.6f, 2.9f, 4.6f, 1.3f, 1}, {5.2f, 2.7f, 3.9f, 1.4f, 1}, {6.3f, 3.3f, 6.0f, 2.5f, 2},
            {5.8f, 2.7f, 5.1f, 1.9f, 2}, {7.1f, 3.0f, 5.9f, 2.1f, 2}, {6.3f, 2.9f, 5.6f, 1.8f, 2},
            {6.5f, 3.0f, 5.8f, 2.2f, 2}, {7.6f, 3.0f, 6.6f, 2.1f, 2}, {4.9f, 2.5f, 4.5f, 1.7f, 2},
            {7.3f, 2.9f, 6.3f, 1.8f, 2}, {6.7f, 2.5f, 5.8f, 1.8f, 2}, {7.2f, 3.6f, 6.1f, 2.5f, 2}
        };

        for (const auto& row : iris_data) {
            std::vector<float> features = {
                row[0] / 8.0f, row[1] / 5.0f, row[2] / 7.0f, row[3] / 3.0f
            };
            samples_.push_back(features);
            labels_.push_back(static_cast<int>(row[4]));
        }

        // Duplicate to get more samples
        size_t original_size = samples_.size();
        for (size_t i = 0; i < original_size * 4; i++) {
            size_t idx = i % original_size;
            std::vector<float> features = samples_[idx];
            // Add slight noise
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> noise(0.0f, 0.02f);
            for (auto& f : features) {
                f += noise(gen);
                f = std::max(0.0f, std::min(1.0f, f));
            }
            samples_.push_back(features);
            labels_.push_back(labels_[idx]);
        }

        SetupSplits();
        return true;
    }

    bool CreateFashionMNISTDataset() {
        class_names_ = {"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};
        num_classes_ = 10;
        shape_ = {28, 28, 1};

        // Generate synthetic Fashion-MNIST-like data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        for (int i = 0; i < 1000; i++) {
            std::vector<float> sample(784);
            int label = i % 10;

            // Create simple patterns for each class
            for (int j = 0; j < 784; j++) {
                int row = j / 28;
                int col = j % 28;
                float value = 0.0f;

                switch (label) {
                    case 0: // T-shirt shape
                        value = (row > 5 && row < 22 && col > 5 && col < 22) ? 0.8f : 0.1f;
                        break;
                    case 1: // Trouser shape
                        value = ((col > 8 && col < 14) || (col > 14 && col < 20)) && row > 5 ? 0.8f : 0.1f;
                        break;
                    default:
                        value = dis(gen) * 0.3f + (row == label || col == label ? 0.5f : 0.0f);
                }
                sample[j] = value + dis(gen) * 0.1f;
            }

            samples_.push_back(sample);
            labels_.push_back(label);
        }

        SetupSplits();
        return true;
    }

    bool CreateDigitsDataset() {
        class_names_ = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
        num_classes_ = 10;
        shape_ = {8, 8, 1}; // Sklearn digits format

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        for (int i = 0; i < 1797; i++) { // Same size as sklearn digits
            std::vector<float> sample(64);
            int label = i % 10;

            // Simple digit patterns
            for (int j = 0; j < 64; j++) {
                int row = j / 8;
                int col = j % 8;
                float value = 0.0f;

                // Create simple numeral patterns
                if (label == 0 && ((row == 0 || row == 7) && col > 1 && col < 6)) value = 0.8f;
                else if (label == 1 && col == 4) value = 0.8f;
                else if (label == 2 && (row == 0 || row == 3 || row == 7)) value = 0.7f;
                else value = dis(gen) * 0.2f;

                sample[j] = std::min(1.0f, value + dis(gen) * 0.15f);
            }

            samples_.push_back(sample);
            labels_.push_back(label);
        }

        SetupSplits();
        return true;
    }

    void SetupSplits() {
        all_indices_.resize(samples_.size());
        for (size_t i = 0; i < samples_.size(); i++) {
            all_indices_[i] = i;
        }

        // Default 80/10/10 split
        size_t train_end = static_cast<size_t>(samples_.size() * 0.8);
        size_t val_end = static_cast<size_t>(samples_.size() * 0.9);

        train_indices_.assign(all_indices_.begin(), all_indices_.begin() + train_end);
        val_indices_.assign(all_indices_.begin() + train_end, all_indices_.begin() + val_end);
        test_indices_.assign(all_indices_.begin() + val_end, all_indices_.end());
    }

    size_t CalculateMemoryUsage() const {
        size_t usage = 0;
        for (const auto& sample : samples_) {
            usage += sample.size() * sizeof(float);
        }
        usage += labels_.size() * sizeof(int);
        return usage;
    }

    KaggleConfig config_;
    std::vector<std::vector<float>> samples_;
    std::vector<int> labels_;
    std::vector<size_t> shape_;
    size_t num_classes_ = 0;
    std::vector<std::string> class_names_;
};

// =============================================================================
// DataRegistry Implementation
// =============================================================================

DataRegistry& DataRegistry::Instance() {
    static DataRegistry instance;
    return instance;
}

std::string DataRegistry::GenerateUniqueName(const std::string& base_name) {
    std::string name = base_name;
    if (name.empty()) {
        name = "dataset";
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (datasets_.find(name) == datasets_.end()) {
        return name;
    }

    // Add suffix to make unique
    int suffix = 1;
    while (datasets_.find(name + "_" + std::to_string(suffix)) != datasets_.end()) {
        suffix++;
    }
    return name + "_" + std::to_string(suffix);
}

DatasetHandle DataRegistry::LoadDataset(const std::string& path, const std::string& name) {
    DatasetType type = DetectType(path);

    switch (type) {
        case DatasetType::MNIST:
            return LoadMNIST(path, name.empty() ? "mnist" : name);
        case DatasetType::CIFAR10:
            return LoadCIFAR10(path, name.empty() ? "cifar10" : name);
        case DatasetType::CSV:
            return LoadCSV(path, name);
        case DatasetType::TSV:
            return LoadTSV(path, name);
        case DatasetType::JSON:
            return LoadJSON(path, name);
        case DatasetType::TXT:
            return LoadTXT(path, name);
        case DatasetType::ImageFolder:
            return LoadImageFolder(path, name);
        case DatasetType::HDF5:
            return LoadHDF5(path, name);
        default:
            spdlog::error("Unknown dataset type for path: {}", path);
            return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadMNIST(const std::string& path, const std::string& name) {
    std::string unique_name = GenerateUniqueName(name);

    try {
        auto dataset = std::make_shared<MNISTDataset>(path);
        if (dataset->Size() == 0) {
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered MNIST dataset as '{}'", unique_name);
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load MNIST dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadCIFAR10(const std::string& path, const std::string& name) {
    std::string unique_name = GenerateUniqueName(name);

    try {
        auto dataset = std::make_shared<CIFAR10Dataset>(path);
        if (dataset->Size() == 0) {
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered CIFAR-10 dataset as '{}'", unique_name);
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load CIFAR-10 dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadCSV(const std::string& path, const std::string& name) {
    std::string base_name = name.empty() ? fs::path(path).stem().string() : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<CSVDataset>(path);
        if (dataset->Size() == 0) {
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered CSV dataset as '{}'", unique_name);
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load CSV dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadTSV(const std::string& path, const std::string& name) {
    std::string base_name = name.empty() ? fs::path(path).stem().string() : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<TSVDataset>(path);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered TSV dataset as '{}'", unique_name);
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load TSV dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadJSON(const std::string& path, const std::string& name) {
    std::string base_name = name.empty() ? fs::path(path).stem().string() : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<JSONDataset>(path);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered JSON dataset as '{}'", unique_name);
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load JSON dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadTXT(const std::string& path, const std::string& name) {
    std::string base_name = name.empty() ? fs::path(path).stem().string() : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<TXTDataset>(path);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered TXT dataset as '{}'", unique_name);
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load TXT dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadImageFolder(const std::string& path, const std::string& name) {
    std::string base_name = name.empty() ? fs::path(path).filename().string() : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<ImageFolderDataset>(path);
        if (dataset->Size() == 0) {
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered ImageFolder dataset as '{}'", unique_name);
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load ImageFolder dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadImageCSV(const std::string& image_folder, const std::string& csv_path,
                                          const std::string& name, int target_width, int target_height,
                                          size_t cache_size) {
    std::string base_name = name.empty() ? fs::path(image_folder).filename().string() : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<ImageCSVDataset>(image_folder, csv_path, target_width, target_height, cache_size);
        if (dataset->Size() == 0) {
            spdlog::warn("ImageCSV dataset loaded with 0 samples");
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered ImageCSV dataset as '{}' with {} samples", unique_name, dataset->Size());
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load ImageCSV dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadHuggingFace(const HuggingFaceConfig& config, const std::string& name) {
    std::string base_name = name.empty() ? config.dataset_name : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<HuggingFaceDataset>(config);
        if (dataset->Size() == 0) {
            spdlog::warn("HuggingFace dataset '{}' loaded with 0 samples", config.dataset_name);
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered HuggingFace dataset '{}' as '{}'", config.dataset_name, unique_name);
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load HuggingFace dataset '{}': {}", config.dataset_name, e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadStreamingDataset(const std::string& path, const StreamingConfig& config, const std::string& name) {
    std::string base_name = name.empty() ? fs::path(path).stem().string() : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<StreamingDataset>(path, config);
        if (dataset->Size() == 0) {
            spdlog::warn("Streaming dataset '{}' has estimated 0 samples", path);
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered streaming dataset as '{}' ({} estimated samples)",
                    unique_name, dataset->Size());
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load streaming dataset '{}': {}", path, e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadKaggle(const KaggleConfig& config, const std::string& name) {
    std::string base_name = name;
    if (base_name.empty()) {
        // Extract name from dataset_slug or competition
        if (!config.dataset_slug.empty()) {
            size_t pos = config.dataset_slug.find_last_of('/');
            base_name = (pos != std::string::npos) ? config.dataset_slug.substr(pos + 1) : config.dataset_slug;
        } else {
            base_name = config.competition;
        }
    }
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<KaggleDataset>(config);
        if (dataset->Size() == 0) {
            spdlog::warn("Kaggle dataset loaded with 0 samples");
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered Kaggle dataset as '{}' ({} samples)",
                    unique_name, dataset->Size());
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load Kaggle dataset: {}", e.what());
        return DatasetHandle();
    }
}

// =============================================================================
// Custom Dataset Implementation
// =============================================================================

class CustomDataset : public Dataset {
public:
    CustomDataset(const CustomConfig& config) : config_(config) {
        // Detect format if not specified or set to "auto"
        if (config_.format.empty() || config_.format == "auto") {
            config_.format = DetectFormat(config_.data_path);
        }

        // Load based on format
        if (config_.format == "json") {
            LoadJSON();
        } else if (config_.format == "csv" || config_.format == "text") {
            LoadText(",");
        } else if (config_.format == "tsv") {
            LoadText("\t");
        } else if (config_.format == "binary" || config_.format == "bin") {
            LoadBinary();
        } else if (config_.format == "folder") {
            LoadFolder();
        } else {
            // Try to create sample data for testing
            spdlog::warn("Unknown format '{}', creating sample data", config_.format);
            CreateSampleData();
        }

        // Auto-detect number of classes
        if (config_.num_classes == 0 && !labels_.empty()) {
            int max_label = *std::max_element(labels_.begin(), labels_.end());
            config_.num_classes = static_cast<size_t>(max_label + 1);
        }

        // Set up default split
        all_indices_.resize(data_.size());
        std::iota(all_indices_.begin(), all_indices_.end(), 0);
        SetSplit(split_config_);

        spdlog::info("CustomDataset loaded: {} samples, {} classes, format={}",
                     data_.size(), config_.num_classes, config_.format);
    }

    size_t Size() const override { return data_.size(); }

    std::pair<std::vector<float>, int> GetItem(size_t index) const override {
        if (index >= data_.size()) {
            return {{}, -1};
        }
        return {data_[index], labels_[index]};
    }

    DatasetInfo GetInfo() const override {
        DatasetInfo info;
        info.name = "custom";
        info.path = config_.data_path;
        info.type = DatasetType::Custom;
        info.shape = config_.shape;
        info.num_samples = data_.size();
        info.num_classes = config_.num_classes;
        info.class_names = config_.class_names;
        info.train_count = train_indices_.size();
        info.val_count = val_indices_.size();
        info.test_count = test_indices_.size();
        info.is_loaded = true;

        // Estimate memory
        size_t sample_size = 1;
        for (auto s : config_.shape) sample_size *= s;
        info.memory_usage = data_.size() * sample_size * sizeof(float);

        return info;
    }

    std::vector<std::string> GetColumnNames() const override { return column_names_; }

    bool HasFloatLabels() const override { return !float_labels_.empty(); }

    float GetFloatLabel(size_t index) const override {
        if (index < float_labels_.size()) return float_labels_[index];
        return 0.0f;
    }

    int GetLabelColumnIndex() const override { return resolved_label_col_; }
    int GetOriginalColumnCount() const override { return original_col_count_; }

private:
    std::vector<std::string> column_names_;

    std::string DetectFormat(const std::string& path) {
        namespace fs = std::filesystem;
        fs::path p(path);

        if (fs::is_directory(p)) {
            return "folder";
        }

        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".json") return "json";
        if (ext == ".csv") return "csv";
        if (ext == ".tsv") return "tsv";
        if (ext == ".txt") return "text";
        if (ext == ".bin" || ext == ".dat") return "binary";
        if (ext == ".npy" || ext == ".npz") return "npy";

        return "unknown";
    }

    void LoadJSON() {
        std::ifstream file(config_.data_path);
        if (!file.is_open()) {
            spdlog::error("Failed to open JSON file: {}", config_.data_path);
            CreateSampleData();
            return;
        }

        try {
            nlohmann::json j;
            file >> j;

            // Get data array
            std::string data_key = config_.data_key.empty() ? "data" : config_.data_key;
            std::string labels_key = config_.labels_key.empty() ? "labels" : config_.labels_key;

            if (!j.contains(data_key)) {
                // Try alternative keys
                for (const auto& key : {"images", "X", "features", "samples"}) {
                    if (j.contains(key)) {
                        data_key = key;
                        break;
                    }
                }
            }

            if (!j.contains(labels_key)) {
                for (const auto& key : {"targets", "y", "classes"}) {
                    if (j.contains(key)) {
                        labels_key = key;
                        break;
                    }
                }
            }

            if (j.contains(data_key)) {
                auto& data_arr = j[data_key];
                for (const auto& sample : data_arr) {
                    std::vector<float> flat_sample;
                    FlattenJSON(sample, flat_sample);
                    data_.push_back(flat_sample);
                }
            }

            if (j.contains(labels_key)) {
                auto& labels_arr = j[labels_key];
                for (const auto& label : labels_arr) {
                    labels_.push_back(label.get<int>());
                }
            }

            // Infer shape from first sample
            if (!data_.empty() && config_.shape.empty()) {
                config_.shape = {data_[0].size()};
            }

            // Ensure labels match data size
            while (labels_.size() < data_.size()) {
                labels_.push_back(0);
            }

        } catch (const std::exception& e) {
            spdlog::error("JSON parse error: {}", e.what());
            CreateSampleData();
        }
    }

    void FlattenJSON(const nlohmann::json& j, std::vector<float>& out) {
        if (j.is_array()) {
            for (const auto& elem : j) {
                FlattenJSON(elem, out);
            }
        } else if (j.is_number()) {
            float val = j.get<float>();
            if (config_.normalize && config_.scale != 1.0f) {
                val *= config_.scale;
            }
            out.push_back(val);
        }
    }

    void LoadText(const std::string& delimiter) {
        std::ifstream file(config_.data_path);
        if (!file.is_open()) {
            spdlog::error("Failed to open text file: {}", config_.data_path);
            CreateSampleData();
            return;
        }

        std::string line;
        bool first_line = true;

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            // Parse or skip header
            if (first_line && config_.has_header) {
                first_line = false;
                // Parse header names for column display
                std::string delim = config_.delimiter.empty() ? delimiter : config_.delimiter;
                std::string remaining = line;
                size_t pos = 0;
                while ((pos = remaining.find(delim)) != std::string::npos) {
                    std::string tok = remaining.substr(0, pos);
                    tok.erase(0, tok.find_first_not_of(" \t\r\n"));
                    if (!tok.empty()) tok.erase(tok.find_last_not_of(" \t\r\n") + 1);
                    column_names_.push_back(tok);
                    remaining = remaining.substr(pos + delim.length());
                }
                remaining.erase(0, remaining.find_first_not_of(" \t\r\n"));
                if (!remaining.empty()) remaining.erase(remaining.find_last_not_of(" \t\r\n") + 1);
                column_names_.push_back(remaining);
                continue;
            }
            first_line = false;

            std::vector<float> sample;
            std::stringstream ss(line);
            std::string token;
            std::vector<std::string> tokens;

            // Split by delimiter
            size_t pos = 0;
            std::string delim = config_.delimiter.empty() ? delimiter : config_.delimiter;
            std::string remaining = line;
            while ((pos = remaining.find(delim)) != std::string::npos) {
                tokens.push_back(remaining.substr(0, pos));
                remaining = remaining.substr(pos + delim.length());
            }
            tokens.push_back(remaining);

            // Determine label column (-2 = no label, -1 = last, >= 0 = specific)
            int label_col = config_.label_column;
            if (label_col == -1) {
                label_col = static_cast<int>(tokens.size()) - 1;
            }
            bool no_label = (label_col == -2);

            // Store original file layout info (once)
            if (original_col_count_ == 0) {
                original_col_count_ = static_cast<int>(tokens.size());
                resolved_label_col_ = no_label ? -2 : label_col;
            }

            // Parse values
            int label = 0;
            float float_label = 0.0f;
            for (size_t i = 0; i < tokens.size(); i++) {
                try {
                    float val = std::stof(tokens[i]);
                    if (!no_label && static_cast<int>(i) == label_col) {
                        label = static_cast<int>(val);
                        float_label = val;
                    } else {
                        if (config_.normalize && config_.scale != 1.0f) {
                            val *= config_.scale;
                        }
                        sample.push_back(val);
                    }
                } catch (...) {
                    // Skip non-numeric values
                }
            }

            if (!sample.empty()) {
                data_.push_back(sample);
                labels_.push_back(label);
                float_labels_.push_back(float_label);
            }
        }

        // Infer shape
        if (!data_.empty() && config_.shape.empty()) {
            config_.shape = {data_[0].size()};
        }

        // Generate column names if not from header
        if (column_names_.empty() && !data_.empty()) {
            for (size_t i = 0; i < data_[0].size(); i++) {
                column_names_.push_back("Feature_" + std::to_string(i));
            }
        }
    }

    void LoadBinary() {
        std::ifstream file(config_.data_path, std::ios::binary);
        if (!file.is_open()) {
            spdlog::error("Failed to open binary file: {}", config_.data_path);
            CreateSampleData();
            return;
        }

        // Read header (simple format: num_samples, sample_size, num_classes)
        uint32_t num_samples = 0, sample_size = 0, num_classes = 0;

        // Check for magic number (optional CYXD format)
        char magic[4];
        file.read(magic, 4);
        if (std::string(magic, 4) == "CYXD") {
            // CyxWiz Dataset format
            file.read(reinterpret_cast<char*>(&num_samples), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&sample_size), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&num_classes), sizeof(uint32_t));
        } else {
            // Raw format - assume floats with shape from config
            file.seekg(0);
            if (!config_.shape.empty()) {
                sample_size = 1;
                for (auto s : config_.shape) sample_size *= static_cast<uint32_t>(s);

                // Calculate num_samples from file size
                file.seekg(0, std::ios::end);
                size_t file_size = file.tellg();
                file.seekg(0);
                num_samples = static_cast<uint32_t>(file_size / (sample_size * sizeof(float)));
            }
        }

        // Read data
        for (uint32_t i = 0; i < num_samples; i++) {
            std::vector<float> sample(sample_size);
            file.read(reinterpret_cast<char*>(sample.data()), sample_size * sizeof(float));

            if (config_.normalize && config_.scale != 1.0f) {
                for (auto& v : sample) v *= config_.scale;
            }

            data_.push_back(sample);
            labels_.push_back(0);  // Labels loaded separately or from file
        }

        // Try to load labels from separate file
        if (!config_.labels_path.empty()) {
            std::ifstream lfile(config_.labels_path, std::ios::binary);
            if (lfile.is_open()) {
                for (size_t i = 0; i < data_.size(); i++) {
                    int32_t label;
                    if (lfile.read(reinterpret_cast<char*>(&label), sizeof(int32_t))) {
                        labels_[i] = label;
                    }
                }
            }
        }

        config_.num_classes = num_classes;
    }

    void LoadFolder() {
        namespace fs = std::filesystem;

        fs::path root(config_.data_path);
        if (!fs::is_directory(root)) {
            spdlog::error("Not a directory: {}", config_.data_path);
            CreateSampleData();
            return;
        }

        // Each subdirectory is a class
        std::map<std::string, int> class_map;
        int class_idx = 0;

        for (const auto& entry : fs::directory_iterator(root)) {
            if (entry.is_directory()) {
                std::string class_name = entry.path().filename().string();
                class_map[class_name] = class_idx;
                config_.class_names.push_back(class_name);

                // Load files in this class directory
                for (const auto& file : fs::directory_iterator(entry.path())) {
                    if (file.is_regular_file()) {
                        // For now, just record the path and label
                        // Full image loading would require stb_image
                        std::vector<float> sample = {static_cast<float>(class_idx)};
                        data_.push_back(sample);
                        labels_.push_back(class_idx);
                    }
                }

                class_idx++;
            }
        }

        config_.num_classes = class_idx;
        spdlog::info("Loaded folder dataset: {} classes, {} samples",
                     class_idx, data_.size());
    }

    void CreateSampleData() {
        // Create synthetic data for testing
        spdlog::info("Creating sample data for testing");

        size_t sample_size = 784;  // Default to MNIST-like
        if (!config_.shape.empty()) {
            sample_size = 1;
            for (auto s : config_.shape) sample_size *= s;
        } else {
            config_.shape = {28, 28, 1};
        }

        config_.num_classes = 10;
        size_t num_samples = 100;

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::uniform_int_distribution<int> label_dist(0, 9);

        for (size_t i = 0; i < num_samples; i++) {
            std::vector<float> sample(sample_size);
            for (auto& v : sample) {
                v = dist(rng);
            }
            data_.push_back(sample);
            labels_.push_back(label_dist(rng));
        }
    }

    CustomConfig config_;
    std::vector<std::vector<float>> data_;
    std::vector<int> labels_;
    std::vector<float> float_labels_;  // Raw float labels for regression data
    int resolved_label_col_ = -2;     // Actual label column index in original file
    int original_col_count_ = 0;      // Total columns in original file
};

DatasetHandle DataRegistry::LoadCustom(const CustomConfig& config, const std::string& name) {
    try {
        std::string unique_name = name.empty() ? GenerateUniqueName("custom") : name;

        // Check if already loaded
        if (HasDataset(unique_name)) {
            spdlog::warn("Dataset '{}' already loaded, returning existing", unique_name);
            return GetDataset(unique_name);
        }

        spdlog::info("Loading custom dataset from: {}", config.data_path);

        auto dataset = std::make_shared<CustomDataset>(config);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered custom dataset as '{}' ({} samples)",
                    unique_name, dataset->Size());
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load custom dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadHDF5(const std::string& path, const std::string& name,
                                      const HDF5Config& config) {
#ifdef CYXWIZ_HAS_HDF5
    std::string base_name = name.empty() ? fs::path(path).stem().string() : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<HDF5Dataset>(path, config);
        if (dataset->Size() == 0) {
            spdlog::warn("HDF5 dataset is empty: {}", path);
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered HDF5 dataset '{}': {} samples", unique_name, dataset->Size());
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load HDF5 dataset: {}", e.what());
        return DatasetHandle();
    }
#else
    spdlog::error("HDF5 support not compiled (HighFive library missing)");
    return DatasetHandle();
#endif
}

bool DataRegistry::ExportHDF5(const std::string& name, const std::string& filepath,
                               const HDF5ExportConfig& config) {
    auto handle = GetDataset(name);
    if (!handle.IsValid()) {
        spdlog::error("ExportHDF5: Dataset '{}' not found", name);
        return false;
    }
    return ExportHDF5(handle, filepath, config);
}

bool DataRegistry::ExportHDF5(DatasetHandle handle, const std::string& filepath,
                               const HDF5ExportConfig& config) {
#ifdef CYXWIZ_HAS_HDF5
    if (!handle.IsValid()) {
        spdlog::error("ExportHDF5: Invalid dataset handle");
        return false;
    }

    try {
        auto info = handle.GetInfo();
        spdlog::info("Exporting dataset to HDF5: {} ({} samples)", filepath, info.num_samples);

        // Create HDF5 file
        HighFive::File file(filepath, HighFive::File::Overwrite);

        // Get sample shape
        std::vector<size_t> sample_shape = info.shape;
        if (sample_shape.empty()) {
            sample_shape = {1};  // Scalar samples
        }

        // Calculate total data shape: [N, ...sample_shape]
        std::vector<size_t> data_shape;
        data_shape.push_back(info.num_samples);

        // Handle NCHW transpose if requested
        bool do_nchw = config.store_as_nchw && sample_shape.size() == 3;
        if (do_nchw) {
            // Input is NHWC [H, W, C], output should be NCHW [C, H, W]
            data_shape.push_back(sample_shape[2]);  // C
            data_shape.push_back(sample_shape[0]);  // H
            data_shape.push_back(sample_shape[1]);  // W
        } else {
            for (auto d : sample_shape) {
                data_shape.push_back(d);
            }
        }

        // Set up chunking
        std::vector<size_t> chunk_dims = data_shape;
        if (config.chunked && info.num_samples > config.chunk_samples) {
            chunk_dims[0] = std::min(config.chunk_samples, info.num_samples);
        }

        // Create property list for compression and chunking
        HighFive::DataSetCreateProps props;
        if (config.chunked) {
            props.add(HighFive::Chunking(chunk_dims));
        }
        if (config.compress && config.chunked) {
            props.add(HighFive::Deflate(config.compression_level));
        }

        // Calculate sample size
        size_t sample_size = 1;
        for (auto d : sample_shape) {
            sample_size *= d;
        }

        // Read all data from dataset
        spdlog::info("Reading {} samples from dataset...", info.num_samples);
        std::vector<float> all_data;
        all_data.reserve(info.num_samples * sample_size);

        std::vector<int> all_labels;
        all_labels.reserve(info.num_samples);

        for (size_t i = 0; i < info.num_samples; i++) {
            auto [data, label] = handle.GetSample(i);

            // Transpose NHWC to NCHW if requested
            if (do_nchw && sample_shape.size() == 3) {
                size_t H = sample_shape[0];
                size_t W = sample_shape[1];
                size_t C = sample_shape[2];
                std::vector<float> transposed(sample_size);

                // NHWC [h, w, c] -> NCHW [c, h, w]
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        for (size_t c = 0; c < C; c++) {
                            size_t src_idx = h * W * C + w * C + c;  // NHWC
                            size_t dst_idx = c * H * W + h * W + w;  // NCHW
                            transposed[dst_idx] = data[src_idx];
                        }
                    }
                }
                all_data.insert(all_data.end(), transposed.begin(), transposed.end());
            } else {
                all_data.insert(all_data.end(), data.begin(), data.end());
            }

            all_labels.push_back(label);

            if ((i + 1) % 1000 == 0) {
                spdlog::info("Read {}/{} samples", i + 1, info.num_samples);
            }
        }

        // Create and write data dataset
        if (config.store_as_uint8) {
            // Convert float [0,1] to uint8 [0,255]
            std::vector<uint8_t> uint8_data(all_data.size());
            for (size_t i = 0; i < all_data.size(); i++) {
                float val = std::clamp(all_data[i], 0.0f, 1.0f) * 255.0f;
                uint8_data[i] = static_cast<uint8_t>(val);
            }

            auto data_ds = file.createDataSet<uint8_t>(config.data_path,
                HighFive::DataSpace(data_shape), props);
            data_ds.write_raw(uint8_data.data());
            spdlog::info("Wrote data as uint8 to {}", config.data_path);
        } else {
            auto data_ds = file.createDataSet<float>(config.data_path,
                HighFive::DataSpace(data_shape), props);
            data_ds.write_raw(all_data.data());
            spdlog::info("Wrote data as float32 to {}", config.data_path);
        }

        // Create and write labels dataset
        auto label_ds = file.createDataSet<int>(config.label_path,
            HighFive::DataSpace({info.num_samples}));
        label_ds.write(all_labels);
        spdlog::info("Wrote {} labels to {}", info.num_samples, config.label_path);

        // Write metadata
        if (config.include_metadata) {
            auto root = file.getGroup("/");

            // Number of samples
            root.createAttribute("num_samples", info.num_samples);

            // Number of classes
            root.createAttribute("num_classes", info.num_classes);

            // Sample shape
            if (!sample_shape.empty()) {
                root.createAttribute("sample_shape", sample_shape);
            }

            // Data format
            std::string format = do_nchw ? "NCHW" : "NHWC";
            root.createAttribute("data_format", format);

            // Class names if provided
            if (!config.class_names.empty()) {
                root.createAttribute("class_names", config.class_names);
            }

            spdlog::info("Wrote metadata attributes");
        }

        spdlog::info("Successfully exported dataset to: {}", filepath);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to export HDF5 dataset: {}", e.what());
        return false;
    }
#else
    spdlog::error("HDF5 support not compiled (HighFive library missing)");
    return false;
#endif
}

void DataRegistry::UnloadDataset(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = datasets_.find(name);
    if (it != datasets_.end()) {
        datasets_.erase(it);
        spdlog::info("Unloaded dataset: {}", name);

        if (on_unloaded_) {
            on_unloaded_(name);
        }
    }
}

void DataRegistry::UnloadAll() {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> names;
    for (const auto& [name, _] : datasets_) {
        names.push_back(name);
    }

    datasets_.clear();

    for (const auto& name : names) {
        if (on_unloaded_) {
            on_unloaded_(name);
        }
    }

    spdlog::info("Unloaded all datasets");
}

DatasetHandle DataRegistry::GetDataset(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = datasets_.find(name);
    if (it != datasets_.end()) {
        // Update LRU access time
        last_access_times_[name] = std::chrono::steady_clock::now();
        return DatasetHandle(it->second, name);
    }
    return DatasetHandle();
}

bool DataRegistry::HasDataset(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return datasets_.find(name) != datasets_.end();
}

std::vector<DatasetInfo> DataRegistry::ListDatasets() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<DatasetInfo> result;
    result.reserve(datasets_.size());

    for (const auto& [name, dataset] : datasets_) {
        auto info = dataset->GetInfo();
        info.name = name;
        result.push_back(info);
    }

    return result;
}

std::vector<std::string> DataRegistry::GetDatasetNames() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> names;
    names.reserve(datasets_.size());

    for (const auto& [name, _] : datasets_) {
        names.push_back(name);
    }

    return names;
}

DatasetPreview DataRegistry::GetPreview(const std::string& path, int max_samples) {
    DatasetPreview preview;
    preview.type = DetectType(path);

    if (!fs::exists(path)) {
        return preview;
    }

    // Get file size
    if (fs::is_regular_file(path)) {
        preview.file_size = fs::file_size(path);
    } else if (fs::is_directory(path)) {
        for (const auto& entry : fs::recursive_directory_iterator(path)) {
            if (entry.is_regular_file()) {
                preview.file_size += entry.file_size();
            }
        }
    }

    // Generate preview based on type
    switch (preview.type) {
        case DatasetType::CSV: {
            std::ifstream file(path);
            if (!file) return preview;

            std::string line;
            int line_count = 0;

            while (std::getline(file, line) && line_count <= max_samples) {
                std::vector<std::string> tokens;
                std::stringstream ss(line);
                std::string token;

                while (std::getline(ss, token, ',')) {
                    token.erase(0, token.find_first_not_of(" \t\r\n"));
                    token.erase(token.find_last_not_of(" \t\r\n") + 1);
                    tokens.push_back(token);
                }

                if (line_count == 0) {
                    // Check if header
                    try {
                        (void)std::stof(tokens[0]);  // Just checking if it's numeric
                        preview.rows.push_back(tokens);
                    } catch (...) {
                        preview.columns = tokens;
                    }
                } else {
                    preview.rows.push_back(tokens);
                }
                line_count++;
            }

            // Count total lines
            file.clear();
            file.seekg(0);
            preview.num_samples = std::count(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(), '\n');
            if (!preview.columns.empty()) preview.num_samples--;

            break;
        }

        case DatasetType::MNIST: {
            preview.shape = {28, 28, 1};
            preview.num_classes = 10;

            // Quick count from header
            std::string images_file = path + "/train-images-idx3-ubyte";
            if (!fs::exists(images_file)) {
                images_file = path + "/train-images.idx3-ubyte";
            }

            if (fs::exists(images_file)) {
                std::ifstream file(images_file, std::ios::binary);
                file.seekg(4);  // Skip magic

                uint32_t num;
                file.read(reinterpret_cast<char*>(&num), 4);
                // Convert from big-endian
                preview.num_samples = ((num & 0xFF) << 24) | ((num & 0xFF00) << 8) |
                                     ((num & 0xFF0000) >> 8) | ((num & 0xFF000000) >> 24);
            }
            break;
        }

        case DatasetType::CIFAR10: {
            preview.shape = {32, 32, 3};
            preview.num_classes = 10;
            preview.num_samples = 50000;  // Standard CIFAR-10 training set
            break;
        }

        default:
            break;
    }

    return preview;
}

DatasetType DataRegistry::DetectType(const std::string& path) {
    if (!fs::exists(path)) {
        return DatasetType::None;
    }

    // Check for directory-based datasets
    if (fs::is_directory(path)) {
        // MNIST
        if (fs::exists(path + "/train-images-idx3-ubyte") ||
            fs::exists(path + "/train-images.idx3-ubyte")) {
            return DatasetType::MNIST;
        }

        // CIFAR-10
        if (fs::exists(path + "/data_batch_1.bin")) {
            return DatasetType::CIFAR10;
        }

        // Check for image folder structure
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.is_directory()) {
                for (const auto& sub : fs::directory_iterator(entry)) {
                    auto ext = sub.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                        return DatasetType::ImageFolder;
                    }
                }
            }
        }
    } else {
        // File-based datasets
        auto ext = fs::path(path).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".csv") return DatasetType::CSV;
        if (ext == ".tsv") return DatasetType::TSV;
        if (ext == ".json") return DatasetType::JSON;
        if (ext == ".txt") return DatasetType::TXT;
        if (ext == ".h5" || ext == ".hdf5" || ext == ".hdf") return DatasetType::HDF5;
    }

    return DatasetType::None;
}

std::string DataRegistry::TypeToString(DatasetType type) {
    switch (type) {
        case DatasetType::None: return "None";
        case DatasetType::CSV: return "CSV";
        case DatasetType::TSV: return "TSV";
        case DatasetType::JSON: return "JSON";
        case DatasetType::TXT: return "TXT";
        case DatasetType::ImageFolder: return "ImageFolder";
        case DatasetType::ImageCSV: return "ImageCSV";
        case DatasetType::MNIST: return "MNIST";
        case DatasetType::FashionMNIST: return "FashionMNIST";
        case DatasetType::CIFAR10: return "CIFAR-10";
        case DatasetType::CIFAR100: return "CIFAR-100";
        case DatasetType::HuggingFace: return "HuggingFace";
        case DatasetType::Kaggle: return "Kaggle";
        case DatasetType::Custom: return "Custom";
        case DatasetType::HDF5: return "HDF5";
        default: return "Unknown";
    }
}

size_t DataRegistry::GetTotalMemoryUsage() const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total = 0;
    for (const auto& [_, dataset] : datasets_) {
        total += dataset->GetInfo().memory_usage;
    }
    return total;
}

void DataRegistry::SetMemoryLimit(size_t bytes) {
    memory_limit_ = bytes;
}

MemoryStats DataRegistry::GetMemoryStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    MemoryStats stats;
    stats.memory_limit = memory_limit_;
    stats.datasets_count = datasets_.size();
    stats.cache_hits = total_cache_hits_;
    stats.cache_misses = total_cache_misses_;
    stats.cache_evictions = total_cache_evictions_;

    // Sum up memory from all datasets
    for (const auto& [_, dataset] : datasets_) {
        auto info = dataset->GetInfo();
        stats.total_allocated += info.memory_usage;
        stats.total_cached += info.cache_usage;
    }

    // Update peak usage
    if (stats.total_allocated > peak_usage_) {
        peak_usage_ = stats.total_allocated;
    }
    stats.peak_usage = peak_usage_;

    // Get texture memory from TextureManager
    // (Note: TextureManager tracks its own memory)
    stats.texture_memory = 0;  // Will be set by caller if needed
    stats.texture_count = 0;

    return stats;
}

void DataRegistry::ResetCacheStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    total_cache_hits_ = 0;
    total_cache_misses_ = 0;
    total_cache_evictions_ = 0;
}

// =============================================================================
// Memory Optimization
// =============================================================================

bool DataRegistry::IsMemoryPressure() const {
    size_t current = GetTotalMemoryUsage();
    return current >= static_cast<size_t>(memory_limit_ * memory_pressure_threshold_);
}

void DataRegistry::EvictOldest() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (datasets_.empty()) {
        return;
    }

    // Find the least recently accessed dataset
    std::string oldest_name;
    auto oldest_time = std::chrono::steady_clock::time_point::max();

    for (const auto& [name, _] : datasets_) {
        auto it = last_access_times_.find(name);
        auto access_time = (it != last_access_times_.end())
            ? it->second
            : std::chrono::steady_clock::time_point::min();  // Never accessed = oldest

        if (access_time < oldest_time) {
            oldest_time = access_time;
            oldest_name = name;
        }
    }

    if (!oldest_name.empty()) {
        auto info = datasets_[oldest_name]->GetInfo();
        spdlog::info("Memory eviction: unloading '{}' ({} bytes)", oldest_name, info.memory_usage);

        datasets_.erase(oldest_name);
        last_access_times_.erase(oldest_name);
        total_cache_evictions_++;

        if (on_unloaded_) {
            on_unloaded_(oldest_name);
        }
    }
}

void DataRegistry::TrimMemory(size_t target_bytes) {
    // If target is 0, use memory_limit_
    size_t target = (target_bytes > 0) ? target_bytes : memory_limit_;

    size_t current = GetTotalMemoryUsage();

    // Notify about memory pressure if callback is set
    if (current > memory_limit_ && on_memory_pressure_) {
        on_memory_pressure_(current, memory_limit_);
    }

    // Keep evicting until we're under target
    int eviction_count = 0;
    while (current > target && !datasets_.empty()) {
        EvictOldest();
        current = GetTotalMemoryUsage();
        eviction_count++;

        // Safety limit to prevent infinite loop
        if (eviction_count > 100) {
            spdlog::warn("TrimMemory: Safety limit reached after {} evictions", eviction_count);
            break;
        }
    }

    if (eviction_count > 0) {
        spdlog::info("TrimMemory: Evicted {} datasets, new usage: {} bytes", eviction_count, current);
    }
}

// =============================================================================
// Configuration Export/Import
// =============================================================================

std::string DataRegistry::SerializeConfig(const DatasetInfo& info, const SplitConfig& split) {
    nlohmann::json j;

    // Dataset info
    j["name"] = info.name;
    j["path"] = info.path;
    j["type"] = TypeToString(info.type);
    j["shape"] = info.shape;
    j["num_samples"] = info.num_samples;
    j["num_classes"] = info.num_classes;
    j["class_names"] = info.class_names;

    // Split config
    j["split"]["train_ratio"] = split.train_ratio;
    j["split"]["val_ratio"] = split.val_ratio;
    j["split"]["test_ratio"] = split.test_ratio;
    j["split"]["stratified"] = split.stratified;
    j["split"]["shuffle"] = split.shuffle;
    j["split"]["seed"] = split.seed;

    // Metadata
    j["version"] = "1.0";
    j["exported_at"] = std::time(nullptr);

    return j.dump(2);
}

bool DataRegistry::DeserializeConfig(const std::string& json_str, DatasetInfo& info, SplitConfig& split) {
    try {
        nlohmann::json j = nlohmann::json::parse(json_str);

        // Dataset info
        info.name = j.value("name", "");
        info.path = j.value("path", "");

        std::string type_str = j.value("type", "None");
        if (type_str == "CSV") info.type = DatasetType::CSV;
        else if (type_str == "TSV") info.type = DatasetType::TSV;
        else if (type_str == "ImageFolder") info.type = DatasetType::ImageFolder;
        else if (type_str == "ImageCSV") info.type = DatasetType::ImageCSV;
        else if (type_str == "MNIST") info.type = DatasetType::MNIST;
        else if (type_str == "FashionMNIST") info.type = DatasetType::FashionMNIST;
        else if (type_str == "CIFAR10") info.type = DatasetType::CIFAR10;
        else if (type_str == "CIFAR100") info.type = DatasetType::CIFAR100;
        else if (type_str == "HuggingFace") info.type = DatasetType::HuggingFace;
        else if (type_str == "Kaggle") info.type = DatasetType::Kaggle;
        else if (type_str == "Custom") info.type = DatasetType::Custom;
        else info.type = DatasetType::None;

        if (j.contains("shape")) {
            info.shape = j["shape"].get<std::vector<size_t>>();
        }
        info.num_samples = j.value("num_samples", size_t(0));
        info.num_classes = j.value("num_classes", size_t(0));
        if (j.contains("class_names")) {
            info.class_names = j["class_names"].get<std::vector<std::string>>();
        }

        // Split config
        if (j.contains("split")) {
            auto& s = j["split"];
            split.train_ratio = s.value("train_ratio", 0.8f);
            split.val_ratio = s.value("val_ratio", 0.1f);
            split.test_ratio = s.value("test_ratio", 0.1f);
            split.stratified = s.value("stratified", true);
            split.shuffle = s.value("shuffle", true);
            split.seed = s.value("seed", 42);
        }

        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to deserialize dataset config: {}", e.what());
        return false;
    }
}

bool DataRegistry::ExportConfig(const std::string& name, const std::string& filepath) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = datasets_.find(name);
    if (it == datasets_.end()) {
        spdlog::error("Cannot export config: dataset '{}' not found", name);
        return false;
    }

    DatasetInfo info = it->second->GetInfo();
    SplitConfig split;
    split.train_ratio = info.train_ratio;
    split.val_ratio = info.val_ratio;
    split.test_ratio = info.test_ratio;

    std::string json_str = SerializeConfig(info, split);

    std::ofstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("Cannot open file for writing: {}", filepath);
        return false;
    }

    file << json_str;
    file.close();

    spdlog::info("Exported dataset config '{}' to {}", name, filepath);
    return true;
}

bool DataRegistry::ExportConfig(const std::string& name, const std::string& filepath, const SplitConfig& split) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = datasets_.find(name);
    if (it == datasets_.end()) {
        spdlog::error("Cannot export config: dataset '{}' not found", name);
        return false;
    }

    DatasetInfo info = it->second->GetInfo();
    std::string json_str = SerializeConfig(info, split);

    std::ofstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("Cannot open file for writing: {}", filepath);
        return false;
    }

    file << json_str;
    file.close();

    spdlog::info("Exported dataset config '{}' to {} (custom split: {:.0f}/{:.0f}/{:.0f})",
                 name, filepath, split.train_ratio * 100, split.val_ratio * 100, split.test_ratio * 100);
    return true;
}

bool DataRegistry::ImportConfig(const std::string& filepath, std::string& out_name) {
    SplitConfig ignored_split;
    return ImportConfig(filepath, out_name, ignored_split);
}

bool DataRegistry::ImportConfig(const std::string& filepath, std::string& out_name, SplitConfig& out_split) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("Cannot open config file: {}", filepath);
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    DatasetInfo info;
    SplitConfig split;

    if (!DeserializeConfig(buffer.str(), info, split)) {
        return false;
    }

    // Load the dataset using the config
    if (info.path.empty()) {
        spdlog::error("Config file does not specify a dataset path");
        return false;
    }

    DatasetHandle handle;

    // Check if dataset with same name or path is already loaded
    bool already_loaded = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // First check by name
        if (!info.name.empty()) {
            auto it = datasets_.find(info.name);
            if (it != datasets_.end()) {
                handle = DatasetHandle(it->second, info.name);
                already_loaded = true;
                spdlog::info("Dataset '{}' already loaded, applying config only", info.name);
            }
        }

        // If not found by name, check by path
        if (!already_loaded) {
            for (const auto& [name, dataset] : datasets_) {
                DatasetInfo existing_info = dataset->GetInfo();
                if (existing_info.path == info.path) {
                    handle = DatasetHandle(dataset, name);
                    already_loaded = true;
                    spdlog::info("Dataset with path '{}' already loaded as '{}', applying config only",
                                 info.path, name);
                    break;
                }
            }
        }
    }

    // Only load if not already present
    if (!already_loaded) {
        // Report progress
        if (on_progress_) {
            on_progress_(0.0f, "Loading dataset from config...");
        }

        handle = LoadDataset(info.path, info.name);
        if (!handle.IsValid()) {
            spdlog::error("Failed to load dataset from path: {}", info.path);
            return false;
        }
    }

    // Apply split configuration
    handle.ApplySplit(split);

    out_name = handle.GetName();
    out_split = split;  // Return the split config from the file

    if (on_progress_) {
        on_progress_(1.0f, already_loaded ? "Config applied" : "Dataset loaded successfully");
    }

    spdlog::info("Imported dataset config from {}, {} '{}' (split: {:.0f}/{:.0f}/{:.0f})", filepath,
                 already_loaded ? "applied to existing" : "loaded as", out_name,
                 split.train_ratio * 100, split.val_ratio * 100, split.test_ratio * 100);
    return true;
}

// =============================================================================
// Dataset Versioning
// =============================================================================

std::vector<DataRegistry::DatasetVersion> DataRegistry::GetVersionHistory(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = version_history_.find(name);
    if (it != version_history_.end()) {
        return it->second;
    }
    return {};
}

bool DataRegistry::SaveVersion(const std::string& name, const std::string& description) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = datasets_.find(name);
    if (it == datasets_.end()) {
        spdlog::error("Cannot save version: dataset '{}' not found", name);
        return false;
    }

    DatasetInfo info = it->second->GetInfo();

    // Create version entry
    DatasetVersion version;

    // Generate version ID (simple incrementing)
    auto& history = version_history_[name];
    version.version_id = "v" + std::to_string(history.size() + 1);

    // Timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    char time_buf[32];
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", std::localtime(&time_t_now));
    version.timestamp = time_buf;

    version.description = description.empty() ? "Auto-saved version" : description;
    version.num_samples = info.num_samples;

    // Simple checksum based on sample count and memory usage
    std::stringstream ss;
    ss << info.num_samples << "_" << info.memory_usage << "_" << info.num_classes;
    version.checksum = ss.str();

    history.push_back(version);

    spdlog::info("Saved version {} for dataset '{}'", version.version_id, name);
    return true;
}

// =============================================================================
// Preprocessing Configuration Management
// =============================================================================

void DataRegistry::SetPreprocessingConfig(const std::string& dataset_id, const PreprocessingConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    preprocessing_configs_[dataset_id] = config;
    spdlog::info("DataRegistry: Set preprocessing config for dataset '{}'", dataset_id);
}

PreprocessingConfig DataRegistry::GetPreprocessingConfig(const std::string& dataset_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = preprocessing_configs_.find(dataset_id);
    if (it != preprocessing_configs_.end()) {
        return it->second;
    }
    // Return empty config if not found
    PreprocessingConfig empty_config;
    empty_config.dataset_id = dataset_id;
    return empty_config;
}

bool DataRegistry::HasPreprocessingConfig(const std::string& dataset_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return preprocessing_configs_.find(dataset_id) != preprocessing_configs_.end();
}

void DataRegistry::ClearPreprocessingConfig(const std::string& dataset_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    preprocessing_configs_.erase(dataset_id);
    spdlog::info("DataRegistry: Cleared preprocessing config for dataset '{}'", dataset_id);
}

// =============================================================================
// Augmentation Pipeline Management
// =============================================================================

void DataRegistry::SetAugmentationPipeline(const std::string& dataset_id,
                                            std::shared_ptr<transforms::Compose> pipeline) {
    std::lock_guard<std::mutex> lock(mutex_);
    augmentation_pipelines_[dataset_id] = pipeline;
    spdlog::info("DataRegistry: Set augmentation pipeline for dataset '{}'", dataset_id);
}

std::shared_ptr<transforms::Compose> DataRegistry::GetAugmentationPipeline(
    const std::string& dataset_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = augmentation_pipelines_.find(dataset_id);
    if (it != augmentation_pipelines_.end()) {
        return it->second;
    }
    return nullptr;
}

bool DataRegistry::HasAugmentationPipeline(const std::string& dataset_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return augmentation_pipelines_.find(dataset_id) != augmentation_pipelines_.end();
}

void DataRegistry::ClearAugmentationPipeline(const std::string& dataset_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    augmentation_pipelines_.erase(dataset_id);
    spdlog::info("DataRegistry: Cleared augmentation pipeline for dataset '{}'", dataset_id);
}

// =============================================================================
// Annotation Manager
// =============================================================================

AnnotationManager& DataRegistry::GetAnnotationManager() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!annotation_manager_) {
        annotation_manager_ = std::make_unique<AnnotationManager>();
    }
    return *annotation_manager_;
}

const AnnotationManager& DataRegistry::GetAnnotationManager() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!annotation_manager_) {
        annotation_manager_ = std::make_unique<AnnotationManager>();
    }
    return *annotation_manager_;
}

} // namespace cyxwiz
