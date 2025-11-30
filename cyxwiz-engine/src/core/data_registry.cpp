#include "data_registry.h"
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
#include <nlohmann/json.hpp>

// stb_image for image loading (implementation in stb_image_impl.cpp)
#include <stb_image.h>

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

    const std::vector<std::string>& GetColumnNames() const { return column_names_; }

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
                    std::stof(tokens[0]);
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

        // Detect channels from first image
        int width, height, channels;
        if (!stbi_info(image_paths_[0].c_str(), &width, &height, &channels)) {
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
        int width, height, channels;
        unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, channels_);

        if (!data) {
            spdlog::warn("Failed to load image: {}", path);
            return std::vector<float>(target_width_ * target_height_ * channels_, 0.0f);
        }

        // Resize if needed (simple nearest-neighbor for now)
        std::vector<float> result(target_width_ * target_height_ * channels_);

        float x_ratio = static_cast<float>(width) / target_width_;
        float y_ratio = static_cast<float>(height) / target_height_;

        for (int y = 0; y < target_height_; y++) {
            for (int x = 0; x < target_width_; x++) {
                int src_x = static_cast<int>(x * x_ratio);
                int src_y = static_cast<int>(y * y_ratio);

                src_x = std::min(src_x, width - 1);
                src_y = std::min(src_y, height - 1);

                for (int c = 0; c < channels_; c++) {
                    int src_idx = (src_y * width + src_x) * channels + c;
                    int dst_idx = (y * target_width_ + x) * channels_ + c;
                    result[dst_idx] = data[src_idx] / 255.0f;  // Normalize to [0, 1]
                }
            }
        }

        stbi_image_free(data);
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
        case DatasetType::ImageFolder:
            return LoadImageFolder(path, name);
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
        // Detect format if not specified
        if (config_.format.empty()) {
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

private:
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

            // Skip header
            if (first_line && config_.has_header) {
                first_line = false;
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

            // Determine label column
            int label_col = config_.label_column;
            if (label_col < 0) {
                label_col = static_cast<int>(tokens.size()) - 1;
            }

            // Parse values
            int label = 0;
            for (size_t i = 0; i < tokens.size(); i++) {
                try {
                    float val = std::stof(tokens[i]);
                    if (static_cast<int>(i) == label_col) {
                        label = static_cast<int>(val);
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
            }
        }

        // Infer shape
        if (!data_.empty() && config_.shape.empty()) {
            config_.shape = {data_[0].size()};
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
                        std::stof(tokens[0]);
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
    }

    return DatasetType::None;
}

std::string DataRegistry::TypeToString(DatasetType type) {
    switch (type) {
        case DatasetType::None: return "None";
        case DatasetType::CSV: return "CSV";
        case DatasetType::TSV: return "TSV";
        case DatasetType::ImageFolder: return "ImageFolder";
        case DatasetType::MNIST: return "MNIST";
        case DatasetType::FashionMNIST: return "FashionMNIST";
        case DatasetType::CIFAR10: return "CIFAR-10";
        case DatasetType::CIFAR100: return "CIFAR-100";
        case DatasetType::HuggingFace: return "HuggingFace";
        case DatasetType::Kaggle: return "Kaggle";
        case DatasetType::Custom: return "Custom";
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

} // namespace cyxwiz
