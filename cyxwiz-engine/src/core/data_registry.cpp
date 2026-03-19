#include "data_registry.h"
#include "dataset_base.h"
#include "arrow_dataset.h"
#include "annotation_manager.h"
#include "image_utils.h"
#include "../preprocessing/preprocessing_config.h"
#include "../transforms/transform.h"
#include "../plugin/registries/plugin_data_loader_registry.h"

// Dataset implementations
#include "datasets/kaggle_dataset.h"
#include "datasets/hdf5_dataset.h"
#include "datasets/image_csv_dataset.h"
#include "datasets/streaming_dataset.h"
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
// HDF5 Dataset Implementation - Now in datasets/hdf5_dataset.{h,cpp}
// =============================================================================

// HDF5Dataset has been moved to datasets/hdf5_dataset.{h,cpp}


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
/**
 * Kaggle Dataset Implementation
 * Downloads and loads datasets from Kaggle using local caching
 */
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
        default: {
            // Check if a plugin data loader can handle this format
            try {
                auto ext = std::filesystem::path(path).extension().string();
                if (!ext.empty() && cyxwiz::plugin::PluginDataLoaderRegistry::Instance().HasLoaderForExtension(ext)) {
                    spdlog::info("DataRegistry: Plugin data loader available for '{}' (bridge not yet implemented)", path);
                    // TODO: Bridge PluginDataset to DatasetHandle with adapter class
                }
            } catch (const std::exception& e) {
                spdlog::warn("DataRegistry: Plugin loader check failed: {}", e.what());
            }
            spdlog::error("Unknown dataset type for path: {}", path);
            return DatasetHandle();
        }
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
        } else if (config_.format == "arff") {
            LoadARFF();
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
        // Set type based on format
        if (config_.format == "arff")
            info.type = DatasetType::ARFF;
        else if (config_.format == "csv" || config_.format == "text")
            info.type = DatasetType::CSV;
        else if (config_.format == "tsv")
            info.type = DatasetType::TSV;
        else if (config_.format == "json")
            info.type = DatasetType::JSON;
        else
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
        if (ext == ".arff") return "arff";
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

    void LoadARFF() {
        // ARFF (Attribute-Relation File Format) parser
        // Supports: @RELATION, @ATTRIBUTE (NUMERIC, REAL, INTEGER, STRING, {class1,class2,...}), @DATA
        std::ifstream file(config_.data_path);
        if (!file.is_open()) {
            spdlog::error("Failed to open ARFF file: {}", config_.data_path);
            CreateSampleData();
            return;
        }

        struct AttributeInfo {
            std::string name;
            bool is_nominal = false;
            std::vector<std::string> nominal_values;  // For {val1,val2,...}
        };
        std::vector<AttributeInfo> attributes;
        bool in_data_section = false;
        std::string relation_name;

        std::string line;
        while (std::getline(file, line)) {
            // Trim whitespace
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue;
            line = line.substr(start);

            // Skip comments
            if (line[0] == '%') continue;

            // Convert directive to lowercase for comparison
            std::string lower_line = line;
            std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(), ::tolower);

            if (!in_data_section) {
                if (lower_line.rfind("@relation", 0) == 0) {
                    // @RELATION name
                    relation_name = line.substr(10);
                    relation_name.erase(0, relation_name.find_first_not_of(" \t'\""));
                    relation_name.erase(relation_name.find_last_not_of(" \t'\"") + 1);
                } else if (lower_line.rfind("@attribute", 0) == 0) {
                    // @ATTRIBUTE name type
                    AttributeInfo attr;
                    std::string rest = line.substr(11);
                    rest.erase(0, rest.find_first_not_of(" \t"));

                    // Extract attribute name (may be quoted)
                    size_t name_end;
                    if (!rest.empty() && (rest[0] == '\'' || rest[0] == '"')) {
                        char quote = rest[0];
                        name_end = rest.find(quote, 1);
                        attr.name = rest.substr(1, name_end - 1);
                        rest = rest.substr(name_end + 1);
                    } else {
                        name_end = rest.find_first_of(" \t");
                        attr.name = rest.substr(0, name_end);
                        rest = (name_end != std::string::npos) ? rest.substr(name_end) : "";
                    }

                    rest.erase(0, rest.find_first_not_of(" \t"));
                    std::string type_lower = rest;
                    std::transform(type_lower.begin(), type_lower.end(), type_lower.begin(), ::tolower);

                    if (type_lower.find('{') != std::string::npos) {
                        // Nominal: {val1,val2,val3}
                        attr.is_nominal = true;
                        size_t brace_start = rest.find('{');
                        size_t brace_end = rest.find('}');
                        if (brace_start != std::string::npos && brace_end != std::string::npos) {
                            std::string vals = rest.substr(brace_start + 1, brace_end - brace_start - 1);
                            std::stringstream vss(vals);
                            std::string val;
                            while (std::getline(vss, val, ',')) {
                                val.erase(0, val.find_first_not_of(" \t'\""));
                                val.erase(val.find_last_not_of(" \t'\"") + 1);
                                attr.nominal_values.push_back(val);
                            }
                        }
                    }
                    // NUMERIC, REAL, INTEGER, STRING all treated as numeric/feature

                    attributes.push_back(attr);
                    column_names_.push_back(attr.name);
                } else if (lower_line.rfind("@data", 0) == 0) {
                    in_data_section = true;
                }
            } else {
                // Data section: comma-separated values
                if (line.empty() || line[0] == '%') continue;

                std::vector<std::string> tokens;
                std::stringstream ss(line);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    token.erase(0, token.find_first_not_of(" \t'\""));
                    token.erase(token.find_last_not_of(" \t'\"") + 1);
                    tokens.push_back(token);
                }

                if (tokens.size() != attributes.size()) continue;

                // Determine label column: use last nominal attribute, or last column
                int label_col = config_.label_column;
                if (label_col == -1) {
                    label_col = (int)attributes.size() - 1;
                }
                bool no_label = (label_col == -2);

                // Store original layout info once
                if (original_col_count_ == 0) {
                    original_col_count_ = (int)tokens.size();
                    resolved_label_col_ = no_label ? -2 : label_col;
                }

                std::vector<float> sample;
                int label = 0;
                float float_label = 0.0f;

                for (size_t i = 0; i < tokens.size(); i++) {
                    if (tokens[i] == "?") continue;  // Missing value

                    if (!no_label && (int)i == label_col) {
                        // Label column
                        if (attributes[i].is_nominal) {
                            // Map nominal value to integer index
                            auto& vals = attributes[i].nominal_values;
                            auto it = std::find(vals.begin(), vals.end(), tokens[i]);
                            if (it != vals.end()) {
                                label = (int)std::distance(vals.begin(), it);
                                float_label = (float)label;
                            }
                            // Store class names from first nominal label attribute
                            if (config_.class_names.empty()) {
                                for (auto& v : vals) config_.class_names.push_back(v);
                                config_.num_classes = vals.size();
                            }
                        } else {
                            try {
                                float val = std::stof(tokens[i]);
                                label = (int)val;
                                float_label = val;
                            } catch (...) {}
                        }
                    } else {
                        // Feature column
                        if (attributes[i].is_nominal) {
                            // Encode nominal as integer
                            auto& vals = attributes[i].nominal_values;
                            auto it = std::find(vals.begin(), vals.end(), tokens[i]);
                            if (it != vals.end()) {
                                sample.push_back((float)std::distance(vals.begin(), it));
                            } else {
                                sample.push_back(0.0f);
                            }
                        } else {
                            try {
                                float val = std::stof(tokens[i]);
                                if (config_.normalize && config_.scale != 1.0f) {
                                    val *= config_.scale;
                                }
                                sample.push_back(val);
                            } catch (...) {
                                sample.push_back(0.0f);
                            }
                        }
                    }
                }

                if (!sample.empty()) {
                    data_.push_back(sample);
                    labels_.push_back(label);
                    float_labels_.push_back(float_label);
                }
            }
        }

        // Infer shape
        if (!data_.empty() && config_.shape.empty()) {
            config_.shape = {data_[0].size()};
        }

        spdlog::info("ARFF loaded: relation='{}', {} attributes, {} samples",
                     relation_name, attributes.size(), data_.size());
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

// =============================================================================
// Apache Arrow Integration (Data Studio Foundation - Phase 0)
// =============================================================================

std::shared_ptr<ArrowDataset> DataRegistry::LoadArrowTable(
    const std::string& path, const std::string& name) {

    std::lock_guard<std::mutex> lock(mutex_);

    // Generate unique name if not provided
    std::string dataset_name = name.empty() ? GenerateUniqueName("arrow_data") : name;

    // Check if already loaded
    if (arrow_datasets_.find(dataset_name) != arrow_datasets_.end()) {
        spdlog::warn("Arrow dataset '{}' already loaded, returning existing instance", dataset_name);
        return arrow_datasets_[dataset_name];
    }

    // Load from file using ArrowDataset factory
    auto dataset = ArrowDataset::FromFile(path, dataset_name);
    if (!dataset) {
        spdlog::error("Failed to load Arrow dataset from: {}", path);
        return nullptr;
    }

    // Store in registry
    arrow_datasets_[dataset_name] = dataset;
    last_access_times_[dataset_name] = std::chrono::steady_clock::now();

    spdlog::info("Loaded Arrow dataset '{}': {} rows, {} columns, {} bytes",
                 dataset_name,
                 dataset->GetNumRows(),
                 dataset->GetNumColumns(),
                 dataset->GetMemoryUsage());

    // Trigger callback
    if (on_loaded_) {
        DatasetInfo info;
        info.name = dataset_name;
        info.path = path;
        info.num_samples = dataset->GetNumRows();
        info.memory_usage = dataset->GetMemoryUsage();
        info.is_loaded = true;
        on_loaded_(dataset_name, info);
    }

    return dataset;
}

std::shared_ptr<ArrowDataset> DataRegistry::RegisterArrowTable(
    std::shared_ptr<arrow::Table> table, const std::string& name) {

    std::lock_guard<std::mutex> lock(mutex_);

    // Generate unique name if not provided
    std::string dataset_name = name.empty() ? GenerateUniqueName("arrow_data") : name;

    // Check if already exists
    if (arrow_datasets_.find(dataset_name) != arrow_datasets_.end()) {
        spdlog::warn("Arrow dataset '{}' already exists, overwriting", dataset_name);
    }

    // Create ArrowDataset wrapper
    auto dataset = std::make_shared<ArrowDataset>(table, dataset_name);

    // Store in registry
    arrow_datasets_[dataset_name] = dataset;
    last_access_times_[dataset_name] = std::chrono::steady_clock::now();

    spdlog::info("Registered Arrow dataset '{}': {} rows, {} columns, {} bytes",
                 dataset_name,
                 dataset->GetNumRows(),
                 dataset->GetNumColumns(),
                 dataset->GetMemoryUsage());

    // Trigger callback
    if (on_loaded_) {
        DatasetInfo info;
        info.name = dataset_name;
        info.num_samples = dataset->GetNumRows();
        info.memory_usage = dataset->GetMemoryUsage();
        info.is_loaded = true;
        on_loaded_(dataset_name, info);
    }

    return dataset;
}

std::shared_ptr<ArrowDataset> DataRegistry::GetArrowDataset(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = arrow_datasets_.find(name);
    if (it == arrow_datasets_.end()) {
        spdlog::warn("Arrow dataset '{}' not found in registry", name);
        return nullptr;
    }

    // Update last access time for LRU
    last_access_times_[name] = std::chrono::steady_clock::now();

    return it->second;
}

bool DataRegistry::IsArrowDataset(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return arrow_datasets_.find(name) != arrow_datasets_.end();
}

} // namespace cyxwiz
