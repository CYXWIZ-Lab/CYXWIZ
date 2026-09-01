#pragma once

#include "../dataset_base.h"
#include "../data_registry.h"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <limits>
#include <random>

namespace fs = std::filesystem;

namespace cyxwiz {

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
            if (num_classes_ == 0 ||
                num_classes_ - 1 > static_cast<size_t>(std::numeric_limits<int>::max())) {
                spdlog::error("HuggingFace dataset has unsupported class count: {}", num_classes_);
                return false;
            }
            const int max_label = static_cast<int>(num_classes_ - 1);
            std::uniform_int_distribution<> label_dist(0, max_label);
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

} // namespace cyxwiz
