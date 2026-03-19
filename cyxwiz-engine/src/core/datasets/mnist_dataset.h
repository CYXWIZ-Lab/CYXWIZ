#pragma once

#include "../dataset_base.h"
#include "../data_registry.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <set>

namespace fs = std::filesystem;

namespace cyxwiz {

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

} // namespace cyxwiz
