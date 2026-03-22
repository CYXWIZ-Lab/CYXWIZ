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

} // namespace cyxwiz
