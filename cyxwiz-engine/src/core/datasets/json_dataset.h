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

} // namespace cyxwiz
