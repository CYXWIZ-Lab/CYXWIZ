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

} // namespace cyxwiz
