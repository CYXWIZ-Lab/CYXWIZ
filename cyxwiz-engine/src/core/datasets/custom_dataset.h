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
#include <random>
#include <map>

namespace fs = std::filesystem;

namespace cyxwiz {

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

} // namespace cyxwiz
