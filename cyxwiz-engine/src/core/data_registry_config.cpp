// Data Registry Configuration Export/Import Implementation
// Extracted from data_registry.cpp to reduce file size

#include "data_registry.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <ctime>

namespace cyxwiz {

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

} // namespace cyxwiz
