#include "data_registry.h"
#include "annotation_manager.h"
#include "../preprocessing/preprocessing_config.h"

#include <spdlog/spdlog.h>

namespace cyxwiz {

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

    int suffix = 1;
    while (datasets_.find(name + "_" + std::to_string(suffix)) != datasets_.end()) {
        suffix++;
    }
    return name + "_" + std::to_string(suffix);
}

void DataRegistry::ForgetTabularSourcePathUnlocked(const std::string& name) {
    auto it = tabular_source_paths_by_name_.find(name);
    if (it == tabular_source_paths_by_name_.end()) return;
    tabular_dataset_by_source_path_.erase(it->second);
    tabular_source_paths_by_name_.erase(it);
}

bool DataRegistry::IsSparseFeatureDataset(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sparse_feature_datasets_.find(name) !=
        sparse_feature_datasets_.end();
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
    names.reserve(datasets_.size());
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

} // namespace cyxwiz
