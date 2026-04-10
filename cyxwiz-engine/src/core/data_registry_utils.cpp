// Data Registry Utility Functions Implementation
// Extracted from data_registry.cpp to reduce file size
// Contains: Versioning, Preprocessing, Augmentation, Annotation, Arrow integration

#include "data_registry.h"
#include "arrow_dataset.h"
#include "parquet_backed_dataset.h"
#include "annotation_manager.h"
#include "../preprocessing/preprocessing_config.h"
#include "../transforms/transform.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <ctime>
#include <sstream>

namespace cyxwiz {

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

// -----------------------------------------------------------------------------
// Parquet-backed dataset accessors (disk-backed lazy tabular data)
// -----------------------------------------------------------------------------

std::shared_ptr<ParquetBackedDataset> DataRegistry::GetParquetBackedDataset(
    const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = parquet_backed_datasets_.find(name);
    if (it == parquet_backed_datasets_.end()) {
        return nullptr;
    }
    return it->second;
}

bool DataRegistry::IsParquetBackedDataset(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return parquet_backed_datasets_.find(name) != parquet_backed_datasets_.end();
}

void DataRegistry::RegisterParquetBacked(
    const std::string& name,
    std::shared_ptr<ParquetBackedDataset> dataset) {
    if (!dataset) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (parquet_backed_datasets_.find(name) != parquet_backed_datasets_.end()) {
        spdlog::warn("RegisterParquetBacked: overwriting existing dataset '{}'", name);
    }
    parquet_backed_datasets_[name] = dataset;
    spdlog::info("Registered ParquetBackedDataset '{}': {} rows, {} columns, {:.1f} MB on disk",
                 name,
                 dataset->GetNumRows(),
                 dataset->GetNumColumns(),
                 dataset->GetFileSizeBytes() / (1024.0 * 1024.0));
}

} // namespace cyxwiz
