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
#include <filesystem>
#include <sstream>

namespace cyxwiz {

namespace {

std::string NormalizeTabularSourcePath(const std::string& path) {
    if (path.empty()) return {};
    try {
        return std::filesystem::weakly_canonical(std::filesystem::absolute(path))
            .lexically_normal()
            .string();
    } catch (...) {
        try {
            return std::filesystem::absolute(path).lexically_normal().string();
        } catch (...) {
            return path;
        }
    }
}

} // namespace


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
    parquet_backed_datasets_.erase(dataset_name);
    sparse_feature_datasets_.erase(dataset_name);
    ForgetTabularSourcePathUnlocked(dataset_name);
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
    parquet_backed_datasets_.erase(dataset_name);
    sparse_feature_datasets_.erase(dataset_name);
    ForgetTabularSourcePathUnlocked(dataset_name);
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
        // Lookup-miss is a normal probe pattern (Properties panel and the
        // compile gate both call this every frame). Caller is expected to
        // handle the null and either fall through or surface its own
        // user-facing error. Logging warn here was producing thousands of
        // entries per second when no dataset was loaded.
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

std::optional<std::string> DataRegistry::FindTabularDatasetBySourcePath(
    const std::string& path) const {
    const std::string normalized = NormalizeTabularSourcePath(path);
    if (normalized.empty()) return std::nullopt;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tabular_dataset_by_source_path_.find(normalized);
    if (it == tabular_dataset_by_source_path_.end()) {
        return std::nullopt;
    }
    const std::string& name = it->second;
    if (arrow_datasets_.find(name) == arrow_datasets_.end() &&
        parquet_backed_datasets_.find(name) == parquet_backed_datasets_.end()) {
        return std::nullopt;
    }
    return name;
}

std::optional<std::string> DataRegistry::GetTabularSourcePath(
    const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tabular_source_paths_by_name_.find(name);
    if (it == tabular_source_paths_by_name_.end()) {
        return std::nullopt;
    }
    return it->second;
}

void DataRegistry::RememberTabularSourcePathUnlocked(
    const std::string& name, const std::string& path) {
    const std::string normalized = NormalizeTabularSourcePath(path);
    if (name.empty() || normalized.empty()) return;

    if (auto old_name = tabular_dataset_by_source_path_.find(normalized);
        old_name != tabular_dataset_by_source_path_.end() && old_name->second != name) {
        tabular_source_paths_by_name_.erase(old_name->second);
    }

    if (auto old_path = tabular_source_paths_by_name_.find(name);
        old_path != tabular_source_paths_by_name_.end() && old_path->second != normalized) {
        tabular_dataset_by_source_path_.erase(old_path->second);
    }

    tabular_source_paths_by_name_[name] = normalized;
    tabular_dataset_by_source_path_[normalized] = name;
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
    arrow_datasets_.erase(name);
    sparse_feature_datasets_.erase(name);
    RememberTabularSourcePathUnlocked(name, dataset->GetFilePath());
    spdlog::info("Registered ParquetBackedDataset '{}': {} rows, {} columns, {:.1f} MB on disk",
                 name,
                 dataset->GetNumRows(),
                 dataset->GetNumColumns(),
                 dataset->GetFileSizeBytes() / (1024.0 * 1024.0));
}

void DataRegistry::UnregisterTabularDataset(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    bool removed_arrow = false;
    bool removed_parquet = false;
    bool removed_sparse = false;

    auto arrow_it = arrow_datasets_.find(name);
    if (arrow_it != arrow_datasets_.end()) {
        arrow_datasets_.erase(arrow_it);
        removed_arrow = true;
    }

    auto parquet_it = parquet_backed_datasets_.find(name);
    if (parquet_it != parquet_backed_datasets_.end()) {
        parquet_backed_datasets_.erase(parquet_it);
        removed_parquet = true;
    }

    auto sparse_it = sparse_feature_datasets_.find(name);
    if (sparse_it != sparse_feature_datasets_.end()) {
        sparse_feature_datasets_.erase(sparse_it);
        removed_sparse = true;
    }

    ForgetTabularSourcePathUnlocked(name);
    if (removed_arrow || removed_parquet || removed_sparse) {
        spdlog::debug(
            "UnregisterTabularDataset '{}': removed arrow={} parquet={} sparse={}",
            name, removed_arrow, removed_parquet, removed_sparse);
    }

    // Cascade: also drop the PipelineMaterializer-produced "__materialized"
    // variant of this dataset, if one exists. Without this, re-loading a CSV
    // under a different name leaks the previous run's materialized variant
    // into the registry until project close. The cascade is inlined under
    // the same lock to avoid recursive lock acquisition. The suffix matches
    // PipelineMaterializer::kMaterializedSuffix; keep them in sync.
    static const std::string kMaterializedSuffix = "__materialized";
    const bool already_materialized =
        name.size() >= kMaterializedSuffix.size() &&
        name.compare(name.size() - kMaterializedSuffix.size(),
                     kMaterializedSuffix.size(),
                     kMaterializedSuffix) == 0;
    if (!already_materialized) {
        const std::string mat_name = name + kMaterializedSuffix;
        bool mat_arrow = false;
        bool mat_parquet = false;
        bool mat_sparse = false;
        if (auto it = arrow_datasets_.find(mat_name); it != arrow_datasets_.end()) {
            arrow_datasets_.erase(it);
            mat_arrow = true;
        }
        if (auto it = parquet_backed_datasets_.find(mat_name);
            it != parquet_backed_datasets_.end()) {
            parquet_backed_datasets_.erase(it);
            mat_parquet = true;
        }
        if (auto it = sparse_feature_datasets_.find(mat_name);
            it != sparse_feature_datasets_.end()) {
            sparse_feature_datasets_.erase(it);
            mat_sparse = true;
        }
        ForgetTabularSourcePathUnlocked(mat_name);
        if (mat_arrow || mat_parquet || mat_sparse) {
            spdlog::debug("UnregisterTabularDataset '{}' cascade: dropped materialized variant '{}'",
                          name, mat_name);
        }
    }
}

// --- Image dataset registry (Phase 1) ---

void DataRegistry::RegisterImageDataset(const std::string& name,
                                          const ImageDatasetEntry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    image_dataset_entries_[name] = entry;
    spdlog::info("RegisterImageDataset '{}': {} images, {} classes, layout={}",
                 name, entry.num_images, entry.num_classes, entry.layout);
}

bool DataRegistry::IsImageDataset(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return image_dataset_entries_.find(name) != image_dataset_entries_.end();
}

const DataRegistry::ImageDatasetEntry*
DataRegistry::GetImageDatasetEntry(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = image_dataset_entries_.find(name);
    if (it == image_dataset_entries_.end()) return nullptr;
    return &it->second;
}

void DataRegistry::UnregisterImageDataset(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    bool removed_any = false;

    auto it = image_dataset_entries_.find(name);
    if (it != image_dataset_entries_.end()) {
        image_dataset_entries_.erase(it);
        removed_any = true;
    }

    // Image Apply currently probes via LoadImageFolder / LoadImageCSV,
    // which also parks a concrete Dataset instance in datasets_ under
    // the same name. If we only drop the lightweight image sidecar,
    // GenerateUniqueName() still sees the old datasets_ entry and the
    // next re-Apply on the same node drifts to "<name>_1", leaving the
    // node's persisted dataset_name out of sync with the registry.
    auto ds_it = datasets_.find(name);
    if (ds_it != datasets_.end()) {
        datasets_.erase(ds_it);
        removed_any = true;
    }

    if (removed_any) {
        spdlog::debug("UnregisterImageDataset '{}'", name);
    }
}

// --- Audio dataset registry (Phase 2) ---

void DataRegistry::RegisterAudioDataset(const std::string& name,
                                          const AudioDatasetEntry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    audio_dataset_entries_[name] = entry;
    spdlog::info("RegisterAudioDataset '{}': {} samples, {} classes, feature_type={}, sr={}",
                 name, entry.num_samples, entry.num_classes,
                 entry.feature_type, entry.target_sr);
}

bool DataRegistry::IsAudioDataset(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return audio_dataset_entries_.find(name) != audio_dataset_entries_.end();
}

const DataRegistry::AudioDatasetEntry*
DataRegistry::GetAudioDatasetEntry(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = audio_dataset_entries_.find(name);
    if (it == audio_dataset_entries_.end()) return nullptr;
    return &it->second;
}

void DataRegistry::UnregisterAudioDataset(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = audio_dataset_entries_.find(name);
    if (it != audio_dataset_entries_.end()) {
        audio_dataset_entries_.erase(it);
        spdlog::debug("UnregisterAudioDataset '{}'", name);
    }
}

// --- Text dataset registry (Phase 3) ---

void DataRegistry::RegisterTextDataset(const std::string& name,
                                         const TextDatasetEntry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    text_dataset_entries_[name] = entry;
    spdlog::info("RegisterTextDataset '{}': {} samples, {} classes, "
                 "vocab_size={}, tokenizer_type={}, max_length={}",
                 name, entry.num_samples, entry.num_classes,
                 entry.vocab_size, entry.tokenizer_type, entry.max_length);
}

bool DataRegistry::IsTextDataset(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return text_dataset_entries_.find(name) != text_dataset_entries_.end();
}

const DataRegistry::TextDatasetEntry*
DataRegistry::GetTextDatasetEntry(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = text_dataset_entries_.find(name);
    if (it == text_dataset_entries_.end()) return nullptr;
    return &it->second;
}

void DataRegistry::UnregisterTextDataset(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = text_dataset_entries_.find(name);
    if (it != text_dataset_entries_.end()) {
        text_dataset_entries_.erase(it);
        spdlog::debug("UnregisterTextDataset '{}'", name);
    }
}

void DataRegistry::RestoreTabularDataset(
    const std::string& name,
    std::shared_ptr<ArrowDataset> arrow_dataset,
    std::shared_ptr<ParquetBackedDataset> parquet_dataset,
    const std::string& source_path) {
    if (name.empty() || (!arrow_dataset && !parquet_dataset)) return;

    std::lock_guard<std::mutex> lock(mutex_);
    arrow_datasets_.erase(name);
    parquet_backed_datasets_.erase(name);
    sparse_feature_datasets_.erase(name);
    if (arrow_dataset) {
        arrow_datasets_[name] = std::move(arrow_dataset);
    } else {
        parquet_backed_datasets_[name] = std::move(parquet_dataset);
    }
    RememberTabularSourcePathUnlocked(name, source_path);
    spdlog::info(
        "RestoreTabularDataset: restored prior valid registration '{}'",
        name);
}

void DataRegistry::ClearAllTabularDatasets() {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t arrow_count = arrow_datasets_.size();
    size_t parquet_count = parquet_backed_datasets_.size();
    size_t sparse_count = sparse_feature_datasets_.size();
    size_t image_count = image_dataset_entries_.size();
    size_t audio_count = audio_dataset_entries_.size();
    size_t text_count = text_dataset_entries_.size();

    arrow_datasets_.clear();
    parquet_backed_datasets_.clear();
    sparse_feature_datasets_.clear();
    tabular_source_paths_by_name_.clear();
    tabular_dataset_by_source_path_.clear();
    image_dataset_entries_.clear();
    audio_dataset_entries_.clear();
    text_dataset_entries_.clear();

    if (arrow_count > 0 || parquet_count > 0 || sparse_count > 0 ||
        image_count > 0 ||
        audio_count > 0 || text_count > 0) {
        spdlog::info("ClearAllDatasets: removed {} Arrow + {} Parquet + "
                     "{} Sparse + {} Image + {} Audio + {} Text entries",
                     arrow_count, parquet_count, sparse_count, image_count,
                     audio_count, text_count);
    }
}

} // namespace cyxwiz
