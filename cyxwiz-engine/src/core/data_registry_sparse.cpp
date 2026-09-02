#include "data_registry.h"
#include "sparse_feature_dataset.h"

#include <arrow/chunked_array.h>

#include <spdlog/spdlog.h>

namespace cyxwiz {
namespace {

constexpr const char* kMaterializedSuffix = "__materialized";

bool HasMaterializedSuffix(const std::string& name) {
    const std::string suffix = kMaterializedSuffix;
    return name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

SparseFeatureDatasetInfo MakeInfo(const SparseFeatureDataset& dataset) {
    SparseFeatureDatasetInfo info;
    info.name = dataset.GetName();
    info.num_rows = dataset.GetNumRows();
    info.num_features = dataset.GetNumFeatures();
    info.nnz = dataset.GetNnz();
    info.density = dataset.GetDensity();
    info.feature_storage_bytes = dataset.GetFeatureStorageBytes();
    info.label_storage_bytes = dataset.GetLabelStorageBytes();
    info.estimated_host_memory_bytes =
        dataset.GetEstimatedHostMemoryBytes();
    info.dense_feature_bytes = dataset.GetDenseFeatureBytes();
    info.feature_name_count = dataset.GetFeatureNames().size();
    info.has_labels = static_cast<bool>(dataset.GetLabels());
    info.label_name = dataset.GetLabelName();
    if (dataset.GetLabels()) {
        info.label_type = dataset.GetLabels()->type()->ToString();
        info.label_null_count = dataset.GetLabels()->null_count();
    }
    return info;
}

} // namespace

bool DataRegistry::RegisterSparseFeatureDataset(
    std::shared_ptr<const SparseFeatureDataset> dataset) {
    if (!dataset || dataset->GetName().empty()) {
        return false;
    }

    const std::string name = dataset->GetName();
    std::lock_guard<std::mutex> lock(mutex_);
    const bool replacing =
        sparse_feature_datasets_.find(name) != sparse_feature_datasets_.end();

    // A dataset name identifies exactly one tabular representation. Publish
    // the already-validated immutable CSR object while holding the same lock
    // used to retire any prior dense representation.
    sparse_feature_datasets_[name] = std::move(dataset);
    arrow_datasets_.erase(name);
    parquet_backed_datasets_.erase(name);
    ForgetTabularSourcePathUnlocked(name);

    spdlog::info(
        "{} sparse feature dataset '{}': rows={} features={} nnz={}",
        replacing ? "Replaced" : "Registered", name,
        sparse_feature_datasets_[name]->GetNumRows(),
        sparse_feature_datasets_[name]->GetNumFeatures(),
        sparse_feature_datasets_[name]->GetNnz());
    return true;
}

std::shared_ptr<const SparseFeatureDataset>
DataRegistry::GetSparseFeatureDataset(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sparse_feature_datasets_.find(name);
    return it == sparse_feature_datasets_.end() ? nullptr : it->second;
}

std::optional<SparseFeatureDatasetInfo>
DataRegistry::InspectSparseFeatureDataset(const std::string& name) const {
    std::shared_ptr<const SparseFeatureDataset> dataset;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = sparse_feature_datasets_.find(name);
        if (it == sparse_feature_datasets_.end()) {
            return std::nullopt;
        }
        dataset = it->second;
    }
    return MakeInfo(*dataset);
}

std::vector<SparseFeatureDatasetInfo>
DataRegistry::ListSparseFeatureDatasets() const {
    std::vector<std::shared_ptr<const SparseFeatureDataset>> datasets;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        datasets.reserve(sparse_feature_datasets_.size());
        for (const auto& [_, dataset] : sparse_feature_datasets_) {
            datasets.push_back(dataset);
        }
    }

    std::vector<SparseFeatureDatasetInfo> result;
    result.reserve(datasets.size());
    for (const auto& dataset : datasets) {
        result.push_back(MakeInfo(*dataset));
    }
    return result;
}

bool DataRegistry::UnregisterSparseFeatureDataset(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    bool removed = sparse_feature_datasets_.erase(name) > 0;

    if (!HasMaterializedSuffix(name)) {
        removed = sparse_feature_datasets_.erase(
                      name + kMaterializedSuffix) > 0 || removed;
    }
    if (removed) {
        spdlog::debug("Unregistered sparse feature dataset '{}'", name);
    }
    return removed;
}

size_t DataRegistry::ClearAllSparseFeatureDatasets() {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t count = sparse_feature_datasets_.size();
    sparse_feature_datasets_.clear();
    if (count > 0) {
        spdlog::info("Cleared {} sparse feature datasets", count);
    }
    return count;
}

} // namespace cyxwiz
