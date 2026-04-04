// Dataset Base Class Implementation
// Extracted from data_registry.cpp to reduce file size

#include "dataset_base.h"
#include "data_registry.h"  // For DatasetHandle
#include <spdlog/spdlog.h>
#include <algorithm>
#include <random>

namespace cyxwiz {

// =============================================================================
// Dataset Base Class Implementation
// =============================================================================

std::pair<std::vector<std::vector<float>>, std::vector<int>>
Dataset::GetBatch(const std::vector<size_t>& indices) const {
    std::vector<std::vector<float>> samples;
    std::vector<int> labels;
    samples.reserve(indices.size());
    labels.reserve(indices.size());

    for (size_t idx : indices) {
        auto [sample, label] = GetItem(idx);
        samples.push_back(std::move(sample));
        labels.push_back(label);
    }

    return {std::move(samples), std::move(labels)};
}

void Dataset::SetSplit(const SplitConfig& config) {
    split_config_ = config;

    // Create all indices
    all_indices_.resize(Size());
    for (size_t i = 0; i < Size(); i++) {
        all_indices_[i] = i;
    }

    // Shuffle if requested
    if (config.shuffle) {
        std::mt19937 rng(config.seed);
        std::shuffle(all_indices_.begin(), all_indices_.end(), rng);
    }

    // Calculate split sizes
    size_t total = Size();
    size_t train_size = static_cast<size_t>(total * config.train_ratio);
    size_t val_size = static_cast<size_t>(total * config.val_ratio);
    size_t test_size = total - train_size - val_size;

    // Assign indices to splits
    train_indices_.clear();
    val_indices_.clear();
    test_indices_.clear();

    train_indices_.reserve(train_size);
    val_indices_.reserve(val_size);
    test_indices_.reserve(test_size);

    for (size_t i = 0; i < train_size; i++) {
        train_indices_.push_back(all_indices_[i]);
    }
    for (size_t i = train_size; i < train_size + val_size; i++) {
        val_indices_.push_back(all_indices_[i]);
    }
    for (size_t i = train_size + val_size; i < total; i++) {
        test_indices_.push_back(all_indices_[i]);
    }

    spdlog::info("Dataset split: train={}, val={}, test={}",
        train_indices_.size(), val_indices_.size(), test_indices_.size());
}

const std::vector<size_t>& Dataset::GetSplitIndices(DatasetSplit split) const {
    switch (split) {
        case DatasetSplit::Train: return train_indices_;
        case DatasetSplit::Validation: return val_indices_;
        case DatasetSplit::Test: return test_indices_;
        case DatasetSplit::All: return all_indices_;
        default: return all_indices_;
    }
}

// =============================================================================
// DatasetHandle Implementation
// =============================================================================

DatasetHandle::DatasetHandle(std::shared_ptr<Dataset> dataset, const std::string& name)
    : dataset_(std::move(dataset)), name_(name) {}

DatasetInfo DatasetHandle::GetInfo() const {
    if (!IsValid()) return DatasetInfo{};
    return dataset_->GetInfo();
}

size_t DatasetHandle::Size() const {
    if (!IsValid()) return 0;
    return dataset_->Size();
}

size_t DatasetHandle::Size(DatasetSplit split) const {
    if (!IsValid()) return 0;
    return dataset_->GetSplitIndices(split).size();
}

std::pair<std::vector<float>, int> DatasetHandle::GetSample(size_t index) const {
    if (!IsValid()) return {{}, -1};
    return dataset_->GetItem(index);
}

std::pair<std::vector<std::vector<float>>, std::vector<int>>
DatasetHandle::GetBatch(const std::vector<size_t>& indices) const {
    if (!IsValid()) return {{}, {}};
    return dataset_->GetBatch(indices);
}

const std::vector<size_t>& DatasetHandle::GetTrainIndices() const {
    static std::vector<size_t> empty;
    if (!IsValid()) return empty;
    return dataset_->GetTrainIndices();
}

const std::vector<size_t>& DatasetHandle::GetValIndices() const {
    static std::vector<size_t> empty;
    if (!IsValid()) return empty;
    return dataset_->GetValIndices();
}

const std::vector<size_t>& DatasetHandle::GetTestIndices() const {
    static std::vector<size_t> empty;
    if (!IsValid()) return empty;
    return dataset_->GetTestIndices();
}

const std::vector<size_t>& DatasetHandle::GetSplitIndices(DatasetSplit split) const {
    static std::vector<size_t> empty;
    if (!IsValid()) return empty;
    return dataset_->GetSplitIndices(split);
}

void DatasetHandle::ApplySplit(const SplitConfig& config) {
    if (IsValid()) {
        dataset_->SetSplit(config);
    }
}

} // namespace cyxwiz
