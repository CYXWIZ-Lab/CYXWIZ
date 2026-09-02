#include "sparse_feature_dataset_batcher.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

void SparseFeatureDatasetBatcher::RebuildBalancedIndices() {
    if (!balance_classes_ || balance_mode_ == "none" ||
        split_phase_ != BatcherPhase::Train || !is_training_ ||
        !dataset_->GetLabels() || base_indices_.empty()) {
        return;
    }

    std::map<int64_t, std::vector<int64_t>> by_label;
    for (int64_t row : base_indices_) {
        by_label[ReadLabel(row, true).class_index].push_back(row);
    }
    if (by_label.size() < 2) {
        spdlog::warn(
            "SparseFeatureDatasetBatcher: class balancing requested but "
            "the train split has fewer than two labels");
        return;
    }

    std::vector<size_t> class_counts;
    class_counts.reserve(by_label.size());
    for (const auto& entry : by_label) {
        class_counts.push_back(entry.second.size());
    }
    const auto minmax = std::minmax_element(
        class_counts.begin(), class_counts.end());
    size_t target_count = *minmax.second;
    if (balance_target_ == "min") {
        target_count = *minmax.first;
    } else if (balance_target_ == "median") {
        auto sorted = class_counts;
        std::sort(sorted.begin(), sorted.end());
        target_count = sorted[sorted.size() / 2];
    } else if (!balance_target_.empty() &&
               std::all_of(balance_target_.begin(), balance_target_.end(),
                   [](unsigned char value) {
                       return std::isdigit(value) != 0;
                   })) {
        try {
            target_count = std::max<size_t>(
                1, static_cast<size_t>(std::stoull(balance_target_)));
        } catch (...) {
            target_count = *minmax.second;
        }
    }

    balance_rng_.seed(balance_seed_ + balance_epoch_++);
    std::vector<int64_t> balanced;
    if (balance_mode_ == "undersample") {
        for (auto& entry : by_label) {
            auto rows = entry.second;
            std::shuffle(rows.begin(), rows.end(), balance_rng_);
            const size_t keep = std::min(target_count, rows.size());
            balanced.insert(
                balanced.end(), rows.begin(), rows.begin() + keep);
        }
    } else if (balance_mode_ == "weighted_sampler") {
        std::vector<int64_t> labels;
        labels.reserve(by_label.size());
        for (const auto& entry : by_label) {
            labels.push_back(entry.first);
        }
        std::uniform_int_distribution<size_t> class_distribution(
            0, labels.size() - 1);
        balanced.reserve(base_indices_.size());
        for (size_t sample = 0; sample < base_indices_.size(); ++sample) {
            auto& rows = by_label[labels[class_distribution(balance_rng_)]];
            std::uniform_int_distribution<size_t> row_distribution(
                0, rows.size() - 1);
            balanced.push_back(rows[row_distribution(balance_rng_)]);
        }
    } else {
        if (balance_mode_ != "oversample") {
            spdlog::warn(
                "SparseFeatureDatasetBatcher: unsupported balance_mode='{}'; "
                "using oversample",
                balance_mode_);
        }
        for (auto& entry : by_label) {
            auto rows = entry.second;
            std::shuffle(rows.begin(), rows.end(), balance_rng_);
            for (size_t sample = 0; sample < target_count; ++sample) {
                balanced.push_back(rows[sample % rows.size()]);
            }
        }
    }
    if (!balanced.empty()) {
        indices_ = std::move(balanced);
    }
}

} // namespace cyxwiz
