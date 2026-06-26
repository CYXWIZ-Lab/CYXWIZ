#pragma once

#include "metric_learning_batcher.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

struct PairDatasetRow {
    std::vector<float> input_a;
    std::vector<float> input_b;
    float label = 0.0f;
    int64_t sample_id_a = 0;
    int64_t sample_id_b = 0;
    int64_t class_id_a = 0;
    int64_t class_id_b = 0;
    bool has_label = true;
    bool has_sample_ids = false;
    bool has_class_ids = false;
};

struct TripletDatasetRow {
    std::vector<float> anchor;
    std::vector<float> positive;
    std::vector<float> negative;
    int64_t anchor_sample_id = 0;
    int64_t positive_sample_id = 0;
    int64_t negative_sample_id = 0;
    int64_t anchor_class_id = 0;
    int64_t positive_class_id = 0;
    int64_t negative_class_id = 0;
    bool has_sample_ids = false;
    bool has_class_ids = false;
};

struct PairDatasetBuilderConfig {
    MetricLearningBatcherConfig batcher;
    bool require_labels = true;
    bool derive_labels_from_class_ids = false;
};

struct TripletDatasetBuilderConfig {
    MetricLearningBatcherConfig batcher;
    bool validate_class_ids = true;
};

struct PairDatasetBuildResult {
    std::vector<PairSample> samples;
    MetricLearningBatcherConfig batcher_config;
    bool has_sample_ids = false;
    bool has_class_ids = false;

    PairBatcher CreateBatcher() const {
        return PairBatcher(samples, batcher_config);
    }
};

struct TripletDatasetBuildResult {
    std::vector<TripletSample> samples;
    MetricLearningBatcherConfig batcher_config;
    bool has_sample_ids = false;
    bool has_class_ids = false;

    TripletBatcher CreateBatcher() const {
        return TripletBatcher(samples, batcher_config);
    }
};

class PairDatasetBuilder {
public:
    explicit PairDatasetBuilder(PairDatasetBuilderConfig config = {})
        : config_(std::move(config)) {}

    PairDatasetBuildResult Build(
        const std::vector<PairDatasetRow>& rows) const {
        ValidateRows(rows);

        PairDatasetBuildResult result;
        result.samples.reserve(rows.size());
        result.batcher_config = config_.batcher;
        result.has_sample_ids = rows.front().has_sample_ids;
        result.has_class_ids = rows.front().has_class_ids;

        for (const auto& row : rows) {
            PairSample sample;
            sample.input_a = row.input_a;
            sample.input_b = row.input_b;
            sample.label = ResolveLabel(row);
            sample.sample_id_a = row.sample_id_a;
            sample.sample_id_b = row.sample_id_b;
            sample.class_id_a = row.class_id_a;
            sample.class_id_b = row.class_id_b;
            sample.has_sample_ids = row.has_sample_ids;
            sample.has_class_ids = row.has_class_ids;
            result.samples.push_back(std::move(sample));
        }

        return result;
    }

private:
    float ResolveLabel(const PairDatasetRow& row) const {
        if (row.has_label) {
            return row.label;
        }
        if (!config_.derive_labels_from_class_ids || !row.has_class_ids) {
            throw std::runtime_error(
                "PairDatasetBuilder row is missing a pair label");
        }
        const bool same_class = row.class_id_a == row.class_id_b;
        switch (config_.batcher.label_convention) {
            case MetricLearningLabelConvention::
                ContrastiveZeroSimilarOneDissimilar:
                return same_class ? 0.0f : 1.0f;
            case MetricLearningLabelConvention::
                CosineOneSimilarNegativeOneDissimilar:
                return same_class ? 1.0f : -1.0f;
            case MetricLearningLabelConvention::TripletNoLabels:
                throw std::runtime_error(
                    "PairDatasetBuilder cannot derive pair labels for triplets");
        }
        throw std::runtime_error(
            "PairDatasetBuilder cannot resolve pair label convention");
    }

    void ValidateRows(const std::vector<PairDatasetRow>& rows) const {
        if (rows.empty()) {
            throw std::runtime_error(
                "PairDatasetBuilder requires at least one row");
        }
        if (config_.batcher.label_convention ==
            MetricLearningLabelConvention::TripletNoLabels) {
            throw std::runtime_error(
                "PairDatasetBuilder requires a pair label convention");
        }

        const size_t width = rows.front().input_a.size();
        const bool has_sample_ids = rows.front().has_sample_ids;
        const bool has_class_ids = rows.front().has_class_ids;
        if (width == 0) {
            throw std::runtime_error(
                "PairDatasetBuilder row 0: input_a is empty");
        }

        for (size_t i = 0; i < rows.size(); ++i) {
            const auto& row = rows[i];
            const std::string prefix =
                "PairDatasetBuilder row " + std::to_string(i) + ": ";
            if (row.input_a.size() != width || row.input_b.size() != width) {
                throw std::runtime_error(
                    prefix + "input_a and input_b must match feature width");
            }
            if (row.has_sample_ids != has_sample_ids ||
                row.has_class_ids != has_class_ids) {
                throw std::runtime_error(
                    prefix + "metadata ID presence must be consistent");
            }
            if (config_.require_labels && !row.has_label &&
                !config_.derive_labels_from_class_ids) {
                throw std::runtime_error(prefix + "pair label is missing");
            }
            const float label = ResolveLabel(row);
            if (!IsValidMetricLearningLabel(config_.batcher.label_convention,
                                            label)) {
                throw std::runtime_error(
                    prefix + "pair label does not match convention");
            }
        }
    }

    PairDatasetBuilderConfig config_;
};

class TripletDatasetBuilder {
public:
    explicit TripletDatasetBuilder(TripletDatasetBuilderConfig config = {})
        : config_(std::move(config)) {
        config_.batcher.label_convention =
            MetricLearningLabelConvention::TripletNoLabels;
    }

    TripletDatasetBuildResult Build(
        const std::vector<TripletDatasetRow>& rows) const {
        ValidateRows(rows);

        TripletDatasetBuildResult result;
        result.samples.reserve(rows.size());
        result.batcher_config = config_.batcher;
        result.batcher_config.label_convention =
            MetricLearningLabelConvention::TripletNoLabels;
        result.has_sample_ids = rows.front().has_sample_ids;
        result.has_class_ids = rows.front().has_class_ids;

        for (const auto& row : rows) {
            TripletSample sample;
            sample.anchor = row.anchor;
            sample.positive = row.positive;
            sample.negative = row.negative;
            sample.anchor_sample_id = row.anchor_sample_id;
            sample.positive_sample_id = row.positive_sample_id;
            sample.negative_sample_id = row.negative_sample_id;
            sample.anchor_class_id = row.anchor_class_id;
            sample.positive_class_id = row.positive_class_id;
            sample.negative_class_id = row.negative_class_id;
            sample.has_sample_ids = row.has_sample_ids;
            sample.has_class_ids = row.has_class_ids;
            result.samples.push_back(std::move(sample));
        }

        return result;
    }

private:
    void ValidateRows(const std::vector<TripletDatasetRow>& rows) const {
        if (rows.empty()) {
            throw std::runtime_error(
                "TripletDatasetBuilder requires at least one row");
        }

        const size_t width = rows.front().anchor.size();
        const bool has_sample_ids = rows.front().has_sample_ids;
        const bool has_class_ids = rows.front().has_class_ids;
        if (width == 0) {
            throw std::runtime_error(
                "TripletDatasetBuilder row 0: anchor is empty");
        }

        for (size_t i = 0; i < rows.size(); ++i) {
            const auto& row = rows[i];
            const std::string prefix =
                "TripletDatasetBuilder row " + std::to_string(i) + ": ";
            if (row.anchor.size() != width ||
                row.positive.size() != width ||
                row.negative.size() != width) {
                throw std::runtime_error(
                    prefix + "branch feature widths must match");
            }
            if (row.has_sample_ids != has_sample_ids ||
                row.has_class_ids != has_class_ids) {
                throw std::runtime_error(
                    prefix + "metadata ID presence must be consistent");
            }
            if (config_.validate_class_ids && row.has_class_ids &&
                (row.anchor_class_id != row.positive_class_id ||
                 row.anchor_class_id == row.negative_class_id)) {
                throw std::runtime_error(
                    prefix +
                    "anchor/positive class IDs must match and negative must differ");
            }
        }
    }

    TripletDatasetBuilderConfig config_;
};

inline PairDatasetBuildResult BuildPairDataset(
    const std::vector<PairDatasetRow>& rows,
    PairDatasetBuilderConfig config = {}) {
    return PairDatasetBuilder(std::move(config)).Build(rows);
}

inline TripletDatasetBuildResult BuildTripletDataset(
    const std::vector<TripletDatasetRow>& rows,
    TripletDatasetBuilderConfig config = {}) {
    return TripletDatasetBuilder(std::move(config)).Build(rows);
}

}  // namespace cyxwiz
