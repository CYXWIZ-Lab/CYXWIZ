#pragma once

#include "metric_learning_batch.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cyxwiz {

struct PairSample {
    std::vector<float> input_a;
    std::vector<float> input_b;
    float label = 0.0f;
    int64_t sample_id_a = 0;
    int64_t sample_id_b = 0;
    int64_t class_id_a = 0;
    int64_t class_id_b = 0;
    bool has_sample_ids = false;
    bool has_class_ids = false;
};

struct TripletSample {
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

struct MetricLearningBatcherConfig {
    size_t batch_size = 32;
    std::vector<size_t> input_shape;  // Empty means infer one flat dimension.
    bool shuffle = false;
    bool drop_last = false;
    uint32_t seed = 42;
    MetricLearningLabelConvention label_convention =
        MetricLearningLabelConvention::ContrastiveZeroSimilarOneDissimilar;
};

class PairBatcher final : public IPairBatcher {
public:
    PairBatcher(std::vector<PairSample> samples,
                MetricLearningBatcherConfig config)
        : samples_(std::move(samples)),
          config_(std::move(config)),
          input_shape_(ResolvePairInputShape(samples_, config_.input_shape)),
          sample_width_(FlattenedElementCount(input_shape_)),
          rng_(config_.seed) {
        config_.batch_size = std::max<size_t>(1, config_.batch_size);
        ValidateSamples();
        indices_.resize(samples_.size());
        std::iota(indices_.begin(), indices_.end(), 0);
        Reset();
    }

    PairBatch GetNextPairBatch() override {
        if (IsEpochComplete() || samples_.empty()) {
            return {};
        }
        const size_t remaining = indices_.size() - current_index_;
        if (config_.drop_last && remaining < config_.batch_size) {
            current_index_ = indices_.size();
            return {};
        }

        const size_t actual_size = std::min(config_.batch_size, remaining);
        std::vector<float> input_a(actual_size * sample_width_);
        std::vector<float> input_b(actual_size * sample_width_);
        std::vector<float> labels(actual_size);
        std::vector<int64_t> sample_id_a;
        std::vector<int64_t> sample_id_b;
        std::vector<int64_t> class_id_a;
        std::vector<int64_t> class_id_b;
        if (emit_sample_ids_) {
            sample_id_a.resize(actual_size);
            sample_id_b.resize(actual_size);
        }
        if (emit_class_ids_) {
            class_id_a.resize(actual_size);
            class_id_b.resize(actual_size);
        }

        for (size_t row = 0; row < actual_size; ++row) {
            const auto& sample = samples_[indices_[current_index_ + row]];
            const size_t offset = row * sample_width_;
            std::copy(sample.input_a.begin(), sample.input_a.end(),
                      input_a.begin() + offset);
            std::copy(sample.input_b.begin(), sample.input_b.end(),
                      input_b.begin() + offset);
            labels[row] = sample.label;
            if (emit_sample_ids_) {
                sample_id_a[row] = sample.sample_id_a;
                sample_id_b[row] = sample.sample_id_b;
            }
            if (emit_class_ids_) {
                class_id_a[row] = sample.class_id_a;
                class_id_b[row] = sample.class_id_b;
            }
        }

        current_index_ += actual_size;

        PairBatch batch;
        const auto tensor_shape = BatchTensorShape(actual_size);
        batch.input_a = Tensor(tensor_shape, input_a.data());
        batch.input_b = Tensor(tensor_shape, input_b.data());
        batch.pair_label = Tensor({actual_size}, labels.data());
        if (emit_sample_ids_) {
            batch.sample_id_a =
                Tensor({actual_size}, sample_id_a.data(), DataType::Int64);
            batch.sample_id_b =
                Tensor({actual_size}, sample_id_b.data(), DataType::Int64);
        }
        if (emit_class_ids_) {
            batch.class_id_a =
                Tensor({actual_size}, class_id_a.data(), DataType::Int64);
            batch.class_id_b =
                Tensor({actual_size}, class_id_b.data(), DataType::Int64);
        }
        batch.size = actual_size;
        return batch;
    }

    void Reset() override {
        current_index_ = 0;
        if (config_.shuffle) {
            std::shuffle(indices_.begin(), indices_.end(), rng_);
        }
    }

    bool IsEpochComplete() const override {
        return current_index_ >= indices_.size();
    }

    size_t GetNumBatches() const override {
        if (indices_.empty()) {
            return 0;
        }
        if (config_.drop_last) {
            return indices_.size() / config_.batch_size;
        }
        return (indices_.size() + config_.batch_size - 1) /
               config_.batch_size;
    }

    size_t GetNumSamples() const override { return indices_.size(); }

private:
    static size_t FlattenedElementCount(const std::vector<size_t>& shape) {
        if (shape.empty()) {
            return 0;
        }
        size_t count = 1;
        for (const size_t dim : shape) {
            count *= dim;
        }
        return count;
    }

    static std::vector<size_t> ResolvePairInputShape(
        const std::vector<PairSample>& samples,
        const std::vector<size_t>& requested_shape) {
        if (!requested_shape.empty()) {
            return requested_shape;
        }
        if (samples.empty() || samples.front().input_a.empty()) {
            return {};
        }
        return {samples.front().input_a.size()};
    }

    std::vector<size_t> BatchTensorShape(size_t batch_size) const {
        std::vector<size_t> shape;
        shape.reserve(input_shape_.size() + 1);
        shape.push_back(batch_size);
        shape.insert(shape.end(), input_shape_.begin(), input_shape_.end());
        return shape;
    }

    void ValidateSamples() {
        if (samples_.empty()) {
            return;
        }
        if (sample_width_ == 0) {
            throw std::invalid_argument(
                "PairBatcher input_shape must have at least one element");
        }
        emit_sample_ids_ = samples_.front().has_sample_ids;
        emit_class_ids_ = samples_.front().has_class_ids;
        for (const auto& sample : samples_) {
            if (sample.input_a.size() != sample_width_ ||
                sample.input_b.size() != sample_width_) {
                throw std::invalid_argument(
                    "PairBatcher samples must match input_shape");
            }
            if (!IsValidMetricLearningLabel(config_.label_convention,
                                            sample.label)) {
                throw std::invalid_argument(
                    "PairBatcher sample label does not match convention");
            }
            if (sample.has_sample_ids != emit_sample_ids_ ||
                sample.has_class_ids != emit_class_ids_) {
                throw std::invalid_argument(
                    "PairBatcher samples must have consistent metadata IDs");
            }
        }
    }

    std::vector<PairSample> samples_;
    MetricLearningBatcherConfig config_;
    std::vector<size_t> input_shape_;
    size_t sample_width_ = 0;
    bool emit_sample_ids_ = false;
    bool emit_class_ids_ = false;
    std::vector<size_t> indices_;
    size_t current_index_ = 0;
    std::mt19937 rng_;
};

class TripletBatcher final : public ITripletBatcher {
public:
    TripletBatcher(std::vector<TripletSample> samples,
                   MetricLearningBatcherConfig config)
        : samples_(std::move(samples)),
          config_(std::move(config)),
          input_shape_(ResolveTripletInputShape(samples_,
                                                config_.input_shape)),
          sample_width_(FlattenedElementCount(input_shape_)),
          rng_(config_.seed) {
        config_.batch_size = std::max<size_t>(1, config_.batch_size);
        ValidateSamples();
        indices_.resize(samples_.size());
        std::iota(indices_.begin(), indices_.end(), 0);
        Reset();
    }

    TripletBatch GetNextTripletBatch() override {
        if (IsEpochComplete() || samples_.empty()) {
            return {};
        }
        const size_t remaining = indices_.size() - current_index_;
        if (config_.drop_last && remaining < config_.batch_size) {
            current_index_ = indices_.size();
            return {};
        }

        const size_t actual_size = std::min(config_.batch_size, remaining);
        std::vector<float> anchor(actual_size * sample_width_);
        std::vector<float> positive(actual_size * sample_width_);
        std::vector<float> negative(actual_size * sample_width_);
        std::vector<int64_t> anchor_sample_id;
        std::vector<int64_t> positive_sample_id;
        std::vector<int64_t> negative_sample_id;
        std::vector<int64_t> anchor_class_id;
        std::vector<int64_t> positive_class_id;
        std::vector<int64_t> negative_class_id;
        if (emit_sample_ids_) {
            anchor_sample_id.resize(actual_size);
            positive_sample_id.resize(actual_size);
            negative_sample_id.resize(actual_size);
        }
        if (emit_class_ids_) {
            anchor_class_id.resize(actual_size);
            positive_class_id.resize(actual_size);
            negative_class_id.resize(actual_size);
        }

        for (size_t row = 0; row < actual_size; ++row) {
            const auto& sample = samples_[indices_[current_index_ + row]];
            const size_t offset = row * sample_width_;
            std::copy(sample.anchor.begin(), sample.anchor.end(),
                      anchor.begin() + offset);
            std::copy(sample.positive.begin(), sample.positive.end(),
                      positive.begin() + offset);
            std::copy(sample.negative.begin(), sample.negative.end(),
                      negative.begin() + offset);
            if (emit_sample_ids_) {
                anchor_sample_id[row] = sample.anchor_sample_id;
                positive_sample_id[row] = sample.positive_sample_id;
                negative_sample_id[row] = sample.negative_sample_id;
            }
            if (emit_class_ids_) {
                anchor_class_id[row] = sample.anchor_class_id;
                positive_class_id[row] = sample.positive_class_id;
                negative_class_id[row] = sample.negative_class_id;
            }
        }

        current_index_ += actual_size;

        TripletBatch batch;
        const auto tensor_shape = BatchTensorShape(actual_size);
        batch.anchor = Tensor(tensor_shape, anchor.data());
        batch.positive = Tensor(tensor_shape, positive.data());
        batch.negative = Tensor(tensor_shape, negative.data());
        if (emit_sample_ids_) {
            batch.anchor_sample_id =
                Tensor({actual_size}, anchor_sample_id.data(),
                       DataType::Int64);
            batch.positive_sample_id =
                Tensor({actual_size}, positive_sample_id.data(),
                       DataType::Int64);
            batch.negative_sample_id =
                Tensor({actual_size}, negative_sample_id.data(),
                       DataType::Int64);
        }
        if (emit_class_ids_) {
            batch.anchor_class_id =
                Tensor({actual_size}, anchor_class_id.data(),
                       DataType::Int64);
            batch.positive_class_id =
                Tensor({actual_size}, positive_class_id.data(),
                       DataType::Int64);
            batch.negative_class_id =
                Tensor({actual_size}, negative_class_id.data(),
                       DataType::Int64);
        }
        batch.size = actual_size;
        return batch;
    }

    void Reset() override {
        current_index_ = 0;
        if (config_.shuffle) {
            std::shuffle(indices_.begin(), indices_.end(), rng_);
        }
    }

    bool IsEpochComplete() const override {
        return current_index_ >= indices_.size();
    }

    size_t GetNumBatches() const override {
        if (indices_.empty()) {
            return 0;
        }
        if (config_.drop_last) {
            return indices_.size() / config_.batch_size;
        }
        return (indices_.size() + config_.batch_size - 1) /
               config_.batch_size;
    }

    size_t GetNumSamples() const override { return indices_.size(); }

private:
    static size_t FlattenedElementCount(const std::vector<size_t>& shape) {
        if (shape.empty()) {
            return 0;
        }
        size_t count = 1;
        for (const size_t dim : shape) {
            count *= dim;
        }
        return count;
    }

    static std::vector<size_t> ResolveTripletInputShape(
        const std::vector<TripletSample>& samples,
        const std::vector<size_t>& requested_shape) {
        if (!requested_shape.empty()) {
            return requested_shape;
        }
        if (samples.empty() || samples.front().anchor.empty()) {
            return {};
        }
        return {samples.front().anchor.size()};
    }

    std::vector<size_t> BatchTensorShape(size_t batch_size) const {
        std::vector<size_t> shape;
        shape.reserve(input_shape_.size() + 1);
        shape.push_back(batch_size);
        shape.insert(shape.end(), input_shape_.begin(), input_shape_.end());
        return shape;
    }

    void ValidateSamples() {
        if (samples_.empty()) {
            return;
        }
        if (sample_width_ == 0) {
            throw std::invalid_argument(
                "TripletBatcher input_shape must have at least one element");
        }
        emit_sample_ids_ = samples_.front().has_sample_ids;
        emit_class_ids_ = samples_.front().has_class_ids;
        for (const auto& sample : samples_) {
            if (sample.anchor.size() != sample_width_ ||
                sample.positive.size() != sample_width_ ||
                sample.negative.size() != sample_width_) {
                throw std::invalid_argument(
                    "TripletBatcher samples must match input_shape");
            }
            if (sample.has_sample_ids != emit_sample_ids_ ||
                sample.has_class_ids != emit_class_ids_) {
                throw std::invalid_argument(
                    "TripletBatcher samples must have consistent metadata IDs");
            }
        }
    }

    std::vector<TripletSample> samples_;
    MetricLearningBatcherConfig config_;
    std::vector<size_t> input_shape_;
    size_t sample_width_ = 0;
    bool emit_sample_ids_ = false;
    bool emit_class_ids_ = false;
    std::vector<size_t> indices_;
    size_t current_index_ = 0;
    std::mt19937 rng_;
};

}  // namespace cyxwiz
