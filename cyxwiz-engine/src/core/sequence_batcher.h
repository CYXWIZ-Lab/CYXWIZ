#pragma once

#include "dataset_batcher.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace cyxwiz {

struct SequenceSample {
    std::vector<int64_t> word_ids;
    std::vector<int64_t> pos_ids;
    std::vector<int64_t> tag_ids;
};

struct SequenceBatcherConfig {
    size_t batch_size = 32;
    size_t max_sequence_length = 0;  // 0 = infer from samples
    bool shuffle = false;
    bool drop_last = false;
    bool create_attention_mask = true;
    bool create_causal_lm_targets = false;
    int64_t word_pad_id = 0;
    int64_t pos_pad_id = 0;
    int64_t tag_ignore_index = -100;
    int64_t target_ignore_index = -100;
    uint32_t seed = 42;
};

class SequenceBatcher : public ISequenceBatcher {
public:
    SequenceBatcher(std::vector<SequenceSample> samples,
                    SequenceBatcherConfig config)
        : samples_(std::move(samples)),
          config_(config),
          rng_(config_.seed) {
        config_.batch_size = std::max<size_t>(1, config_.batch_size);
        sequence_length_ = ResolveSequenceLength();
        indices_.resize(samples_.size());
        std::iota(indices_.begin(), indices_.end(), 0);
        Reset();
    }

    SequenceBatch GetNextSequenceBatch() override {
        if (IsEpochComplete() || sequence_length_ == 0) {
            return {};
        }

        const size_t remaining = indices_.size() - current_index_;
        if (config_.drop_last && remaining < config_.batch_size) {
            current_index_ = indices_.size();
            return {};
        }

        const size_t actual_size = std::min(config_.batch_size, remaining);
        std::vector<int64_t> word_data(
            actual_size * sequence_length_, config_.word_pad_id);
        std::vector<int64_t> pos_data(
            actual_size * sequence_length_, config_.pos_pad_id);
        std::vector<int64_t> mask_data(
            actual_size * sequence_length_, 0);
        std::vector<int64_t> tag_data(
            actual_size * sequence_length_, config_.tag_ignore_index);
        std::vector<int64_t> target_data(
            actual_size * sequence_length_, config_.target_ignore_index);

        bool has_pos = false;
        bool has_tags = false;
        for (size_t row = 0; row < actual_size; ++row) {
            const auto& sample = samples_[indices_[current_index_ + row]];
            if (config_.create_causal_lm_targets) {
                CopyCausalLmSequence(sample.word_ids, word_data, target_data,
                                     mask_data, row);
            } else {
                CopySequence(sample.word_ids, word_data, row);
            }
            if (!sample.pos_ids.empty()) {
                has_pos = true;
                CopySequence(sample.pos_ids, pos_data, row);
            }
            if (!sample.tag_ids.empty()) {
                has_tags = true;
                CopySequence(sample.tag_ids, tag_data, row);
            }

            const size_t token_count =
                std::min(sequence_length_, sample.word_ids.size());
            for (size_t col = 0; col < token_count; ++col) {
                if (!config_.create_causal_lm_targets) {
                    mask_data[row * sequence_length_ + col] = 1;
                }
            }
        }

        current_index_ += actual_size;

        SequenceBatch batch;
        batch.word_ids = Tensor({actual_size, sequence_length_},
                                word_data.data(), DataType::Int64);
        if (has_pos) {
            batch.pos_ids = Tensor({actual_size, sequence_length_},
                                   pos_data.data(), DataType::Int64);
        }
        if (config_.create_attention_mask) {
            batch.attention_mask = Tensor({actual_size, sequence_length_},
                                          mask_data.data(), DataType::Int64);
        }
        if (has_tags) {
            batch.tag_ids = Tensor({actual_size, sequence_length_},
                                   tag_data.data(), DataType::Int64);
        }
        if (config_.create_causal_lm_targets) {
            batch.target_ids = Tensor({actual_size, sequence_length_},
                                      target_data.data(), DataType::Int64);
        }
        batch.size = actual_size;
        batch.sequence_length = sequence_length_;
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
        if (indices_.empty() || sequence_length_ == 0) {
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
    size_t ResolveSequenceLength() const {
        if (config_.max_sequence_length > 0) {
            return config_.max_sequence_length;
        }
        size_t max_length = 0;
        for (const auto& sample : samples_) {
            max_length = std::max(max_length, sample.word_ids.size());
        }
        return max_length;
    }

    void CopySequence(const std::vector<int64_t>& source,
                      std::vector<int64_t>& dest,
                      size_t row) const {
        const size_t count = std::min(sequence_length_, source.size());
        for (size_t col = 0; col < count; ++col) {
            dest[row * sequence_length_ + col] = source[col];
        }
    }

    void CopyCausalLmSequence(const std::vector<int64_t>& source,
                              std::vector<int64_t>& word_dest,
                              std::vector<int64_t>& target_dest,
                              std::vector<int64_t>& mask_dest,
                              size_t row) const {
        const size_t count = std::min(sequence_length_, source.size());
        for (size_t col = 0; col < count; ++col) {
            const size_t offset = row * sequence_length_ + col;
            const int64_t input_id = source[col];
            word_dest[offset] = input_id;
            mask_dest[offset] = input_id == config_.word_pad_id ? 0 : 1;

            const size_t target_col = col + 1;
            if (input_id != config_.word_pad_id &&
                target_col < source.size() &&
                source[target_col] != config_.word_pad_id) {
                target_dest[offset] = source[target_col];
            }
        }
    }

    std::vector<SequenceSample> samples_;
    SequenceBatcherConfig config_;
    size_t sequence_length_ = 0;
    std::vector<size_t> indices_;
    size_t current_index_ = 0;
    std::mt19937 rng_;
};

} // namespace cyxwiz
