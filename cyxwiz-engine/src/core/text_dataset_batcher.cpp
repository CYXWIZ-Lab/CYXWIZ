#include "text_dataset_batcher.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>
#include <utility>
#include <thread>

namespace cyxwiz {

TextDatasetBatcher::TextDatasetBatcher(
    const DataRegistry::TextDatasetEntry& entry,
    const TextPreprocessingConfig& preprocess_config,
    int batch_size,
    float train_split,
    bool shuffle,
    int num_workers)
    : batch_size_(batch_size), shuffle_(shuffle),
      num_workers_(std::max(0, num_workers)), rng_(42)
{
    // Build the TextDatasetConfig. Start from the entry's dialog-baked
    // defaults, then apply graph-preprocessing overrides on top. Each
    // of the three text node types can independently override its
    // sub-config via the `has_*_node` presence flags on
    // TextPreprocessingConfig.
    TextDatasetConfig cfg;
    cfg.text_column = entry.text_column;
    cfg.label_column = entry.label_column;
    cfg.has_labels = entry.has_labels;

    // Tokenizer + shared fields from entry (dialog defaults)
    switch (entry.tokenizer_type) {
        case 0: cfg.tokenizer_type = TokenizerType::Whitespace; break;
        case 2: cfg.tokenizer_type = TokenizerType::Character; break;
        case 1:
        default: cfg.tokenizer_type = TokenizerType::Word; break;
    }
    cfg.max_length     = entry.max_length;
    cfg.lowercase      = entry.lowercase;
    cfg.do_padding     = entry.do_padding;
    cfg.do_truncation  = entry.do_truncation;
    cfg.min_word_freq  = entry.min_word_freq;
    cfg.max_vocab_size = entry.max_vocab_size;
    cfg.vocab_file     = entry.vocab_file;

    // Graph-node overrides (Phase 3). Each extractor sets a presence
    // flag, so we only override the sub-config it explicitly touched.
    if (preprocess_config.has_tokenizer_node) {
        switch (preprocess_config.tokenizer_type) {
            case 0: cfg.tokenizer_type = TokenizerType::Whitespace; break;
            case 2: cfg.tokenizer_type = TokenizerType::Character; break;
            case 1:
            default: cfg.tokenizer_type = TokenizerType::Word; break;
        }
        cfg.lowercase     = preprocess_config.lowercase;
        cfg.do_padding    = preprocess_config.do_padding;
        cfg.do_truncation = preprocess_config.do_truncation;
        cfg.max_length    = preprocess_config.max_length;
        cfg.min_word_freq = preprocess_config.min_word_freq;
        cfg.max_vocab_size = preprocess_config.max_vocab_size;
        spdlog::info("TextDatasetBatcher: TextTokenizer node overrides dialog "
                     "defaults (type={}, max_length={}, lowercase={})",
                     preprocess_config.tokenizer_type,
                     cfg.max_length, cfg.lowercase);
    }
    if (preprocess_config.has_vocabulary_node) {
        // TextVocabulary stomps on vocab sub-config even if
        // TextTokenizer already set it. Useful for "use a pre-built
        // vocab file" workflow.
        cfg.min_word_freq  = preprocess_config.min_word_freq;
        cfg.max_vocab_size = preprocess_config.max_vocab_size;
        cfg.vocab_file     = preprocess_config.vocab_file;
        spdlog::info("TextDatasetBatcher: TextVocabulary node overrides vocab config "
                     "(min_freq={}, max_vocab_size={}, vocab_file='{}')",
                     cfg.min_word_freq, cfg.max_vocab_size, cfg.vocab_file);
    }
    if (preprocess_config.has_padding_node) {
        cfg.do_padding = true;
        cfg.max_length = preprocess_config.max_length;
        spdlog::info("TextDatasetBatcher: TextPadding node overrides padding config "
                     "(max_length={})", cfg.max_length);
    }

    max_length_ = cfg.max_length;

    try {
        dataset_ = std::make_shared<TextDataset>(entry.source_path, cfg);
    } catch (const std::exception& e) {
        spdlog::error("TextDatasetBatcher: failed to construct TextDataset: {}", e.what());
        return;
    }

    if (!dataset_ || dataset_->Size() == 0) {
        spdlog::error("TextDatasetBatcher: dataset is empty or null");
        return;
    }

    // Shuffled train/val split — same leakage-free pattern as Phase 2.1
    // audio. All_indices gets shuffled once, then we take [0..train_count)
    // as train and [train_count..total) as val. Reset() re-shuffles only
    // the train indices between epochs; val order stays deterministic.
    size_t total = dataset_->Size();
    size_t train_count = static_cast<size_t>(total * train_split);
    if (train_count == 0) train_count = total;
    if (train_count > total) train_count = total;

    std::vector<size_t> all_indices(total);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::shuffle(all_indices.begin(), all_indices.end(), rng_);

    train_indices_.assign(all_indices.begin(), all_indices.begin() + train_count);
    val_indices_.assign(all_indices.begin() + train_count, all_indices.end());

    num_classes_ = entry.num_classes;
    if (num_classes_ == 0) {
        auto info = dataset_->GetInfo();
        num_classes_ = info.num_classes;
    }

    Reset();
    spdlog::info("TextDatasetBatcher: {} train / {} val samples, {} classes, "
                 "vocab_size={}, max_length={}, batch_size={}, num_workers={}",
                 train_indices_.size(), val_indices_.size(), num_classes_,
                 GetVocabSize(), max_length_, batch_size_, num_workers_);
}

size_t TextDatasetBatcher::GetVocabSize() const {
    return dataset_ ? dataset_->GetVocabSize() : 0;
}

Batch TextDatasetBatcher::GetNextBatch() {
    Batch batch;
    if (!dataset_ || current_idx_ >= epoch_order_.size()) return batch;

    size_t actual_size = std::min(static_cast<size_t>(batch_size_),
                                   epoch_order_.size() - current_idx_);
    if (actual_size == 0) return batch;

    size_t sample_dim = static_cast<size_t>(max_length_);
    std::vector<std::pair<std::vector<float>, int>> samples(actual_size);

    auto load_range = [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            size_t idx = epoch_order_[current_idx_ + i];
            samples[i] = dataset_->GetItem(idx);
        }
    };

    if (num_workers_ > 1 && actual_size > 1) {
        size_t worker_count = std::min(static_cast<size_t>(num_workers_), actual_size);
        size_t chunk_size = (actual_size + worker_count - 1) / worker_count;
        std::vector<std::thread> workers;
        workers.reserve(worker_count);

        for (size_t worker = 0; worker < worker_count; ++worker) {
            size_t begin = worker * chunk_size;
            size_t end = std::min(actual_size, begin + chunk_size);
            if (begin >= end) break;
            workers.emplace_back(load_range, begin, end);
        }

        for (auto& worker : workers) {
            worker.join();
        }
    } else {
        load_range(0, actual_size);
    }

    std::vector<float> batch_data;
    batch_data.reserve(actual_size * sample_dim);
    std::vector<float> batch_labels;

    if (do_onehot_ && num_classes_ > 0) {
        batch_labels.resize(actual_size * num_classes_, 0.0f);
    } else {
        batch_labels.reserve(actual_size);
    }

    for (size_t i = 0; i < actual_size; ++i) {
        auto& [tokens, label] = samples[i];
        if (tokens.size() != sample_dim) {
            // TextDataset::GetItem should produce a padded/truncated
            // sequence already. If it doesn't, zero-fill so the batch
            // tensor has uniform shape.
            tokens.assign(sample_dim, 0.0f);
            label = 0;
        }

        batch_data.insert(batch_data.end(), tokens.begin(), tokens.end());

        if (do_onehot_ && num_classes_ > 0) {
            if (label >= 0 && static_cast<size_t>(label) < num_classes_) {
                batch_labels[i * num_classes_ + label] = 1.0f;
            }
        } else {
            batch_labels.push_back(static_cast<float>(label));
        }
    }

    batch.data = Tensor({actual_size, sample_dim},
                        batch_data.data(), DataType::Float32);

    if (do_onehot_ && num_classes_ > 0) {
        batch.labels = Tensor({actual_size, num_classes_},
                              batch_labels.data(), DataType::Float32);
    } else {
        batch.labels = Tensor({actual_size},
                              batch_labels.data(), DataType::Float32);
    }

    batch.size = actual_size;
    current_idx_ += actual_size;
    return batch;
}

void TextDatasetBatcher::Reset() {
    if (current_phase_ == BatcherPhase::Val) {
        epoch_order_ = val_indices_;
    } else {
        epoch_order_ = train_indices_;
        if (shuffle_) {
            std::shuffle(epoch_order_.begin(), epoch_order_.end(), rng_);
        }
    }
    current_idx_ = 0;
}

void TextDatasetBatcher::SetPhase(BatcherPhase phase) {
    current_phase_ = phase;
    // Caller is expected to call Reset() afterwards; the training
    // executor's RunValidationArrow does exactly that.
}

bool TextDatasetBatcher::IsEpochComplete() const {
    return current_idx_ >= epoch_order_.size();
}

size_t TextDatasetBatcher::GetNumBatches() const {
    if (batch_size_ <= 0) return 0;
    return (epoch_order_.size() + batch_size_ - 1) / batch_size_;
}

size_t TextDatasetBatcher::GetNumSamples() const {
    return train_indices_.size();
}

void TextDatasetBatcher::SetNormalization(float mean, float std_dev) {
    // Normalization is a no-op for text — token IDs are categorical.
    // Setter is only here to satisfy the IBatcher interface.
    norm_mean_ = mean;
    norm_std_ = (std_dev > 0.0f) ? std_dev : 1.0f;
    do_normalize_ = true;
}

void TextDatasetBatcher::SetOneHotEncoding(size_t num_classes) {
    num_classes_ = num_classes;
    do_onehot_ = true;
}

void TextDatasetBatcher::SetFlatten(bool /*flatten*/) {
    // Text is always 1D (flat) per sample; flag is ignored.
}

} // namespace cyxwiz
