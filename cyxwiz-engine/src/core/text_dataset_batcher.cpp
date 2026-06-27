#include "text_dataset_batcher.h"
#include "formats/text_dataset.h"
#include "node_executors/text_tokenizer_operator.h"
#include "split_partitioning.h"
#include "text_arrow_adapter.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>

namespace cyxwiz {

namespace {

TextDatasetConfig BuildTextDatasetConfig(
    const DataRegistry::TextDatasetEntry& entry,
    const TextPreprocessingConfig& preprocess_config) {

    TextDatasetConfig cfg;
    cfg.text_column = entry.text_column;
    cfg.label_column = entry.label_column;
    cfg.has_labels = entry.has_labels;

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

    if (preprocess_config.has_tokenizer_node) {
        switch (preprocess_config.tokenizer_type) {
            case 0: cfg.tokenizer_type = TokenizerType::Whitespace; break;
            case 2: cfg.tokenizer_type = TokenizerType::Character; break;
            case 1:
            default: cfg.tokenizer_type = TokenizerType::Word; break;
        }
        cfg.lowercase      = preprocess_config.lowercase;
        cfg.do_padding     = preprocess_config.do_padding;
        cfg.do_truncation  = preprocess_config.do_truncation;
        cfg.max_length     = preprocess_config.max_length;
        cfg.min_word_freq  = preprocess_config.min_word_freq;
        cfg.max_vocab_size = preprocess_config.max_vocab_size;
        spdlog::info("TextDatasetBatcher: tokenizer override config applies "
                     "(type={}, max_length={}, lowercase={})",
                     preprocess_config.tokenizer_type,
                     cfg.max_length, cfg.lowercase);
    }
    if (preprocess_config.has_vocabulary_node) {
        cfg.min_word_freq  = preprocess_config.min_word_freq;
        cfg.max_vocab_size = preprocess_config.max_vocab_size;
        cfg.vocab_file     = preprocess_config.vocab_file;
        spdlog::info("TextDatasetBatcher: vocabulary override config applies "
                     "(min_freq={}, max_vocab_size={}, vocab_file='{}')",
                     cfg.min_word_freq, cfg.max_vocab_size, cfg.vocab_file);
    }
    if (preprocess_config.has_padding_node) {
        cfg.do_padding = true;
        cfg.max_length = preprocess_config.max_length;
        spdlog::info("TextDatasetBatcher: padding override config applies "
                     "(max_length={})", cfg.max_length);
    }

    return cfg;
}

int TokenizerTypeToParam(TokenizerType type) {
    switch (type) {
        case TokenizerType::Whitespace: return 0;
        case TokenizerType::Character: return 2;
        case TokenizerType::Word:
        default: return 1;
    }
}

} // namespace

TextDatasetBatcher::TextDatasetBatcher(
    const DataRegistry::TextDatasetEntry& entry,
    const TextPreprocessingConfig& preprocess_config,
    int batch_size,
    float train_split,
    float val_split,
    float test_split,
    bool shuffle,
    int num_workers,
    uint32_t seed,
    bool stratified,
    uint32_t split_seed,
    bool balance_classes,
    const std::string& balance_mode,
    const std::string& balance_target,
    uint32_t balance_seed)
    : batch_size_(batch_size),
      num_workers_(std::max(0, num_workers))
{
    const TextDatasetConfig cfg =
        BuildTextDatasetConfig(entry, preprocess_config);
    max_length_ = cfg.max_length;

    std::shared_ptr<TextDataset> raw_dataset;
    try {
        raw_dataset = std::make_shared<TextDataset>(
            entry.source_path, cfg, TextDatasetLoadMode::RawOnly);
    } catch (const std::exception& e) {
        spdlog::error("TextDatasetBatcher: failed to construct TextDataset: {}", e.what());
        return;
    }

    if (!raw_dataset || raw_dataset->Size() == 0) {
        spdlog::error("TextDatasetBatcher: dataset is empty or null");
        return;
    }

    const auto info = raw_dataset->GetInfo();
    num_classes_ = entry.num_classes > 0 ? entry.num_classes : info.num_classes;

    const std::string raw_label_col =
        cfg.has_labels ? cfg.label_column : std::string{};
    auto raw_table_result = BuildRawTextArrowTable(
        *raw_dataset, cfg.text_column, raw_label_col);
    if (!raw_table_result.ok()) {
        spdlog::error("TextDatasetBatcher: failed to build raw Arrow table: {}",
                      raw_table_result.status().ToString());
        return;
    }

    std::map<std::string, std::string> tokenizer_params = {
        {"text_col", cfg.text_column},
        {"tokenizer_type", std::to_string(TokenizerTypeToParam(cfg.tokenizer_type))},
        {"max_length", std::to_string(cfg.max_length)},
        {"lowercase", cfg.lowercase ? "true" : "false"},
        {"min_word_freq", std::to_string(cfg.min_word_freq)},
        {"max_vocab_size", std::to_string(cfg.max_vocab_size)},
    };
    if (!raw_label_col.empty()) {
        tokenizer_params["label_col"] = raw_label_col;
    }

    TextTokenizerOperator tokenizer;
    std::string error;
    if (!tokenizer.Configure(tokenizer_params, error)) {
        spdlog::error("TextDatasetBatcher: tokenizer configure failed: {}", error);
        return;
    }

    auto tokenized_result = tokenizer.Apply(raw_table_result.ValueOrDie());
    if (!tokenized_result.ok()) {
        spdlog::error("TextDatasetBatcher: tokenizer apply failed: {}",
                      tokenized_result.status().ToString());
        return;
    }
    auto tokenized_table = tokenized_result.ValueOrDie();
    if (!tokenized_table) {
        spdlog::error("TextDatasetBatcher: tokenizer returned null table");
        return;
    }
    vocab_size_ = tokenizer.GetLastVocabSize();

    auto partitioned_result = AddSplitPartitionColumn(
        tokenized_table,
        SplitPartitionOptions{
            "y",
            train_split,
            val_split,
            test_split,
            shuffle,
            split_seed,
            stratified,
            "TextDatasetBatcher"});
    if (!partitioned_result.ok()) {
        spdlog::error("TextDatasetBatcher: partitioning tokenized table failed: {}",
                      partitioned_result.status().ToString());
        return;
    }
    tokenized_dataset_ = std::make_shared<ArrowDataset>(
        partitioned_result.ValueOrDie(), "legacy_text_tokenized");

    const size_t normalized_batch_size =
        static_cast<size_t>(std::max(1, batch_size_));
    train_batcher_ = std::make_unique<ArrowDatasetBatcher>(
        tokenized_dataset_, "y", normalized_batch_size,
        shuffle, 1.0f, true, "__partition__", 0, num_workers_,
        BatcherPhase::Train, 0.0f, seed,
        balance_classes, balance_mode, balance_target, balance_seed);
    val_batcher_ = std::make_unique<ArrowDatasetBatcher>(
        tokenized_dataset_, "y", normalized_batch_size,
        false, 1.0f, false, "__partition__", 1, num_workers_,
        BatcherPhase::Val, 0.0f, seed);
    test_batcher_ = std::make_unique<ArrowDatasetBatcher>(
        tokenized_dataset_, "y", normalized_batch_size,
        false, 1.0f, false, "__partition__", 2, num_workers_,
        BatcherPhase::Test, 0.0f, seed);

    if (num_classes_ > 0) {
        train_batcher_->SetOneHotEncoding(num_classes_);
        val_batcher_->SetOneHotEncoding(num_classes_);
        test_batcher_->SetOneHotEncoding(num_classes_);
    }

    active_batcher_ = train_batcher_.get();
    val_samples_ = val_batcher_ ? val_batcher_->GetNumSamples() : 0;
    test_samples_ = test_batcher_ ? test_batcher_->GetNumSamples() : 0;

    spdlog::info("TextDatasetBatcher: Arrow-backed compatibility path ready "
                 "({} train / {} val / {} test samples, {} classes, vocab_size={}, "
                 "max_length={}, batch_size={}, num_workers={})",
                 train_batcher_->GetNumSamples(), val_samples_, test_samples_, num_classes_,
                 vocab_size_, max_length_, batch_size_, num_workers_);
}

size_t TextDatasetBatcher::GetVocabSize() const {
    return vocab_size_;
}

Batch TextDatasetBatcher::GetNextBatch() {
    return active_batcher_ ? active_batcher_->GetNextBatch() : Batch{};
}

void TextDatasetBatcher::Reset() {
    if (active_batcher_) {
        active_batcher_->Reset();
    }
}

void TextDatasetBatcher::SetPhase(BatcherPhase phase) {
    switch (phase) {
        case BatcherPhase::Val:
            active_batcher_ = val_batcher_.get();
            break;
        case BatcherPhase::Test:
            active_batcher_ = test_batcher_.get();
            break;
        case BatcherPhase::Train:
        default:
            active_batcher_ = train_batcher_.get();
            break;
    }
}

bool TextDatasetBatcher::IsEpochComplete() const {
    return !active_batcher_ || active_batcher_->IsEpochComplete();
}

size_t TextDatasetBatcher::GetNumBatches() const {
    return active_batcher_ ? active_batcher_->GetNumBatches() : 0;
}

size_t TextDatasetBatcher::GetNumSamples() const {
    return active_batcher_ ? active_batcher_->GetNumSamples() : 0;
}

void TextDatasetBatcher::SetNormalization(float mean, float std_dev) {
    norm_mean_ = mean;
    norm_std_ = (std_dev > 0.0f) ? std_dev : 1.0f;
    do_normalize_ = true;
    spdlog::debug("TextDatasetBatcher: normalization ignored for token IDs "
                  "(mean={}, std={})", norm_mean_, norm_std_);
}

void TextDatasetBatcher::SetOneHotEncoding(size_t num_classes) {
    num_classes_ = num_classes;
    if (train_batcher_) {
        train_batcher_->SetOneHotEncoding(num_classes);
    }
    if (val_batcher_) {
        val_batcher_->SetOneHotEncoding(num_classes);
    }
    if (test_batcher_) {
        test_batcher_->SetOneHotEncoding(num_classes);
    }
}

void TextDatasetBatcher::SetFlatten(bool /*flatten*/) {
    // Text token tables are already flat per sample.
}

} // namespace cyxwiz
