#pragma once

#include "dataset_batcher.h"
#include "data_registry.h"
#include "graph_compiler.h"
#include "formats/text_dataset.h"
#include <memory>
#include <random>
#include <vector>

namespace cyxwiz {

/**
 * TextDatasetBatcher — IBatcher implementation for text datasets.
 *
 * Wraps a TextDataset (which owns the tokenizer + vocabulary and
 * produces token-ID sequences per sample) and stacks samples into
 * fixed-size batches of shape [batch_size, max_length]. The tokenized
 * vectors are already padded/truncated to max_length by the
 * underlying TextDataset::GetItem, so we just concatenate.
 *
 * Supports both dialog-fallback mode (entry's TextDatasetConfig is
 * authoritative) and graph-override mode (TextPreprocessingConfig
 * from TextTokenizer/TextVocabulary/TextPadding nodes overrides
 * specific sub-configs). Presence flags on the preprocess_config
 * tell us which sub-configs to respect.
 *
 * Uses the same shuffled-split + SetPhase pattern as Phase 2.1
 * audio — train and val are separate index lists drawn from a
 * single shuffled permutation of the dataset.
 */
class TextDatasetBatcher : public IBatcher {
public:
    TextDatasetBatcher(const DataRegistry::TextDatasetEntry& entry,
                       const TextPreprocessingConfig& preprocess_config,
                       int batch_size,
                       float train_split = 0.8f,
                       bool shuffle = true);

    // IBatcher interface
    Batch GetNextBatch() override;
    void Reset() override;
    bool IsEpochComplete() const override;
    size_t GetNumBatches() const override;
    size_t GetNumSamples() const override;

    void SetNormalization(float mean, float std_dev) override;
    void SetOneHotEncoding(size_t num_classes) override;
    void SetFlatten(bool flatten) override;
    void SetPhase(BatcherPhase phase) override;

    // Text-specific accessors
    int GetMaxLength() const { return max_length_; }
    size_t GetVocabSize() const;
    size_t GetNumValSamples() const { return val_indices_.size(); }

private:
    std::shared_ptr<TextDataset> dataset_;

    int batch_size_;
    bool shuffle_;
    int max_length_ = 0;     // sequence length per sample (from config after overrides)

    // Normalization is a no-op for text (token IDs are categorical).
    // Kept only to satisfy the IBatcher interface.
    float norm_mean_ = 0.0f;
    float norm_std_ = 1.0f;
    bool do_normalize_ = false;

    size_t num_classes_ = 0;
    bool do_onehot_ = false;

    // Separate train/val index sets, shuffled split at construction.
    std::vector<size_t> train_indices_;
    std::vector<size_t> val_indices_;
    BatcherPhase current_phase_ = BatcherPhase::Train;

    std::vector<size_t> epoch_order_;
    size_t current_idx_ = 0;

    std::mt19937 rng_;
};

} // namespace cyxwiz
