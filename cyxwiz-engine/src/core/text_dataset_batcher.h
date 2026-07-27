#pragma once

#include "data_registry.h"
#include "dataset_batcher.h"
#include "graph_compiler.h"

#include <cstdint>
#include <memory>
#include <string>

namespace cyxwiz {

/**
 * TextDatasetBatcher - IBatcher compatibility wrapper for legacy text
 * training.
 *
 * It materializes the TextDataset source into a raw Arrow table, applies
 * TextTokenizerOperator, then delegates batch iteration to
 * ArrowDatasetBatcher. The public API stays stable for TrainingManager and
 * TestExecutor while text batches now consume tokenized Arrow rows.
 *
 * Supports dialog-fallback config plus explicit legacy override config.
 * Train/val/test are represented with an internal partition column on the
 * tokenized table.
 */
class TextDatasetBatcher : public IBatcher {
public:
    TextDatasetBatcher(const DataRegistry::TextDatasetEntry& entry,
                       const TextPreprocessingConfig& preprocess_config,
                       int batch_size,
                       float train_split = 0.8f,
                       float val_split = 0.1f,
                       float test_split = 0.1f,
                       bool shuffle = true,
                       int num_workers = 0,
                       uint32_t seed = 42,
                       bool stratified = false,
                       uint32_t split_seed = 42,
                       bool balance_classes = false,
                       const std::string& balance_mode = "none",
                       const std::string& balance_target = "max",
                       uint32_t balance_seed = 42);

    Batch GetNextBatch() override;
    void Reset() override;
    bool IsEpochComplete() const override;
    size_t GetNumBatches() const override;
    size_t GetNumSamples() const override;

    void SetNormalization(float mean, float std_dev) override;
    void SetOneHotEncoding(size_t num_classes) override;
    void SetScalarLabelMode(bool enable) override;
    void SetFlatten(bool flatten) override;
    void SetPhase(BatcherPhase phase) override;

    int GetMaxLength() const { return max_length_; }
    size_t GetVocabSize() const;
    size_t GetNumValSamples() const { return val_samples_; }
    size_t GetNumTestSamples() const { return test_samples_; }
    bool TryApplyBalancedClassWeights(TrainingConfiguration& config) const;

private:
    std::shared_ptr<ArrowDataset> tokenized_dataset_;
    std::unique_ptr<ArrowDatasetBatcher> train_batcher_;
    std::unique_ptr<ArrowDatasetBatcher> val_batcher_;
    std::unique_ptr<ArrowDatasetBatcher> test_batcher_;
    ArrowDatasetBatcher* active_batcher_ = nullptr;

    int batch_size_ = 0;
    int num_workers_ = 0;
    int max_length_ = 0;

    float norm_mean_ = 0.0f;
    float norm_std_ = 1.0f;
    bool do_normalize_ = false;

    size_t num_classes_ = 0;
    size_t vocab_size_ = 0;
    size_t val_samples_ = 0;
    size_t test_samples_ = 0;
};

} // namespace cyxwiz
