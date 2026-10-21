#pragma once

#include "checkpoint_tokenizer.h"
#include "graph_compiler.h"
#include "dataset_batcher.h"
#include "parquet_sequence_batcher.h"

#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class ArrowDataset;

struct SequenceArrowBatcherBuildResult {
    std::unique_ptr<ISequenceBatcher> batcher;
    std::vector<std::string> id_to_label;
    std::string tokenizer_config_json;
    std::string tokenizer_vocabulary_artifact;
    std::string error_message;
    size_t sample_count = 0;
    size_t sequence_length = 0;
    size_t token_vocabulary_size = 0;
    size_t pos_vocabulary_size = 0;
    size_t tag_vocabulary_size = 0;
    int64_t word_pad_id = 0;
    int64_t pos_pad_id = 0;

    bool success() const {
        return batcher != nullptr && error_message.empty();
    }
};

// Causal-LM token windows with the training set on disk (TOFIX154 step 6):
// read a row group at a time (ParquetSequenceBatcher). Validation / Test are
// their own sources (a file or a table in memory); ratio splitting a windows
// file is refused, since a document could straddle a row group.
SequenceArrowBatcherBuildResult BuildSequenceBatcherFromWindowSources(
    const SequenceWindowSource& train,
    const TrainingConfiguration& config,
    int batch_size,
    const SequenceWindowSource& validation = {},
    const SequenceWindowSource& test = {});

SequenceArrowBatcherBuildResult BuildSequenceBatcherFromArrowDataset(
    const std::shared_ptr<ArrowDataset>& dataset,
    const TrainingConfiguration& config,
    int batch_size,
    const std::shared_ptr<ArrowDataset>& validation = nullptr,
    const std::shared_ptr<ArrowDataset>& test = nullptr);

void ApplySequenceBatcherBuildResultToTrainingConfig(
    const SequenceArrowBatcherBuildResult& build,
    TrainingConfiguration& config);

// Evaluation of a loaded causal-LM checkpoint (any sequence graph): a compiled
// graph has no frozen token vocabulary; training gets it by building the
// sequence batcher from its prepared dataset. Do the same from `dataset` (the
// graph's prepared training dataset) when the vocabulary is missing. Returns
// false with `error` set if it cannot be prepared; a no-op for other graphs.
bool PrepareSequenceEvaluationVocabulary(
    TrainingConfiguration& config,
    const std::shared_ptr<ArrowDataset>& dataset,
    std::string& error);

// The run's tokenizer and what training read from the data, kept in each
// checkpoint (TOFIX159); empty when the run has no tokenizer.
CheckpointTokenizer CheckpointTokenizerFromConfig(const TrainingConfiguration& config);

// A checkpoint's stored tokenizer into `config`: the sequence vocabulary,
// length and pad ids, the layer sizes, and the generation previews'
// tokenizer - what PrepareSequenceEvaluationVocabulary (and the compiler,
// for previews) otherwise read from the training data.
void ApplyCheckpointTokenizer(const CheckpointTokenizer& tokenizer, TrainingConfiguration& config);

} // namespace cyxwiz
