#pragma once

#include "graph_compiler.h"
#include "dataset_batcher.h"

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

} // namespace cyxwiz
