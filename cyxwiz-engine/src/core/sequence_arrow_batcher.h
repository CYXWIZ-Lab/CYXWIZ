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
    std::string error_message;
    size_t sample_count = 0;
    size_t sequence_length = 0;
    size_t token_vocabulary_size = 0;
    size_t pos_vocabulary_size = 0;
    size_t tag_vocabulary_size = 0;

    bool success() const {
        return batcher != nullptr && error_message.empty();
    }
};

SequenceArrowBatcherBuildResult BuildSequenceBatcherFromArrowDataset(
    const std::shared_ptr<ArrowDataset>& dataset,
    const TrainingConfiguration& config,
    int batch_size);

void ApplySequenceBatcherBuildResultToTrainingConfig(
    const SequenceArrowBatcherBuildResult& build,
    TrainingConfiguration& config);

} // namespace cyxwiz
