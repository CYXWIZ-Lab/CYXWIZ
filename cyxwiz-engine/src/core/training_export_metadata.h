#pragma once

#include "graph_compiler.h"
#include "model_format.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace cyxwiz {
// Metadata only: no model computation, registry lookup or file I/O at export time.
inline ExportOptions TrainingExportMetadata(const TrainingConfiguration& source) {
    ExportOptions options;
    TrainingConfig config;
    config.optimizer_type = source.GetOptimizerName();
    config.learning_rate = source.learning_rate;
    config.momentum = source.momentum;
    config.weight_decay = source.weight_decay;
    config.beta1 = source.beta1; config.beta2 = source.beta2; config.epsilon = source.epsilon;
    config.batch_size = source.batch_size;
    config.epochs = source.epochs;
    config.loss_function = source.GetLossName();
    config.dataset_name = source.dataset_name;
    if (source.output_size > static_cast<size_t>((std::numeric_limits<int>::max)()))
        throw std::invalid_argument("Export output width exceeds metadata range");
    config.num_classes = static_cast<int>(source.output_size);
    for (size_t dim : source.input_shape) {
        if (dim > static_cast<size_t>((std::numeric_limits<int64_t>::max)()))
            throw std::invalid_argument("Export input dimension exceeds metadata range");
        config.input_shape.push_back(static_cast<int64_t>(dim));
    }
    options.trained_config = std::move(config);
    const auto& sequence = source.sequence_batch;
    options.text_tokenizer_config_json = sequence.tokenizer_config_json;
    options.text_tokenizer_vocab_data = sequence.tokenizer_vocabulary_artifact;
    options.sequence_create_causal_lm_targets = sequence.create_causal_lm_targets;
    options.sequence_max_sequence_length = static_cast<size_t>(std::max(0, sequence.max_sequence_length));
    options.sequence_batch_first = sequence.batch_first;
    options.sequence_create_attention_mask = sequence.create_attention_mask;
    options.sequence_word_pad_id = sequence.word_pad_id;
    options.sequence_pos_pad_id = sequence.pos_pad_id;
    options.sequence_tag_ignore_index = sequence.ignore_index;
    options.sequence_target_ignore_index = sequence.target_ignore_index;
    return options;
}
} // namespace cyxwiz
