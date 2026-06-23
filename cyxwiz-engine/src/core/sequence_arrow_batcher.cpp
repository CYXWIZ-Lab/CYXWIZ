#include "sequence_arrow_batcher.h"

#include "arrow_dataset.h"
#include "ner_sequence_builder.h"

#include <arrow/api.h>

#include <algorithm>
#include <cctype>
#include <sstream>

namespace cyxwiz {
namespace {

std::string Trim(std::string value) {
    auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(value.begin(),
                std::find_if(value.begin(), value.end(),
                             [&](char ch) { return !is_space(ch); }));
    value.erase(std::find_if(value.rbegin(), value.rend(),
                             [&](char ch) { return !is_space(ch); }).base(),
                value.end());
    return value;
}

std::vector<std::string> SplitWhitespace(const std::string& value) {
    std::istringstream in(value);
    std::vector<std::string> tokens;
    std::string token;
    while (in >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

bool IsStringType(const std::shared_ptr<arrow::DataType>& type) {
    return type &&
           (type->id() == arrow::Type::STRING ||
            type->id() == arrow::Type::LARGE_STRING);
}

bool ReadStringCell(const std::shared_ptr<arrow::Table>& table,
                    int column_index,
                    int64_t row_index,
                    std::string& value,
                    std::string& error) {
    auto column = table->column(column_index);
    if (!column) {
        error = "column is not readable";
        return false;
    }

    auto scalar_result = column->GetScalar(row_index);
    if (!scalar_result.ok()) {
        error = scalar_result.status().ToString();
        return false;
    }

    auto scalar = *scalar_result;
    if (!scalar || !scalar->is_valid) {
        value.clear();
        return true;
    }

    value = Trim(scalar->ToString());
    return true;
}

bool ValidateStringColumn(const std::shared_ptr<arrow::Table>& table,
                          const std::string& column_name,
                          const char* role,
                          int& column_index,
                          std::string& error) {
    if (column_name.empty()) {
        error = std::string("sequence ") + role + " column is empty";
        return false;
    }
    column_index = table->schema()->GetFieldIndex(column_name);
    if (column_index < 0) {
        error = std::string("sequence ") + role + " column '" +
                column_name + "' not found";
        return false;
    }
    auto field = table->schema()->field(column_index);
    if (!field || !IsStringType(field->type())) {
        error = std::string("sequence ") + role + " column '" +
                column_name + "' must be string/large_string";
        return false;
    }
    return true;
}

std::string DefaultColumn(const std::string& value,
                          const char* fallback) {
    return value.empty() ? std::string(fallback) : value;
}

} // namespace

size_t ResolveSequenceLength(const std::vector<SequenceSample>& samples,
                             size_t configured_length) {
    if (configured_length > 0) {
        return configured_length;
    }
    size_t max_length = 0;
    for (const auto& sample : samples) {
        max_length = std::max(max_length, sample.word_ids.size());
    }
    return max_length;
}

SequenceArrowBatcherBuildResult BuildSequenceBatcherFromArrowDataset(
    const std::shared_ptr<ArrowDataset>& dataset,
    const TrainingConfiguration& config,
    int batch_size) {
    SequenceArrowBatcherBuildResult result;

    if (!config.sequence_batch.enabled) {
        result.error_message = "sequence batch config is not enabled";
        return result;
    }
    if (!dataset) {
        result.error_message = "sequence Arrow dataset is null";
        return result;
    }

    auto table = dataset->GetArrowTable();
    if (!table || !table->schema()) {
        result.error_message = "sequence Arrow dataset has no table";
        return result;
    }

    const std::string token_column =
        DefaultColumn(config.sequence_batch.token_column, "tokens");
    const std::string tag_column =
        DefaultColumn(config.sequence_batch.tag_column, "ner_tags");

    int token_index = -1;
    int tag_index = -1;
    std::string error;
    if (!ValidateStringColumn(table, token_column, "token",
                              token_index, error) ||
        !ValidateStringColumn(table, tag_column, "tag",
                              tag_index, error)) {
        result.error_message = error;
        return result;
    }

    int pos_index = -1;
    if (!config.sequence_batch.pos_column.empty() &&
        !ValidateStringColumn(table, config.sequence_batch.pos_column, "POS",
                              pos_index, error)) {
        result.error_message = error;
        return result;
    }

    std::vector<NERSequenceRow> rows;
    rows.reserve(static_cast<size_t>(table->num_rows()));
    for (int64_t row_index = 0; row_index < table->num_rows(); ++row_index) {
        std::string token_text;
        std::string tag_text;
        std::string pos_text;
        if (!ReadStringCell(table, token_index, row_index,
                            token_text, error) ||
            !ReadStringCell(table, tag_index, row_index,
                            tag_text, error)) {
            result.error_message =
                "sequence row " + std::to_string(row_index) + ": " + error;
            return result;
        }
        if (pos_index >= 0 &&
            !ReadStringCell(table, pos_index, row_index,
                            pos_text, error)) {
            result.error_message =
                "sequence row " + std::to_string(row_index) + ": " + error;
            return result;
        }

        rows.push_back({
            SplitWhitespace(token_text),
            SplitWhitespace(pos_text),
            SplitWhitespace(tag_text),
        });
    }

    try {
        NERSequenceBuilderConfig builder_config;
        builder_config.use_pos_tags = pos_index >= 0;
        builder_config.require_tags = true;
        builder_config.batcher.batch_size =
            static_cast<size_t>(std::max(1, batch_size));
        builder_config.batcher.max_sequence_length =
            static_cast<size_t>(
                std::max(0, config.sequence_batch.max_sequence_length));
        builder_config.batcher.shuffle = config.shuffle;
        builder_config.batcher.drop_last = config.drop_last;
        builder_config.batcher.create_attention_mask =
            config.sequence_batch.create_attention_mask;
        builder_config.batcher.create_causal_lm_targets =
            config.sequence_batch.create_causal_lm_targets;
        builder_config.batcher.tag_ignore_index =
            config.sequence_batch.ignore_index;
        builder_config.batcher.target_ignore_index =
            config.sequence_batch.target_ignore_index;
        builder_config.batcher.seed =
            static_cast<uint32_t>(std::max(0, config.dataloader_seed));

        auto build = BuildNERSequenceData(rows, builder_config);
        result.id_to_label = build.tag_vocabulary.Values();
        result.sample_count = build.samples.size();
        result.sequence_length = ResolveSequenceLength(
            build.samples, build.batcher_config.max_sequence_length);
        result.token_vocabulary_size = build.token_vocabulary.Size();
        result.tag_vocabulary_size = build.tag_vocabulary.Size();
        result.batcher = std::make_unique<SequenceBatcher>(
            std::move(build.samples), build.batcher_config);
    } catch (const std::exception& ex) {
        result.error_message = ex.what();
    }

    return result;
}

void ApplySequenceBatcherBuildResultToTrainingConfig(
    const SequenceArrowBatcherBuildResult& build,
    TrainingConfiguration& config) {
    if (build.sequence_length > 0) {
        config.input_size = build.sequence_length;
        config.input_shape = {build.sequence_length};
    }
    if (build.tag_vocabulary_size > 0) {
        config.output_size = build.tag_vocabulary_size;
    }

    bool embedding_vocab_applied = false;
    int last_time_distributed = -1;
    for (size_t i = 0; i < config.layers.size(); ++i) {
        auto& layer = config.layers[i];
        if (!embedding_vocab_applied &&
            layer.type == gui::NodeType::Embedding &&
            build.token_vocabulary_size > 0) {
            layer.parameters["num_embeddings"] =
                std::to_string(build.token_vocabulary_size);
            embedding_vocab_applied = true;
        }
        if (layer.type == gui::NodeType::TimeDistributed) {
            last_time_distributed = static_cast<int>(i);
        }
    }

    if (last_time_distributed >= 0 && build.tag_vocabulary_size > 0) {
        auto& head = config.layers[static_cast<size_t>(last_time_distributed)];
        head.units = static_cast<int>(build.tag_vocabulary_size);
        head.parameters["units"] = std::to_string(build.tag_vocabulary_size);
    }
}

} // namespace cyxwiz
