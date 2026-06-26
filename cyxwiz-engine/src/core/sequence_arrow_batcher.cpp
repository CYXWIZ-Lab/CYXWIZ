#include "sequence_arrow_batcher.h"

#include "arrow_dataset.h"
#include "ner_sequence_builder.h"
#include "sequence_model_input.h"

#include <arrow/api.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <sstream>
#include <unordered_map>

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

bool ReadAnyCell(const std::shared_ptr<arrow::Table>& table,
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

void SplitByRatios(size_t total_units,
                   float train_ratio,
                   float val_ratio,
                   std::vector<size_t>& train_units,
                   std::vector<size_t>& val_units,
                   std::vector<size_t>& test_units) {
    size_t train_count = static_cast<size_t>(std::floor(total_units * train_ratio));
    size_t val_count = static_cast<size_t>(std::floor(total_units * val_ratio));
    if (train_count > total_units) {
        train_count = total_units;
    }
    if (train_count + val_count > total_units) {
        val_count = (train_count >= total_units) ? 0 : (total_units - train_count);
    }
    const size_t test_count = total_units - train_count - val_count;

    train_units.clear();
    val_units.clear();
    test_units.clear();
    train_units.reserve(train_count);
    val_units.reserve(val_count);
    test_units.reserve(test_count);

    for (size_t i = 0; i < train_count; ++i) {
        train_units.push_back(i);
    }
    for (size_t i = train_count; i < train_count + val_count; ++i) {
        val_units.push_back(i);
    }
    for (size_t i = train_count + val_count; i < total_units; ++i) {
        test_units.push_back(i);
    }
}

void AppendGroupsToFlat(const std::vector<std::vector<size_t>>& groups,
                       const std::vector<size_t>& units,
                       std::vector<size_t>& flat) {
    for (const auto unit : units) {
        if (unit >= groups.size()) {
            continue;
        }
        flat.insert(flat.end(), groups[unit].begin(), groups[unit].end());
    }
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
    int sentence_index = -1;
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

    if (config.has_data_split &&
        !config.sequence_batch.sentence_id_column.empty()) {
        sentence_index = table->schema()->GetFieldIndex(
            config.sequence_batch.sentence_id_column);
        if (sentence_index < 0) {
            result.error_message =
                "sequence sentence id column '" +
                config.sequence_batch.sentence_id_column + "' not found";
            return result;
        }
    }

    std::vector<std::vector<size_t>> sentence_groups;
    if (sentence_index >= 0) {
        sentence_groups.reserve(static_cast<size_t>(table->num_rows()));
        std::unordered_map<std::string, size_t> sentence_id_to_group;
        for (int64_t row_index = 0; row_index < table->num_rows(); ++row_index) {
            std::string sentence_id;
            if (!ReadAnyCell(table, sentence_index, row_index, sentence_id, error)) {
                result.error_message =
                    "sequence sentence id row " + std::to_string(row_index) +
                    ": " + error;
                return result;
            }
            if (sentence_id.empty()) {
                sentence_id = "__row__" + std::to_string(row_index);
            }

            auto it = sentence_id_to_group.find(sentence_id);
            size_t group_index = it != sentence_id_to_group.end()
                ? it->second
                : sentence_groups.size();
            if (it == sentence_id_to_group.end()) {
                sentence_id_to_group[sentence_id] = group_index;
                sentence_groups.push_back({});
            }
            sentence_groups[group_index].push_back(
                static_cast<size_t>(row_index));
        }
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
        if (config.has_data_split) {
            std::vector<size_t> train_units;
            std::vector<size_t> val_units;
            std::vector<size_t> test_units;

            if (sentence_index >= 0 && !sentence_groups.empty()) {
                SplitByRatios(sentence_groups.size(),
                              config.train_ratio,
                              config.val_ratio,
                              train_units,
                              val_units,
                              test_units);

                AppendGroupsToFlat(sentence_groups, train_units,
                                   builder_config.batcher.train_indices);
                AppendGroupsToFlat(sentence_groups, val_units,
                                   builder_config.batcher.val_indices);
                AppendGroupsToFlat(sentence_groups, test_units,
                                   builder_config.batcher.test_indices);
            } else {
                SplitByRatios(static_cast<size_t>(rows.size()),
                              config.train_ratio,
                              config.val_ratio,
                              train_units,
                              val_units,
                              test_units);

                builder_config.batcher.train_indices = std::move(train_units);
                builder_config.batcher.val_indices = std::move(val_units);
                builder_config.batcher.test_indices = std::move(test_units);
            }
        }

        auto build = BuildNERSequenceData(rows, builder_config);
        result.id_to_label = build.tag_vocabulary.Values();
        result.sample_count = build.samples.size();
        result.sequence_length = ResolveSequenceLength(
            build.samples, build.batcher_config.max_sequence_length);
        result.token_vocabulary_size = build.token_vocabulary.Size();
        result.pos_vocabulary_size = build.pos_vocabulary.Size();
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
        if (IsSequenceFeatureFusionLayer(layer)) {
            if (build.token_vocabulary_size > 0) {
                layer.parameters["word_num_embeddings"] =
                    std::to_string(build.token_vocabulary_size);
            }
            if (build.pos_vocabulary_size > 0) {
                layer.parameters["pos_num_embeddings"] =
                    std::to_string(build.pos_vocabulary_size);
            }
        }
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
