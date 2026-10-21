#include "language_model_training.h"

#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace cyxwiz {

CausalLmBatch BuildCausalLmBatch(
    const std::vector<std::vector<int>>& token_sequences,
    size_t sequence_length,
    int pad_token_id,
    int ignore_index) {

    if (token_sequences.empty()) {
        throw std::invalid_argument(
            "BuildCausalLmBatch requires at least one token sequence");
    }
    if (sequence_length == 0) {
        throw std::invalid_argument(
            "BuildCausalLmBatch requires sequence_length > 0");
    }

    CausalLmBatch batch;
    batch.batch_size = token_sequences.size();
    batch.sequence_length = sequence_length;

    const size_t total = batch.batch_size * batch.sequence_length;
    batch.input_ids.assign(total, pad_token_id);
    batch.target_ids.assign(total, ignore_index);
    batch.attention_mask.assign(total, 0);

    for (size_t row = 0; row < token_sequences.size(); ++row) {
        const auto& tokens = token_sequences[row];
        for (size_t t = 0; t < sequence_length; ++t) {
            const size_t offset = row * sequence_length + t;

            if (t < tokens.size()) {
                const int input_id = tokens[t];
                batch.input_ids[offset] = input_id;
                batch.attention_mask[offset] =
                    input_id == pad_token_id ? 0 : 1;
            }

            const size_t target_pos = t + 1;
            if (t < tokens.size() &&
                tokens[t] != pad_token_id &&
                target_pos < tokens.size() &&
                tokens[target_pos] != pad_token_id) {
                batch.target_ids[offset] = tokens[target_pos];
            }
        }
    }

    return batch;
}

std::vector<std::vector<int>> PackTokenStreamForCausalLm(
    const std::vector<int>& token_stream,
    size_t window_length,
    size_t stride) {

    if (window_length < 2) {
        throw std::invalid_argument(
            "PackTokenStreamForCausalLm requires window_length >= 2");
    }
    if (stride == 0) {
        throw std::invalid_argument(
            "PackTokenStreamForCausalLm requires stride > 0");
    }

    std::vector<std::vector<int>> windows;
    if (token_stream.size() < window_length) {
        return windows;
    }

    for (size_t start = 0;
         start + window_length <= token_stream.size();
         start += stride) {
        windows.emplace_back(
            token_stream.begin() + static_cast<std::ptrdiff_t>(start),
            token_stream.begin() +
                static_cast<std::ptrdiff_t>(start + window_length));
    }

    return windows;
}

std::string FormatInstructionTextRecord(
    const InstructionTextRecord& record,
    const InstructionTextFormatOptions& options) {

    if (record.prompt.empty()) {
        throw std::invalid_argument(
            "FormatInstructionTextRecord requires a non-empty prompt");
    }
    if (record.response.empty()) {
        throw std::invalid_argument(
            "FormatInstructionTextRecord requires a non-empty response");
    }
    if (options.separator.empty()) {
        throw std::invalid_argument(
            "FormatInstructionTextRecord requires a non-empty separator");
    }

    std::string text;
    if (!record.system.empty()) {
        text += options.system_prefix;
        text += record.system;
        text += options.separator;
    }
    text += options.prompt_prefix;
    text += record.prompt;
    text += options.separator;
    text += options.response_prefix;
    text += record.response;
    if (options.append_eos_text && !options.eos_text.empty()) {
        text += options.separator;
        text += options.eos_text;
    }
    return text;
}

std::vector<std::string> FormatInstructionTextRecords(
    const std::vector<InstructionTextRecord>& records,
    const InstructionTextFormatOptions& options) {

    if (records.empty()) {
        throw std::invalid_argument(
            "FormatInstructionTextRecords requires at least one record");
    }

    std::vector<std::string> formatted;
    formatted.reserve(records.size());
    for (const auto& record : records) {
        formatted.push_back(FormatInstructionTextRecord(record, options));
    }
    return formatted;
}

std::vector<InstructionTextRecord> BuildInstructionRecordsFromRows(
    const std::vector<std::map<std::string, std::string>>& rows,
    const InstructionRecordColumnMapping& mapping) {

    if (rows.empty()) {
        throw std::invalid_argument(
            "BuildInstructionRecordsFromRows requires at least one row");
    }
    if (mapping.prompt_column.empty()) {
        throw std::invalid_argument(
            "BuildInstructionRecordsFromRows requires a prompt column");
    }
    if (mapping.response_column.empty()) {
        throw std::invalid_argument(
            "BuildInstructionRecordsFromRows requires a response column");
    }

    std::vector<InstructionTextRecord> records;
    records.reserve(rows.size());
    for (size_t row_index = 0; row_index < rows.size(); ++row_index) {
        const auto& row = rows[row_index];
        const auto prompt_it = row.find(mapping.prompt_column);
        const auto response_it = row.find(mapping.response_column);
        if (prompt_it == row.end()) {
            throw std::invalid_argument(
                "BuildInstructionRecordsFromRows missing prompt column");
        }
        if (response_it == row.end()) {
            throw std::invalid_argument(
                "BuildInstructionRecordsFromRows missing response column");
        }

        InstructionTextRecord record;
        record.prompt = prompt_it->second;
        record.response = response_it->second;
        if (!mapping.system_column.empty()) {
            const auto system_it = row.find(mapping.system_column);
            if (system_it == row.end()) {
                throw std::invalid_argument(
                    "BuildInstructionRecordsFromRows missing system column");
            }
            record.system = system_it->second;
        }

        if (record.prompt.empty() || record.response.empty()) {
            if (mapping.skip_empty_records) {
                continue;
            }
            throw std::invalid_argument(
                "BuildInstructionRecordsFromRows requires non-empty prompt and response values");
        }

        records.push_back(std::move(record));
    }

    if (records.empty()) {
        throw std::invalid_argument(
            "BuildInstructionRecordsFromRows produced no instruction records");
    }
    return records;
}

std::vector<std::vector<int>> PackInstructionRecordsForCausalLm(
    const std::vector<InstructionTextRecord>& records,
    const std::function<std::vector<int>(const std::string&)>& encode,
    size_t window_length,
    size_t stride,
    const InstructionTextFormatOptions& options) {

    if (!encode) {
        throw std::invalid_argument(
            "PackInstructionRecordsForCausalLm requires an encoder");
    }

    const auto formatted = FormatInstructionTextRecords(records, options);
    std::vector<std::vector<int>> windows;
    for (const auto& text : formatted) {
        const auto token_ids = encode(text);
        const auto record_windows =
            PackTokenStreamForCausalLm(token_ids, window_length, stride);
        windows.insert(windows.end(),
                       record_windows.begin(),
                       record_windows.end());
    }
    return windows;
}

std::vector<float> BuildCausalAttentionMask(
    size_t sequence_length,
    float visible_value,
    float masked_value) {

    if (sequence_length == 0) {
        throw std::invalid_argument(
            "BuildCausalAttentionMask requires sequence_length > 0");
    }

    std::vector<float> mask(sequence_length * sequence_length, masked_value);
    for (size_t row = 0; row < sequence_length; ++row) {
        for (size_t col = 0; col <= row; ++col) {
            mask[row * sequence_length + col] = visible_value;
        }
    }

    return mask;
}

std::vector<float> BuildBatchedCausalKeyPaddingMask(
    const std::vector<int>& attention_mask,
    size_t batch_size,
    size_t sequence_length,
    float visible_value,
    float masked_value) {

    if (batch_size == 0) {
        throw std::invalid_argument(
            "BuildBatchedCausalKeyPaddingMask requires batch_size > 0");
    }
    if (sequence_length == 0) {
        throw std::invalid_argument(
            "BuildBatchedCausalKeyPaddingMask requires sequence_length > 0");
    }
    if (attention_mask.size() != batch_size * sequence_length) {
        throw std::invalid_argument(
            "BuildBatchedCausalKeyPaddingMask attention_mask shape mismatch");
    }

    std::vector<float> mask(
        batch_size * sequence_length * sequence_length,
        masked_value);

    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t query = 0; query < sequence_length; ++query) {
            for (size_t key = 0; key <= query; ++key) {
                const size_t key_mask_offset = batch * sequence_length + key;
                if (attention_mask[key_mask_offset] != 0) {
                    const size_t mask_offset =
                        (batch * sequence_length * sequence_length) +
                        (query * sequence_length) +
                        key;
                    mask[mask_offset] = visible_value;
                }
            }
        }
    }

    return mask;
}

std::vector<float> ScaledDotProductAttentionCpu(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>& additive_mask,
    size_t sequence_length,
    size_t head_dim) {

    if (sequence_length == 0) {
        throw std::invalid_argument(
            "ScaledDotProductAttentionCpu requires sequence_length > 0");
    }
    if (head_dim == 0) {
        throw std::invalid_argument(
            "ScaledDotProductAttentionCpu requires head_dim > 0");
    }

    const size_t matrix_size = sequence_length * head_dim;
    const size_t mask_size = sequence_length * sequence_length;
    if (query.size() != matrix_size ||
        key.size() != matrix_size ||
        value.size() != matrix_size) {
        throw std::invalid_argument(
            "ScaledDotProductAttentionCpu query/key/value shape mismatch");
    }
    if (!additive_mask.empty() && additive_mask.size() != mask_size) {
        throw std::invalid_argument(
            "ScaledDotProductAttentionCpu additive mask shape mismatch");
    }

    std::vector<float> output(matrix_size, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (size_t query_pos = 0; query_pos < sequence_length; ++query_pos) {
        std::vector<float> scores(sequence_length, 0.0f);
        float max_score = -std::numeric_limits<float>::infinity();

        for (size_t key_pos = 0; key_pos < sequence_length; ++key_pos) {
            float dot = 0.0f;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                dot += query[query_pos * head_dim + dim] *
                       key[key_pos * head_dim + dim];
            }

            float score = dot * scale;
            if (!additive_mask.empty()) {
                score += additive_mask[query_pos * sequence_length + key_pos];
            }
            scores[key_pos] = score;
            if (score > max_score) {
                max_score = score;
            }
        }

        float denom = 0.0f;
        for (float& score : scores) {
            score = std::exp(score - max_score);
            denom += score;
        }

        if (denom == 0.0f) {
            continue;
        }

        for (size_t key_pos = 0; key_pos < sequence_length; ++key_pos) {
            const float weight = scores[key_pos] / denom;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                output[query_pos * head_dim + dim] +=
                    weight * value[key_pos * head_dim + dim];
            }
        }
    }

    return output;
}

std::vector<float> ScaledDotProductAttentionBatchedCpu(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>& additive_mask,
    size_t batch_size,
    size_t sequence_length,
    size_t head_dim) {

    if (batch_size == 0) {
        throw std::invalid_argument(
            "ScaledDotProductAttentionBatchedCpu requires batch_size > 0");
    }
    if (sequence_length == 0) {
        throw std::invalid_argument(
            "ScaledDotProductAttentionBatchedCpu requires sequence_length > 0");
    }
    if (head_dim == 0) {
        throw std::invalid_argument(
            "ScaledDotProductAttentionBatchedCpu requires head_dim > 0");
    }

    const size_t batch_matrix_size = sequence_length * head_dim;
    const size_t total_matrix_size = batch_size * batch_matrix_size;
    const size_t batch_mask_size = sequence_length * sequence_length;
    const size_t total_mask_size = batch_size * batch_mask_size;

    if (query.size() != total_matrix_size ||
        key.size() != total_matrix_size ||
        value.size() != total_matrix_size) {
        throw std::invalid_argument(
            "ScaledDotProductAttentionBatchedCpu query/key/value shape mismatch");
    }
    if (!additive_mask.empty() && additive_mask.size() != total_mask_size) {
        throw std::invalid_argument(
            "ScaledDotProductAttentionBatchedCpu additive mask shape mismatch");
    }

    std::vector<float> output(total_matrix_size, 0.0f);
    for (size_t batch = 0; batch < batch_size; ++batch) {
        const size_t matrix_offset = batch * batch_matrix_size;
        const size_t mask_offset = batch * batch_mask_size;

        std::vector<float> batch_query(
            query.begin() + static_cast<std::ptrdiff_t>(matrix_offset),
            query.begin() +
                static_cast<std::ptrdiff_t>(matrix_offset + batch_matrix_size));
        std::vector<float> batch_key(
            key.begin() + static_cast<std::ptrdiff_t>(matrix_offset),
            key.begin() +
                static_cast<std::ptrdiff_t>(matrix_offset + batch_matrix_size));
        std::vector<float> batch_value(
            value.begin() + static_cast<std::ptrdiff_t>(matrix_offset),
            value.begin() +
                static_cast<std::ptrdiff_t>(matrix_offset + batch_matrix_size));

        std::vector<float> batch_mask;
        if (!additive_mask.empty()) {
            batch_mask.assign(
                additive_mask.begin() + static_cast<std::ptrdiff_t>(mask_offset),
                additive_mask.begin() +
                    static_cast<std::ptrdiff_t>(mask_offset + batch_mask_size));
        }

        const auto batch_output = ScaledDotProductAttentionCpu(
            batch_query,
            batch_key,
            batch_value,
            batch_mask,
            sequence_length,
            head_dim);

        std::copy(batch_output.begin(),
                  batch_output.end(),
                  output.begin() + static_cast<std::ptrdiff_t>(matrix_offset));
    }

    return output;
}

std::vector<float> ScaledDotProductAttentionMultiHeadCpu(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>& additive_mask,
    size_t batch_size,
    size_t num_heads,
    size_t sequence_length,
    size_t head_dim) {

    if (batch_size == 0) {
        throw std::invalid_argument(
            "ScaledDotProductAttentionMultiHeadCpu requires batch_size > 0");
    }
    if (num_heads == 0) {
        throw std::invalid_argument(
            "ScaledDotProductAttentionMultiHeadCpu requires num_heads > 0");
    }

    const size_t batch_heads = batch_size * num_heads;
    const size_t matrix_size =
        batch_heads * sequence_length * head_dim;
    const size_t mask_size =
        batch_heads * sequence_length * sequence_length;

    if (query.size() != matrix_size ||
        key.size() != matrix_size ||
        value.size() != matrix_size) {
        throw std::invalid_argument(
            "ScaledDotProductAttentionMultiHeadCpu query/key/value shape mismatch");
    }
    if (!additive_mask.empty() && additive_mask.size() != mask_size) {
        throw std::invalid_argument(
            "ScaledDotProductAttentionMultiHeadCpu additive mask shape mismatch");
    }

    return ScaledDotProductAttentionBatchedCpu(
        query,
        key,
        value,
        additive_mask,
        batch_heads,
        sequence_length,
        head_dim);
}

TokenCrossEntropyResult TokenCrossEntropyLossCpu(
    const std::vector<float>& logits,
    const std::vector<int>& target_ids,
    size_t batch_size,
    size_t sequence_length,
    size_t vocab_size,
    int ignore_index) {

    if (batch_size == 0) {
        throw std::invalid_argument(
            "TokenCrossEntropyLossCpu requires batch_size > 0");
    }
    if (sequence_length == 0) {
        throw std::invalid_argument(
            "TokenCrossEntropyLossCpu requires sequence_length > 0");
    }
    if (vocab_size == 0) {
        throw std::invalid_argument(
            "TokenCrossEntropyLossCpu requires vocab_size > 0");
    }

    const size_t token_count = batch_size * sequence_length;
    const size_t logits_size = token_count * vocab_size;
    if (logits.size() != logits_size || target_ids.size() != token_count) {
        throw std::invalid_argument(
            "TokenCrossEntropyLossCpu logits/target shape mismatch");
    }

    double loss_sum = 0.0;
    size_t valid_count = 0;

    for (size_t token = 0; token < token_count; ++token) {
        const int target = target_ids[token];
        if (target == ignore_index) {
            continue;
        }
        if (target < 0 || static_cast<size_t>(target) >= vocab_size) {
            throw std::invalid_argument(
                "TokenCrossEntropyLossCpu target id out of range");
        }

        const size_t offset = token * vocab_size;
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t vocab = 0; vocab < vocab_size; ++vocab) {
            max_logit = std::max(max_logit, logits[offset + vocab]);
        }

        double exp_sum = 0.0;
        for (size_t vocab = 0; vocab < vocab_size; ++vocab) {
            exp_sum += std::exp(static_cast<double>(logits[offset + vocab] - max_logit));
        }

        const double log_sum_exp = static_cast<double>(max_logit) + std::log(exp_sum);
        loss_sum += log_sum_exp - static_cast<double>(logits[offset + static_cast<size_t>(target)]);
        ++valid_count;
    }

    if (valid_count == 0) {
        throw std::invalid_argument(
            "TokenCrossEntropyLossCpu requires at least one non-ignored target");
    }

    TokenCrossEntropyResult result;
    result.loss = static_cast<float>(loss_sum / static_cast<double>(valid_count));
    result.valid_token_count = valid_count;
    return result;
}

std::vector<float> TokenCrossEntropyGradientCpu(
    const std::vector<float>& logits,
    const std::vector<int>& target_ids,
    size_t batch_size,
    size_t sequence_length,
    size_t vocab_size,
    int ignore_index) {

    const auto loss = TokenCrossEntropyLossCpu(
        logits,
        target_ids,
        batch_size,
        sequence_length,
        vocab_size,
        ignore_index);

    const size_t token_count = batch_size * sequence_length;
    std::vector<float> gradient(logits.size(), 0.0f);
    const float normalizer = 1.0f / static_cast<float>(loss.valid_token_count);

    for (size_t token = 0; token < token_count; ++token) {
        const int target = target_ids[token];
        if (target == ignore_index) {
            continue;
        }

        const size_t offset = token * vocab_size;
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t vocab = 0; vocab < vocab_size; ++vocab) {
            max_logit = std::max(max_logit, logits[offset + vocab]);
        }

        double exp_sum = 0.0;
        for (size_t vocab = 0; vocab < vocab_size; ++vocab) {
            exp_sum += std::exp(static_cast<double>(logits[offset + vocab] - max_logit));
        }

        for (size_t vocab = 0; vocab < vocab_size; ++vocab) {
            const float probability = static_cast<float>(
                std::exp(static_cast<double>(logits[offset + vocab] - max_logit)) /
                exp_sum);
            gradient[offset + vocab] = probability * normalizer;
        }
        gradient[offset + static_cast<size_t>(target)] -= normalizer;
    }

    return gradient;
}

int64_t GreedyNextTokenFromLogits(
    const std::vector<float>& logits,
    size_t batch_size,
    size_t sequence_length,
    size_t vocab_size,
    size_t batch_index) {

    if (batch_size == 0) {
        throw std::invalid_argument(
            "GreedyNextTokenFromLogits requires batch_size > 0");
    }
    if (sequence_length == 0) {
        throw std::invalid_argument(
            "GreedyNextTokenFromLogits requires sequence_length > 0");
    }
    if (vocab_size == 0) {
        throw std::invalid_argument(
            "GreedyNextTokenFromLogits requires vocab_size > 0");
    }
    if (batch_index >= batch_size) {
        throw std::invalid_argument(
            "GreedyNextTokenFromLogits batch_index out of range");
    }

    const size_t expected_size = batch_size * sequence_length * vocab_size;
    if (logits.size() != expected_size) {
        throw std::invalid_argument(
            "GreedyNextTokenFromLogits logits shape mismatch");
    }

    const size_t offset =
        ((batch_index * sequence_length) + (sequence_length - 1)) * vocab_size;
    size_t best_token = 0;
    float best_logit = logits[offset];
    for (size_t token = 1; token < vocab_size; ++token) {
        const float value = logits[offset + token];
        if (value > best_logit) {
            best_logit = value;
            best_token = token;
        }
    }

    return static_cast<int64_t>(best_token);
}

std::vector<int64_t> GenerateGreedyTokenIds(
    SequentialModel& model,
    const std::vector<int64_t>& prompt_ids,
    size_t max_new_tokens,
    int64_t eos_token_id) {

    if (prompt_ids.empty()) {
        throw std::invalid_argument(
            "GenerateGreedyTokenIds requires at least one prompt token");
    }

    std::vector<int64_t> generated = prompt_ids;
    for (size_t step = 0; step < max_new_tokens; ++step) {
        Tensor input({1, generated.size()}, generated.data(), DataType::Int64);
        Tensor logits = model.Forward(input);
        const auto& shape = logits.Shape();
        if (shape.size() != 3 || shape[0] != 1 ||
            shape[1] != generated.size() || shape[2] == 0 ||
            logits.GetDataType() != DataType::Float32) {
            throw std::invalid_argument(
                "GenerateGreedyTokenIds model must return Float32 [1, seq, vocab] logits");
        }

        const float* data = logits.ReadData<float>();
        const std::vector<float> logits_values(data, data + logits.NumElements());
        const int64_t next_token = GreedyNextTokenFromLogits(
            logits_values,
            shape[0],
            shape[1],
            shape[2],
            0);
        generated.push_back(next_token);

        if (eos_token_id >= 0 && next_token == eos_token_id) {
            break;
        }
    }

    return generated;
}

} // namespace cyxwiz
