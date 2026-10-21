#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

class SequentialModel;

struct CausalLmBatch {
    size_t batch_size = 0;
    size_t sequence_length = 0;
    std::vector<int> input_ids;
    std::vector<int> target_ids;
    std::vector<int> attention_mask;
};

struct TokenCrossEntropyResult {
    float loss = 0.0f;
    size_t valid_token_count = 0;
};

struct InstructionTextRecord {
    std::string prompt;
    std::string response;
    std::string system;
};

struct InstructionTextFormatOptions {
    std::string system_prefix = "System: ";
    std::string prompt_prefix = "User: ";
    std::string response_prefix = "Assistant: ";
    std::string separator = "\n";
    bool append_eos_text = true;
    std::string eos_text = "[EOS]";
};

struct InstructionRecordColumnMapping {
    std::string prompt_column = "prompt";
    std::string response_column = "response";
    std::string system_column;
    bool skip_empty_records = false;
};

CausalLmBatch BuildCausalLmBatch(
    const std::vector<std::vector<int>>& token_sequences,
    size_t sequence_length,
    int pad_token_id,
    int ignore_index);

std::vector<std::vector<int>> PackTokenStreamForCausalLm(
    const std::vector<int>& token_stream,
    size_t window_length,
    size_t stride);

std::string FormatInstructionTextRecord(
    const InstructionTextRecord& record,
    const InstructionTextFormatOptions& options = {});

std::vector<std::string> FormatInstructionTextRecords(
    const std::vector<InstructionTextRecord>& records,
    const InstructionTextFormatOptions& options = {});

std::vector<InstructionTextRecord> BuildInstructionRecordsFromRows(
    const std::vector<std::map<std::string, std::string>>& rows,
    const InstructionRecordColumnMapping& mapping = {});

std::vector<std::vector<int>> PackInstructionRecordsForCausalLm(
    const std::vector<InstructionTextRecord>& records,
    const std::function<std::vector<int>(const std::string&)>& encode,
    size_t window_length,
    size_t stride,
    const InstructionTextFormatOptions& options = {});

std::vector<float> BuildCausalAttentionMask(
    size_t sequence_length,
    float visible_value = 0.0f,
    float masked_value = -1.0e9f);

std::vector<float> BuildBatchedCausalKeyPaddingMask(
    const std::vector<int>& attention_mask,
    size_t batch_size,
    size_t sequence_length,
    float visible_value = 0.0f,
    float masked_value = -1.0e9f);

std::vector<float> ScaledDotProductAttentionCpu(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>& additive_mask,
    size_t sequence_length,
    size_t head_dim);

std::vector<float> ScaledDotProductAttentionBatchedCpu(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>& additive_mask,
    size_t batch_size,
    size_t sequence_length,
    size_t head_dim);

std::vector<float> ScaledDotProductAttentionMultiHeadCpu(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>& additive_mask,
    size_t batch_size,
    size_t num_heads,
    size_t sequence_length,
    size_t head_dim);

TokenCrossEntropyResult TokenCrossEntropyLossCpu(
    const std::vector<float>& logits,
    const std::vector<int>& target_ids,
    size_t batch_size,
    size_t sequence_length,
    size_t vocab_size,
    int ignore_index);

std::vector<float> TokenCrossEntropyGradientCpu(
    const std::vector<float>& logits,
    const std::vector<int>& target_ids,
    size_t batch_size,
    size_t sequence_length,
    size_t vocab_size,
    int ignore_index);

int64_t GreedyNextTokenFromLogits(
    const std::vector<float>& logits,
    size_t batch_size,
    size_t sequence_length,
    size_t vocab_size,
    size_t batch_index = 0);

std::vector<int64_t> GenerateGreedyTokenIds(
    SequentialModel& model,
    const std::vector<int64_t>& prompt_ids,
    size_t max_new_tokens,
    int64_t eos_token_id = -1);

// Prefer GenerateTokenIdsWithConfig from language_model_generation.h for new
// code that needs temperature/top-k/top-p controls.

} // namespace cyxwiz
