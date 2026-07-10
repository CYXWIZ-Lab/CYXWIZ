#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace cyxwiz {

class SequentialModel;

enum class LanguageModelSamplingMode {
    Greedy,
    Multinomial
};

enum class LanguageModelGenerationStopReason {
    MaxTokens,
    EosToken,
    Error,
    UserCancelled
};

std::string LanguageModelGenerationStopReasonName(
    LanguageModelGenerationStopReason reason);

struct LanguageModelGenerationConfig {
    size_t max_new_tokens = 32;
    float temperature = 1.0f;
    size_t top_k = 0;
    float top_p = 1.0f;
    int64_t eos_token_id = -1;
    LanguageModelSamplingMode sampling_mode = LanguageModelSamplingMode::Greedy;
    bool include_prompt = true;
    size_t max_context_tokens = 0;
};

struct NextTokenCandidate {
    int64_t token_id = 0;
    float probability = 0.0f;
};

struct NextTokenSelection {
    int64_t token_id = 0;
    float probability = 0.0f;
    std::vector<NextTokenCandidate> candidates;
};

struct LanguageModelGenerationStep {
    size_t step_index = 0;
    size_t input_length = 0;
    int64_t token_id = 0;
    float probability = 0.0f;
    std::vector<NextTokenCandidate> candidates;
};

struct LanguageModelGenerationResult {
    std::vector<int64_t> token_ids;
    std::vector<int64_t> new_token_ids;
    std::vector<LanguageModelGenerationStep> steps;
    LanguageModelGenerationStopReason stop_reason =
        LanguageModelGenerationStopReason::MaxTokens;
    size_t prompt_length = 0;
    size_t max_new_tokens = 0;
    size_t remaining_budget = 0;
    bool include_prompt = true;
};

std::vector<std::string> ValidateLanguageModelGenerationConfig(
    const LanguageModelGenerationConfig& config);

void RequireValidLanguageModelGenerationConfig(
    const LanguageModelGenerationConfig& config);

std::vector<NextTokenCandidate> BuildNextTokenDistribution(
    const std::vector<float>& logits,
    size_t batch_size,
    size_t sequence_length,
    size_t vocab_size,
    const LanguageModelGenerationConfig& config,
    size_t batch_index = 0);

NextTokenSelection SelectNextTokenFromDistribution(
    const std::vector<NextTokenCandidate>& candidates,
    const LanguageModelGenerationConfig& config,
    std::mt19937& rng);

NextTokenSelection SelectNextTokenFromLogits(
    const std::vector<float>& logits,
    size_t batch_size,
    size_t sequence_length,
    size_t vocab_size,
    const LanguageModelGenerationConfig& config,
    std::mt19937& rng,
    size_t batch_index = 0);

LanguageModelGenerationResult GenerateTokenIdsWithReport(
    SequentialModel& model,
    const std::vector<int64_t>& prompt_ids,
    const LanguageModelGenerationConfig& config,
    uint32_t seed = 5489u);

std::vector<int64_t> GenerateTokenIdsWithConfig(
    SequentialModel& model,
    const std::vector<int64_t>& prompt_ids,
    const LanguageModelGenerationConfig& config,
    uint32_t seed = 5489u);

} // namespace cyxwiz
