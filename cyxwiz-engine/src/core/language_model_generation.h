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

struct LanguageModelGenerationConfig {
    size_t max_new_tokens = 32;
    float temperature = 1.0f;
    size_t top_k = 0;
    float top_p = 1.0f;
    int64_t eos_token_id = -1;
    LanguageModelSamplingMode sampling_mode = LanguageModelSamplingMode::Greedy;
    bool include_prompt = true;
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

std::vector<int64_t> GenerateTokenIdsWithConfig(
    SequentialModel& model,
    const std::vector<int64_t>& prompt_ids,
    const LanguageModelGenerationConfig& config,
    uint32_t seed = 5489u);

} // namespace cyxwiz
