#pragma once

#include <cstddef>
#include <functional>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace cyxwiz {

struct TrainingGenerationPreviewSettings;
class IExecutableModel;
class SequentialModel;
struct LanguageModelGenerationStep;

enum class LanguageModelSamplingMode {
    Greedy,
    Multinomial
};

enum class LanguageModelGenerationStopReason {
    MaxTokens,
    EosToken,
    Error,
    UserCancelled,
    StopText  // The caller's stop check matched, for example a stop text in the decoded output.
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
    std::function<bool()> should_cancel; // Called between tokens; no callback preserves legacy behavior.
    // Called after each new token with all new tokens so far; true ends
    // generation with StopReason::StopText (the caller decodes and matches).
    std::function<bool(const std::vector<int64_t>&)> stop_check;
    // Divides positive logits (multiplies negative ones) of tokens already in
    // the prompt or output by this factor; 1 = off.
    float repetition_penalty = 1.0f;
    // Called after each new token, before the stop checks (streaming).
    std::function<void(const LanguageModelGenerationStep&)> on_step;
    // Keeps candidates whose probability is at least min_p times the most
    // likely one's (after temperature and top-k); 0 = off.
    float min_p = 0.0f;
    // Subtracted from the logit of each token already generated: frequency
    // times its count, presence once; 0 = off.
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    // A token that would repeat an n-gram of this size already in the prompt
    // or output is never chosen; 0 = off.
    size_t no_repeat_ngram_size = 0;
    // Beam search with this many beams when above 1 (deterministic; scores are
    // summed log-probabilities divided by length^length_penalty).
    size_t num_beams = 1;
    float length_penalty = 1.0f;
    // Layer stacks with a KV cache: the token IDs the model's cache holds from
    // the previous call. When the new prompt starts with them, only the rest
    // is fed (a chat's earlier turns are not recomputed), and on return it
    // holds what the cache now covers. Nullptr: the cache is emptied at the
    // start and released at the end.
    std::vector<int64_t>* reuse_cache = nullptr;
    // Reuse attention keys/values between tokens when every module supports
    // incremental decoding (same tokens as full recomputation, much faster).
    bool use_kv_cache = true;
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
    bool used_kv_cache = false;  // true when the KV-cached path produced the tokens
    size_t reused_cache_tokens = 0;  // prompt tokens taken from the previous call's cache
};

// The decoding settings (greedy, sampling or beam search, and the
// penalties) of generation previews and the LM Generation panel.
void ApplyGenerationDecoding(const TrainingGenerationPreviewSettings& settings,
                             LanguageModelGenerationConfig& config);

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

// Any executable model whose Forward maps Int64 [1, seq] token IDs to Float32
// [1, seq, vocab] logits: a layer stack (KV-cached when it supports it) or a
// graph with branches (each token recomputes the whole sequence).
LanguageModelGenerationResult GenerateTokenIdsWithReport(
    IExecutableModel& model,
    const std::vector<int64_t>& prompt_ids,
    const LanguageModelGenerationConfig& config,
    uint32_t seed = 5489u);

std::vector<int64_t> GenerateTokenIdsWithConfig(
    SequentialModel& model,
    const std::vector<int64_t>& prompt_ids,
    const LanguageModelGenerationConfig& config,
    uint32_t seed = 5489u);

} // namespace cyxwiz
