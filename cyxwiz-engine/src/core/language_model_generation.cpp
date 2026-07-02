#include "language_model_generation.h"

#include <cyxwiz/sequential.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cyxwiz {

std::vector<std::string> ValidateLanguageModelGenerationConfig(
    const LanguageModelGenerationConfig& config) {
    std::vector<std::string> issues;
    if (config.max_new_tokens == 0) {
        issues.emplace_back("max_new_tokens must be greater than 0");
    }
    if (!std::isfinite(config.temperature) || config.temperature <= 0.0f) {
        issues.emplace_back("temperature must be finite and greater than 0");
    }
    if (!std::isfinite(config.top_p) ||
        config.top_p <= 0.0f ||
        config.top_p > 1.0f) {
        issues.emplace_back("top_p must be finite and in the range (0, 1]");
    }
    return issues;
}

void RequireValidLanguageModelGenerationConfig(
    const LanguageModelGenerationConfig& config) {
    const auto issues = ValidateLanguageModelGenerationConfig(config);
    if (!issues.empty()) {
        std::string message = "invalid language-model generation config:";
        for (const auto& issue : issues) {
            message += " " + issue + ";";
        }
        throw std::invalid_argument(message);
    }
}

std::vector<NextTokenCandidate> BuildNextTokenDistribution(
    const std::vector<float>& logits,
    size_t batch_size,
    size_t sequence_length,
    size_t vocab_size,
    const LanguageModelGenerationConfig& config,
    size_t batch_index) {

    RequireValidLanguageModelGenerationConfig(config);
    if (batch_size == 0) {
        throw std::invalid_argument(
            "BuildNextTokenDistribution requires batch_size > 0");
    }
    if (sequence_length == 0) {
        throw std::invalid_argument(
            "BuildNextTokenDistribution requires sequence_length > 0");
    }
    if (vocab_size == 0) {
        throw std::invalid_argument(
            "BuildNextTokenDistribution requires vocab_size > 0");
    }
    if (batch_index >= batch_size) {
        throw std::invalid_argument(
            "BuildNextTokenDistribution batch_index out of range");
    }

    const size_t expected_size = batch_size * sequence_length * vocab_size;
    if (logits.size() != expected_size) {
        throw std::invalid_argument(
            "BuildNextTokenDistribution logits shape mismatch");
    }

    const size_t offset =
        ((batch_index * sequence_length) + (sequence_length - 1)) * vocab_size;

    std::vector<NextTokenCandidate> candidates;
    candidates.reserve(vocab_size);

    float max_scaled = -std::numeric_limits<float>::infinity();
    for (size_t token = 0; token < vocab_size; ++token) {
        const float scaled = logits[offset + token] / config.temperature;
        max_scaled = std::max(max_scaled, scaled);
        candidates.push_back({static_cast<int64_t>(token), scaled});
    }

    double exp_sum = 0.0;
    for (auto& candidate : candidates) {
        const double value =
            std::exp(static_cast<double>(candidate.probability - max_scaled));
        candidate.probability = static_cast<float>(value);
        exp_sum += value;
    }
    if (exp_sum <= 0.0 || !std::isfinite(exp_sum)) {
        throw std::runtime_error(
            "BuildNextTokenDistribution failed to normalize logits");
    }
    for (auto& candidate : candidates) {
        candidate.probability =
            static_cast<float>(static_cast<double>(candidate.probability) /
                               exp_sum);
    }

    std::sort(candidates.begin(),
              candidates.end(),
              [](const NextTokenCandidate& a, const NextTokenCandidate& b) {
                  if (a.probability == b.probability) {
                      return a.token_id < b.token_id;
                  }
                  return a.probability > b.probability;
              });

    if (config.top_k > 0 && candidates.size() > config.top_k) {
        candidates.resize(config.top_k);
    }

    if (config.top_p < 1.0f) {
        std::vector<NextTokenCandidate> nucleus;
        nucleus.reserve(candidates.size());
        float cumulative = 0.0f;
        for (const auto& candidate : candidates) {
            nucleus.push_back(candidate);
            cumulative += candidate.probability;
            if (cumulative >= config.top_p) {
                break;
            }
        }
        candidates = std::move(nucleus);
    }

    float filtered_sum = 0.0f;
    for (const auto& candidate : candidates) {
        filtered_sum += candidate.probability;
    }
    if (filtered_sum <= 0.0f) {
        throw std::runtime_error(
            "BuildNextTokenDistribution produced no viable candidates");
    }
    for (auto& candidate : candidates) {
        candidate.probability /= filtered_sum;
    }

    return candidates;
}

NextTokenSelection SelectNextTokenFromDistribution(
    const std::vector<NextTokenCandidate>& candidates,
    const LanguageModelGenerationConfig& config,
    std::mt19937& rng) {

    RequireValidLanguageModelGenerationConfig(config);
    if (candidates.empty()) {
        throw std::invalid_argument(
            "SelectNextTokenFromDistribution requires candidates");
    }

    NextTokenSelection selection;
    selection.candidates = candidates;

    if (config.sampling_mode == LanguageModelSamplingMode::Greedy) {
        const auto best = std::max_element(
            candidates.begin(),
            candidates.end(),
            [](const NextTokenCandidate& a, const NextTokenCandidate& b) {
                if (a.probability == b.probability) {
                    return a.token_id > b.token_id;
                }
                return a.probability < b.probability;
            });
        selection.token_id = best->token_id;
        selection.probability = best->probability;
        return selection;
    }

    std::vector<double> weights;
    weights.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        weights.push_back(static_cast<double>(candidate.probability));
    }

    std::discrete_distribution<size_t> distribution(
        weights.begin(), weights.end());
    const size_t index = distribution(rng);
    selection.token_id = candidates[index].token_id;
    selection.probability = candidates[index].probability;
    return selection;
}

NextTokenSelection SelectNextTokenFromLogits(
    const std::vector<float>& logits,
    size_t batch_size,
    size_t sequence_length,
    size_t vocab_size,
    const LanguageModelGenerationConfig& config,
    std::mt19937& rng,
    size_t batch_index) {

    const auto candidates = BuildNextTokenDistribution(
        logits,
        batch_size,
        sequence_length,
        vocab_size,
        config,
        batch_index);
    return SelectNextTokenFromDistribution(candidates, config, rng);
}

std::vector<int64_t> GenerateTokenIdsWithConfig(
    SequentialModel& model,
    const std::vector<int64_t>& prompt_ids,
    const LanguageModelGenerationConfig& config,
    uint32_t seed) {

    RequireValidLanguageModelGenerationConfig(config);
    if (prompt_ids.empty()) {
        throw std::invalid_argument(
            "GenerateTokenIdsWithConfig requires at least one prompt token");
    }

    std::vector<int64_t> generated = prompt_ids;
    std::vector<int64_t> new_tokens;
    new_tokens.reserve(config.max_new_tokens);

    std::mt19937 rng(seed);
    for (size_t step = 0; step < config.max_new_tokens; ++step) {
        Tensor input({1, generated.size()}, generated.data(), DataType::Int64);
        Tensor logits = model.Forward(input);
        const auto& shape = logits.Shape();
        if (shape.size() != 3 || shape[0] != 1 ||
            shape[1] != generated.size() || shape[2] == 0 ||
            logits.GetDataType() != DataType::Float32) {
            throw std::invalid_argument(
                "GenerateTokenIdsWithConfig model must return Float32 [1, seq, vocab] logits");
        }

        const float* data = logits.Data<float>();
        const std::vector<float> logits_values(data, data + logits.NumElements());
        const auto selection = SelectNextTokenFromLogits(
            logits_values,
            shape[0],
            shape[1],
            shape[2],
            config,
            rng,
            0);

        generated.push_back(selection.token_id);
        new_tokens.push_back(selection.token_id);

        if (config.eos_token_id >= 0 &&
            selection.token_id == config.eos_token_id) {
            break;
        }
    }

    return config.include_prompt ? generated : new_tokens;
}

} // namespace cyxwiz
