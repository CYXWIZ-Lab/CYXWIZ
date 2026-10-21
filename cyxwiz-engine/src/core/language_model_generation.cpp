#include "language_model_generation.h"

#include "executable_model.h"
#include "training_generation_preview_settings.h"
#include <cyxwiz/sequential.h>
#include "algorithms/arrayfire_backend_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

size_t RemainingGenerationBudget(size_t max_new_remaining,
                                 size_t current_sequence_length,
                                 size_t max_context_tokens) {
    if (max_context_tokens == 0) {
        return max_new_remaining;
    }
    if (current_sequence_length >= max_context_tokens) {
        return 0;
    }
    return std::min(max_new_remaining,
                    max_context_tokens - current_sequence_length);
}

} // namespace

namespace cyxwiz {

std::string LanguageModelGenerationStopReasonName(
    LanguageModelGenerationStopReason reason) {
    switch (reason) {
        case LanguageModelGenerationStopReason::MaxTokens:
            return "max_tokens";
        case LanguageModelGenerationStopReason::EosToken:
            return "eos";
        case LanguageModelGenerationStopReason::Error:
            return "error";
        case LanguageModelGenerationStopReason::UserCancelled:
            return "user_cancel";
        case LanguageModelGenerationStopReason::StopText:
            return "stop_text";
    }
    return "unknown";
}

void ApplyGenerationDecoding(const TrainingGenerationPreviewSettings& s,
                             LanguageModelGenerationConfig& config) {
    config.repetition_penalty = s.repetition_penalty;
    config.frequency_penalty = s.frequency_penalty;
    config.presence_penalty = s.presence_penalty;
    config.no_repeat_ngram_size = static_cast<size_t>(std::max(0, s.no_repeat_ngram));
    config.sampling_mode = LanguageModelSamplingMode::Greedy;
    config.num_beams = 1;
    if (s.decoding == GenerationPreviewDecoding::Sampling) {
        config.sampling_mode = LanguageModelSamplingMode::Multinomial;
        config.temperature = s.temperature;
        config.top_k = static_cast<size_t>(std::max(0, s.top_k));
        config.top_p = s.top_p;
        config.min_p = s.min_p;
    } else if (s.decoding == GenerationPreviewDecoding::Beam) {
        config.num_beams = static_cast<size_t>(std::max(2, s.num_beams));
        config.length_penalty = s.length_penalty;
    }
}

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
    if (!std::isfinite(config.repetition_penalty) ||
        config.repetition_penalty < 1.0f) {
        issues.emplace_back("repetition_penalty must be finite and at least 1");
    }
    if (!std::isfinite(config.min_p) || config.min_p < 0.0f || config.min_p >= 1.0f) {
        issues.emplace_back("min_p must be in the range [0, 1)");
    }
    if (!std::isfinite(config.frequency_penalty) || config.frequency_penalty < 0.0f ||
        !std::isfinite(config.presence_penalty) || config.presence_penalty < 0.0f) {
        issues.emplace_back("frequency and presence penalties must be finite and at least 0");
    }
    if (config.num_beams == 0) {
        issues.emplace_back("num_beams must be at least 1");
    }
    if (!std::isfinite(config.length_penalty)) {
        issues.emplace_back("length_penalty must be finite");
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

    if (config.min_p > 0.0f && !candidates.empty()) {
        const float floor = candidates.front().probability * config.min_p;
        size_t keep = 1;
        while (keep < candidates.size() && candidates[keep].probability >= floor) ++keep;
        candidates.resize(keep);
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

namespace {

// One forward pass: `input` holds the newest tokens, `offset` how many were
// fed before them (non-zero only on the cached path).
using GenerationForward = std::function<Tensor(const Tensor& input, size_t offset)>;

// Penalties on one row of next-token logits, given every token so far
// (`generated`) and how many of them are new (the rest is the prompt).
void ApplyLogitPenalties(std::vector<float>& logits, const std::vector<int64_t>& generated, size_t prompt_length,
                         const LanguageModelGenerationConfig& config) {
    const auto valid = [&logits](int64_t id) { return id >= 0 && static_cast<size_t>(id) < logits.size(); };
    if (config.repetition_penalty > 1.0f) {
        std::vector<bool> penalized(logits.size(), false);
        for (const int64_t id : generated) {
            if (!valid(id) || penalized[static_cast<size_t>(id)]) continue;
            penalized[static_cast<size_t>(id)] = true;
            float& value = logits[static_cast<size_t>(id)];
            value = value > 0.0f ? value / config.repetition_penalty : value * config.repetition_penalty;
        }
    }
    if (config.frequency_penalty > 0.0f || config.presence_penalty > 0.0f) {
        std::vector<size_t> counts(logits.size(), 0);
        for (size_t i = prompt_length; i < generated.size(); ++i)
            if (valid(generated[i])) ++counts[static_cast<size_t>(generated[i])];
        for (size_t id = 0; id < counts.size(); ++id)
            if (counts[id]) logits[id] -= config.frequency_penalty * counts[id] + config.presence_penalty;
    }
    const size_t n = config.no_repeat_ngram_size;
    if (n > 0 && generated.size() + 1 >= n) {
        // The last n-1 tokens followed by any token that followed them before.
        const size_t context = n - 1;
        const size_t tail = generated.size() - context;
        for (size_t start = 0; start + n <= generated.size(); ++start) {
            if (!std::equal(generated.begin() + static_cast<std::ptrdiff_t>(start),
                            generated.begin() + static_cast<std::ptrdiff_t>(start + context),
                            generated.begin() + static_cast<std::ptrdiff_t>(tail)))
                continue;
            const int64_t banned = generated[start + context];
            if (valid(banned)) logits[static_cast<size_t>(banned)] = -std::numeric_limits<float>::infinity();
        }
    }
}

// The last position's logits as host values, checked for shape.
std::vector<float> LastLogits(const Tensor& logits, size_t fed, const char* who) {
    const auto& shape = logits.Shape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != fed || shape[2] == 0 ||
        logits.GetDataType() != DataType::Float32) {
        throw std::invalid_argument(std::string(who) + " model must return Float32 [1, seq, vocab] logits");
    }
    // Sampling consumes only the final position. Keep the prefix logits on
    // device and materialize one vocabulary row at this output boundary.
    const Tensor next_logits = logits.Slice(1, -1);
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::OutputMaterialization,
        "GenerateTokenIdsWithReport::NextTokenLogits");
    const float* data = next_logits.ReadData<float>();
    return std::vector<float>(data, data + next_logits.NumElements());
}

LanguageModelGenerationResult GenerateWithForward(
    const GenerationForward& forward,
    bool use_cache,
    size_t start_offset,
    const std::vector<int64_t>& prompt_ids,
    const LanguageModelGenerationConfig& config,
    uint32_t seed) {

    RequireValidLanguageModelGenerationConfig(config);
    if (prompt_ids.empty()) {
        throw std::invalid_argument(
            "GenerateTokenIdsWithReport requires at least one prompt token");
    }
    if (config.max_context_tokens > 0 &&
        prompt_ids.size() >= config.max_context_tokens) {
        throw std::invalid_argument(
            "GenerateTokenIdsWithReport prompt length meets or exceeds available generation context");
    }

    LanguageModelGenerationResult result;
    result.prompt_length = prompt_ids.size();
    result.max_new_tokens = config.max_new_tokens;
    result.remaining_budget = RemainingGenerationBudget(
        config.max_new_tokens,
        prompt_ids.size(),
        config.max_context_tokens);
    result.include_prompt = config.include_prompt;
    result.stop_reason = LanguageModelGenerationStopReason::MaxTokens;

    std::vector<int64_t> generated = prompt_ids;
    std::vector<int64_t> new_tokens;
    new_tokens.reserve(config.max_new_tokens);
    result.steps.reserve(config.max_new_tokens);

    std::mt19937 rng(seed);
    result.used_kv_cache = use_cache;
    for (size_t step = 0; step < config.max_new_tokens; ++step) {
        if (config.should_cancel && config.should_cancel()) {
            result.stop_reason = LanguageModelGenerationStopReason::UserCancelled;
            break;
        }
        if (config.max_context_tokens > 0 &&
            generated.size() >= config.max_context_tokens) {
            break;
        }

        // Cached path: the prompt once (from `start_offset` when the cache
        // already holds its beginning), then only the newest token.
        const bool incremental_step = use_cache && step > 0;
        const size_t fed = incremental_step ? 1 : generated.size() - start_offset;
        Tensor input({1, fed}, generated.data() + (generated.size() - fed), DataType::Int64);
        Tensor logits = forward(input, generated.size() - fed);
        std::vector<float> logits_values = LastLogits(logits, fed, "GenerateTokenIdsWithReport");
        const size_t vocab = logits_values.size();
        ApplyLogitPenalties(logits_values, generated, prompt_ids.size(), config);
        const auto selection = SelectNextTokenFromLogits(
            logits_values,
            1,
            1,
            vocab,
            config,
            rng,
            0);

        result.steps.push_back({
            step,
            generated.size(),
            selection.token_id,
            selection.probability,
            selection.candidates
        });
        generated.push_back(selection.token_id);
        new_tokens.push_back(selection.token_id);
        result.remaining_budget = RemainingGenerationBudget(
            config.max_new_tokens - new_tokens.size(),
            generated.size(),
            config.max_context_tokens);
        if (config.on_step) config.on_step(result.steps.back());

        if (config.eos_token_id >= 0 &&
            selection.token_id == config.eos_token_id) {
            result.stop_reason = LanguageModelGenerationStopReason::EosToken;
            break;
        }
        if (config.stop_check && config.stop_check(new_tokens)) {
            result.stop_reason = LanguageModelGenerationStopReason::StopText;
            break;
        }
    }

    result.new_token_ids = new_tokens;
    result.token_ids = config.include_prompt ? generated : new_tokens;
    return result;
}

// Beam search over full forwards (each beam recomputes its sequence).
// Deterministic; penalties and the stop check apply per beam.
LanguageModelGenerationResult GenerateBeamSearch(
    const std::function<Tensor(const Tensor&)>& forward,
    const std::vector<int64_t>& prompt_ids,
    const LanguageModelGenerationConfig& config) {
    RequireValidLanguageModelGenerationConfig(config);
    if (prompt_ids.empty()) throw std::invalid_argument("GenerateBeamSearch requires at least one prompt token");
    if (config.max_context_tokens > 0 && prompt_ids.size() >= config.max_context_tokens)
        throw std::invalid_argument("GenerateBeamSearch prompt length meets or exceeds available generation context");
    struct Beam {
        std::vector<int64_t> tokens;  // Prompt and new tokens.
        double log_probability = 0.0;
        LanguageModelGenerationStopReason stop = LanguageModelGenerationStopReason::MaxTokens;
    };
    const auto score = [&](const Beam& beam) {
        const size_t length = std::max<size_t>(1, beam.tokens.size() - prompt_ids.size());
        return beam.log_probability / std::pow(static_cast<double>(length), config.length_penalty);
    };
    std::vector<Beam> alive = {{prompt_ids, 0.0}};
    std::vector<Beam> finished;
    bool cancelled = false;
    size_t steps = 0;
    for (; steps < config.max_new_tokens && !alive.empty(); ++steps) {
        if (config.should_cancel && config.should_cancel()) {
            cancelled = true;
            break;
        }
        if (config.max_context_tokens > 0 && alive.front().tokens.size() >= config.max_context_tokens) break;
        struct Option {
            size_t beam;
            int64_t token;
            double log_probability;
        };
        std::vector<Option> options;
        for (size_t b = 0; b < alive.size(); ++b) {
            const auto& tokens = alive[b].tokens;
            Tensor input({1, tokens.size()}, tokens.data(), DataType::Int64);
            std::vector<float> logits = LastLogits(forward(input), tokens.size(), "GenerateBeamSearch");
            ApplyLogitPenalties(logits, tokens, prompt_ids.size(), config);
            const float max_logit = *std::max_element(logits.begin(), logits.end());
            double sum = 0.0;
            for (const float value : logits) sum += std::exp(static_cast<double>(value - max_logit));
            const double log_sum = std::log(sum) + max_logit;
            std::vector<size_t> order(logits.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            const size_t take = std::min(order.size(), config.num_beams * 2);
            std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(take), order.end(),
                              [&logits](size_t a, size_t c) { return logits[a] > logits[c]; });
            for (size_t i = 0; i < take; ++i) {
                if (!std::isfinite(logits[order[i]])) continue;
                options.push_back({b, static_cast<int64_t>(order[i]),
                                   alive[b].log_probability + logits[order[i]] - log_sum});
            }
        }
        std::sort(options.begin(), options.end(),
                  [](const Option& a, const Option& c) { return a.log_probability > c.log_probability; });
        std::vector<Beam> next;
        for (const auto& option : options) {
            if (next.size() >= config.num_beams) break;
            Beam beam = alive[option.beam];
            beam.tokens.push_back(option.token);
            beam.log_probability = option.log_probability;
            const std::vector<int64_t> fresh(beam.tokens.begin() + static_cast<std::ptrdiff_t>(prompt_ids.size()),
                                             beam.tokens.end());
            if (config.eos_token_id >= 0 && option.token == config.eos_token_id) {
                beam.stop = LanguageModelGenerationStopReason::EosToken;
                finished.push_back(std::move(beam));
            } else if (config.stop_check && config.stop_check(fresh)) {
                beam.stop = LanguageModelGenerationStopReason::StopText;
                finished.push_back(std::move(beam));
            } else {
                next.push_back(std::move(beam));
            }
        }
        alive = std::move(next);
        // Done once the best finished beam beats every live one.
        if (!finished.empty() && !alive.empty()) {
            double best_finished = -std::numeric_limits<double>::infinity();
            for (const auto& beam : finished) best_finished = std::max(best_finished, score(beam));
            bool any_better = false;
            for (const auto& beam : alive) any_better |= score(beam) > best_finished;
            if (!any_better) break;
        }
    }
    for (auto& beam : alive) {
        if (cancelled) beam.stop = LanguageModelGenerationStopReason::UserCancelled;
        finished.push_back(std::move(beam));
    }
    if (finished.empty()) throw std::runtime_error("GenerateBeamSearch produced no sequence");
    const auto best = std::max_element(finished.begin(), finished.end(),
                                       [&score](const Beam& a, const Beam& c) { return score(a) < score(c); });
    LanguageModelGenerationResult result;
    result.prompt_length = prompt_ids.size();
    result.max_new_tokens = config.max_new_tokens;
    result.include_prompt = config.include_prompt;
    result.stop_reason = best->stop;
    result.new_token_ids.assign(best->tokens.begin() + static_cast<std::ptrdiff_t>(prompt_ids.size()),
                                best->tokens.end());
    result.token_ids = config.include_prompt ? best->tokens : result.new_token_ids;
    for (size_t i = 0; i < result.new_token_ids.size(); ++i)
        result.steps.push_back({i, prompt_ids.size() + i, result.new_token_ids[i], 0.0f, {}});
    result.remaining_budget = RemainingGenerationBudget(config.max_new_tokens - result.new_token_ids.size(),
                                                        best->tokens.size(), config.max_context_tokens);
    if (config.on_step)
        for (const auto& step : result.steps) config.on_step(step);
    return result;
}

}  // namespace

LanguageModelGenerationResult GenerateTokenIdsWithReport(
    SequentialModel& model,
    const std::vector<int64_t>& prompt_ids,
    const LanguageModelGenerationConfig& config,
    uint32_t seed) {
    if (config.num_beams > 1) {
        if (config.reuse_cache) config.reuse_cache->clear();
        return GenerateBeamSearch([&model](const Tensor& input) { return model.Forward(input); }, prompt_ids, config);
    }
    const bool use_cache = config.use_kv_cache && model.SupportsIncrementalDecoding();
    // A reused cache must hold exactly the start of this prompt and leave at
    // least one token to feed; otherwise it is emptied (offset 0 resets it).
    size_t start_offset = 0;
    auto* reuse = use_cache ? config.reuse_cache : nullptr;
    if (config.reuse_cache && !use_cache) config.reuse_cache->clear();
    if (reuse && !reuse->empty() && reuse->size() < prompt_ids.size() &&
        std::equal(reuse->begin(), reuse->end(), prompt_ids.begin()))
        start_offset = reuse->size();
    struct CacheRelease {  // free cached keys/values however generation ends
        SequentialModel& model;
        bool active;
        std::vector<int64_t>* reuse;
        ~CacheRelease() {
            if (!active) return;
            model.ResetIncrementalState();
            if (reuse) reuse->clear();
        }
    } cache_release{model, use_cache, reuse};
    auto result = GenerateWithForward(
        [&model, use_cache](const Tensor& input, size_t offset) {
            return use_cache ? model.ForwardIncremental(input, offset) : model.Forward(input);
        },
        use_cache, start_offset, prompt_ids, config, seed);
    result.reused_cache_tokens = start_offset;
    if (reuse) {
        // The cache now covers every token fed: the prompt and each new token
        // except the last (which no forward has seen yet).
        std::vector<int64_t> covered = prompt_ids;
        if (!result.new_token_ids.empty())
            covered.insert(covered.end(), result.new_token_ids.begin(), result.new_token_ids.end() - 1);
        *reuse = std::move(covered);
        cache_release.active = false;  // kept for the next call
    }
    return result;
}

LanguageModelGenerationResult GenerateTokenIdsWithReport(
    IExecutableModel& model,
    const std::vector<int64_t>& prompt_ids,
    const LanguageModelGenerationConfig& config,
    uint32_t seed) {
    if (auto* sequential = model.AsSequentialModel())
        return GenerateTokenIdsWithReport(*sequential, prompt_ids, config, seed);
    if (config.reuse_cache) config.reuse_cache->clear();
    if (config.num_beams > 1)
        return GenerateBeamSearch([&model](const Tensor& input) { return model.Forward(input); }, prompt_ids, config);
    return GenerateWithForward(
        [&model](const Tensor& input, size_t) { return model.Forward(input); },
        false, 0, prompt_ids, config, seed);
}

std::vector<int64_t> GenerateTokenIdsWithConfig(
    SequentialModel& model,
    const std::vector<int64_t>& prompt_ids,
    const LanguageModelGenerationConfig& config,
    uint32_t seed) {
    return GenerateTokenIdsWithReport(model, prompt_ids, config, seed).token_ids;
}
} // namespace cyxwiz
