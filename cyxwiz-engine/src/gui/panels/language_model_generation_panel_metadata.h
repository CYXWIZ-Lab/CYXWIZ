#pragma once

#include "../../core/language_model_generation.h"

#include <cstdint>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace cyxwiz {

struct LanguageModelGenerationPanelRunMetadata {
    std::string stop_reason;
    std::string sampling_settings;
    size_t prompt_length = 0;
    size_t max_context_length = 0;
    size_t remaining_budget = 0;
    std::vector<NextTokenCandidate> last_candidates;
};

inline std::string FormatLanguageModelGenerationSamplingSettings(
    const LanguageModelGenerationConfig& config,
    uint32_t seed) {
    std::ostringstream settings;
    settings << (config.sampling_mode == LanguageModelSamplingMode::Multinomial
                     ? "multinomial"
                     : "greedy")
             << ", max_new_tokens=" << config.max_new_tokens
             << ", temperature=" << config.temperature
             << ", top_k=" << config.top_k
             << ", top_p=" << config.top_p
             << ", eos=" << config.eos_token_id
             << ", seed=" << seed
             << ", include_prompt="
             << (config.include_prompt ? "true" : "false");
    return settings.str();
}

inline LanguageModelGenerationPanelRunMetadata
BuildLanguageModelGenerationPanelRunMetadata(
    const LanguageModelGenerationResult& report,
    const LanguageModelGenerationConfig& config,
    size_t max_context_length,
    uint32_t seed) {
    LanguageModelGenerationPanelRunMetadata metadata;
    metadata.stop_reason =
        LanguageModelGenerationStopReasonName(report.stop_reason);
    metadata.sampling_settings =
        FormatLanguageModelGenerationSamplingSettings(config, seed);
    metadata.prompt_length = report.prompt_length;
    metadata.max_context_length = max_context_length;
    metadata.remaining_budget = report.remaining_budget;
    metadata.last_candidates = report.steps.empty()
        ? std::vector<NextTokenCandidate>{}
        : report.steps.back().candidates;
    return metadata;
}

} // namespace cyxwiz