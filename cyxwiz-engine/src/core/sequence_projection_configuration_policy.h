#pragma once

#include "../gui/node_editor.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <string>

namespace cyxwiz {
namespace sequence_projection_configuration_policy_detail {

inline std::optional<long long> ParseInteger(const std::string& text) {
    if (text.empty()) return std::nullopt;
    errno = 0;
    char* end = nullptr;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (errno == ERANGE || end != text.c_str() + text.size()) {
        return std::nullopt;
    }
    return value;
}

inline std::optional<double> ParseFiniteDouble(const std::string& text) {
    if (text.empty()) return std::nullopt;
    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (errno == ERANGE || end != text.c_str() + text.size() ||
        !std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

inline const std::string* FindNonEmpty(
    const std::map<std::string, std::string>& parameters,
    const char* key) {
    const auto it = parameters.find(key);
    return it != parameters.end() && !it->second.empty() ? &it->second
                                                         : nullptr;
}

inline std::optional<std::string> ValidatePositiveInt(
    const std::map<std::string, std::string>& parameters,
    const char* key,
    long long minimum,
    const char* layer_name) {
    const std::string* text = FindNonEmpty(parameters, key);
    if (!text) return std::nullopt;
    const auto value = ParseInteger(*text);
    if (!value || *value < minimum ||
        *value > static_cast<long long>(std::numeric_limits<int>::max())) {
        return std::string(layer_name) + " " + key + " must be an integer >= " +
            std::to_string(minimum) + ".";
    }
    return std::nullopt;
}

} // namespace sequence_projection_configuration_policy_detail

// Parameter validation shared by compiler, training construction, and test
// construction for the executable sequence lookup/projection pair.
inline std::optional<std::string>
ResolveInvalidSequenceProjectionConfigurationReason(
    gui::NodeType node_type,
    const std::map<std::string, std::string>& parameters) {
    using namespace sequence_projection_configuration_policy_detail;

    if (node_type == gui::NodeType::TimeDistributed) {
        if (const auto error =
                ValidatePositiveInt(parameters, "units", 1,
                                    "TimeDistributed Dense")) {
            return error;
        }
        if (const auto error =
                ValidatePositiveInt(parameters, "out_features", 1,
                                    "TimeDistributed Dense")) {
            return error;
        }
        const std::string* units = FindNonEmpty(parameters, "units");
        const std::string* alias = FindNonEmpty(parameters, "out_features");
        if (units && alias && ParseInteger(*units) != ParseInteger(*alias)) {
            return std::string(
                "TimeDistributed Dense units conflicts with legacy "
                "out_features. Keep one effective per-timestep output width.");
        }
        return std::nullopt;
    }

    if (node_type != gui::NodeType::Embedding) {
        return std::nullopt;
    }

    if (const auto error =
            ValidatePositiveInt(parameters, "num_embeddings", 2,
                                "Embedding")) {
        return error;
    }
    if (const auto error =
            ValidatePositiveInt(parameters, "embedding_dim", 1,
                                "Embedding")) {
        return error;
    }

    long long num_embeddings = 10000;
    if (const std::string* text = FindNonEmpty(parameters, "num_embeddings")) {
        if (const auto parsed = ParseInteger(*text)) num_embeddings = *parsed;
    }
    if (const std::string* text = FindNonEmpty(parameters, "padding_idx")) {
        const auto padding = ParseInteger(*text);
        if (!padding || *padding < -1 || *padding >= num_embeddings) {
            return std::string(
                "Embedding padding_idx must be -1 or a token id smaller than "
                "num_embeddings.");
        }
    }
    if (const std::string* text = FindNonEmpty(parameters, "max_norm")) {
        const auto max_norm = ParseFiniteDouble(*text);
        if (!max_norm || *max_norm < 0.0) {
            return std::string(
                "Embedding max_norm must be a finite number >= 0. Use 0 to "
                "disable norm clipping.");
        }
    }

    const std::string* weights = FindNonEmpty(parameters, "weights_file");
    const std::string* legacy_weights =
        FindNonEmpty(parameters, "embedding_weights_file");
    if (weights && legacy_weights && *weights != *legacy_weights) {
        return std::string(
            "Embedding weights_file conflicts with legacy "
            "embedding_weights_file. Keep one effective pretrained matrix.");
    }

    return std::nullopt;
}

} // namespace cyxwiz
