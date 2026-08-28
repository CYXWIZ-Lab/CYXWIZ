#pragma once

#include "../gui/node_editor.h"

#include <cerrno>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <string>

namespace cyxwiz {

struct DenseActivationConfiguration {
    size_t dense_units = 64;
    float negative_slope = 0.01f;
    float elu_alpha = 1.0f;
};

inline bool IsExecutableActivationNode(gui::NodeType node_type) {
    switch (node_type) {
        case gui::NodeType::ReLU:
        case gui::NodeType::LeakyReLU:
        case gui::NodeType::ELU:
        case gui::NodeType::GELU:
        case gui::NodeType::Swish:
        case gui::NodeType::Mish:
        case gui::NodeType::Sigmoid:
        case gui::NodeType::Tanh:
        case gui::NodeType::Softmax:
            return true;
        default:
            return false;
    }
}

namespace dense_activation_configuration_policy_detail {

inline const std::string* FindNonEmpty(
    const std::map<std::string, std::string>& parameters,
    const char* key) {
    const auto it = parameters.find(key);
    return it != parameters.end() && !it->second.empty() ? &it->second
                                                         : nullptr;
}

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

inline std::optional<double> ParseFiniteFloat(const std::string& text) {
    if (text.empty()) return std::nullopt;
    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (errno == ERANGE || end != text.c_str() + text.size() ||
        !std::isfinite(value) ||
        value > static_cast<double>(std::numeric_limits<float>::max()) ||
        value < -static_cast<double>(std::numeric_limits<float>::max())) {
        return std::nullopt;
    }
    return value;
}

} // namespace dense_activation_configuration_policy_detail

// One exact construction contract shared by graph compilation, model
// construction, Properties truth, and generated-code previews. The explicit
// CompiledLayer values preserve focused/direct model fixtures; persisted
// parameters take precedence when present because they are the user-authored
// graph contract.
inline std::optional<std::string> ResolveDenseActivationConfiguration(
    gui::NodeType node_type,
    const std::map<std::string, std::string>& parameters,
    DenseActivationConfiguration& configuration,
    int compiled_dense_units = 0,
    float compiled_negative_slope = 0.01f,
    float compiled_elu_alpha = 1.0f) {
    using namespace dense_activation_configuration_policy_detail;

    configuration = {};
    if (node_type == gui::NodeType::Dense) {
        long long units = compiled_dense_units > 0 ? compiled_dense_units : 64;
        if (const std::string* text = FindNonEmpty(parameters, "units")) {
            const auto parsed = ParseInteger(*text);
            if (!parsed || *parsed < 1 || *parsed > 1048576) {
                return std::string(
                    "Dense units must be an integer in [1, 1048576].");
            }
            units = *parsed;
        } else if (compiled_dense_units < 0 || compiled_dense_units > 1048576) {
            return std::string(
                "Dense units must be an integer in [1, 1048576].");
        }
        configuration.dense_units = static_cast<size_t>(units);
        return std::nullopt;
    }

    if (node_type == gui::NodeType::LeakyReLU) {
        double slope = compiled_negative_slope;
        if (const std::string* text = FindNonEmpty(parameters, "negative_slope")) {
            const auto parsed = ParseFiniteFloat(*text);
            if (!parsed || *parsed < 0.0) {
                return std::string(
                    "LeakyReLU negative_slope must be a finite number >= 0.");
            }
            slope = *parsed;
        } else if (!std::isfinite(slope) || slope < 0.0) {
            return std::string(
                "LeakyReLU negative_slope must be a finite number >= 0.");
        }
        configuration.negative_slope = static_cast<float>(slope);
        return std::nullopt;
    }

    if (node_type == gui::NodeType::ELU) {
        double alpha = compiled_elu_alpha;
        if (const std::string* text = FindNonEmpty(parameters, "alpha")) {
            const auto parsed = ParseFiniteFloat(*text);
            if (!parsed || *parsed <= 0.0) {
                return std::string(
                    "ELU alpha must be a finite number > 0.");
            }
            alpha = *parsed;
        } else if (!std::isfinite(alpha) || alpha <= 0.0) {
            return std::string(
                "ELU alpha must be a finite number > 0.");
        }
        configuration.elu_alpha = static_cast<float>(alpha);
        return std::nullopt;
    }

    return std::nullopt;
}

inline std::optional<std::string>
ResolveInvalidDenseActivationConfigurationReason(
    gui::NodeType node_type,
    const std::map<std::string, std::string>& parameters) {
    DenseActivationConfiguration configuration;
    return ResolveDenseActivationConfiguration(
        node_type, parameters, configuration);
}

} // namespace cyxwiz
