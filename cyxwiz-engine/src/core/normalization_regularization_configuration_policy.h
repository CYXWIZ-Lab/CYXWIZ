#pragma once

#include "../gui/node_editor.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace cyxwiz {

struct NormalizationRegularizationConfiguration {
    float dropout_rate = 0.5f;
    float epsilon = 1e-5f;
    float momentum = 0.1f;
    bool elementwise_affine = true;
    bool automatic_normalized_shape = true;
    std::vector<int> normalized_shape;
};

namespace normalization_regularization_configuration_policy_detail {

inline const std::string* FindNonEmpty(
    const std::map<std::string, std::string>& parameters,
    const char* key) {
    const auto it = parameters.find(key);
    return it != parameters.end() && !it->second.empty() ? &it->second
                                                         : nullptr;
}

inline std::optional<double> ParseFiniteDouble(const std::string& text) {
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

inline std::optional<int> ParsePositiveInt(const std::string& text) {
    if (text.empty()) return std::nullopt;
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (errno == ERANGE || end != text.c_str() + text.size() || value < 1 ||
        value > std::numeric_limits<int>::max()) {
        return std::nullopt;
    }
    return static_cast<int>(value);
}

inline std::string TrimAscii(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

inline std::optional<std::vector<int>> ParseNormalizedShape(
    const std::string& text) {
    if (TrimAscii(text).empty()) return std::vector<int>{};

    std::vector<int> shape;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        token = TrimAscii(std::move(token));
        const auto dimension = ParsePositiveInt(token);
        if (!dimension) return std::nullopt;
        shape.push_back(*dimension);
    }
    if (shape.empty() || (!text.empty() && text.back() == ',')) {
        return std::nullopt;
    }
    return shape;
}

inline std::optional<bool> ParseBool(const std::string& text) {
    if (text == "true" || text == "1") return true;
    if (text == "false" || text == "0") return false;
    return std::nullopt;
}

inline std::optional<std::string> ResolveEpsilon(
    const std::map<std::string, std::string>& parameters,
    const char* layer_name,
    float& epsilon) {
    const std::string* canonical = FindNonEmpty(parameters, "eps");
    const std::string* legacy = FindNonEmpty(parameters, "epsilon");
    const auto canonical_value = canonical ? ParseFiniteDouble(*canonical)
                                           : std::optional<double>{};
    const auto legacy_value = legacy ? ParseFiniteDouble(*legacy)
                                     : std::optional<double>{};
    if (canonical && !canonical_value) {
        return std::string(layer_name) + " eps must be a finite number > 0.";
    }
    if (legacy && !legacy_value) {
        return std::string(layer_name) +
            " legacy epsilon must be a finite number > 0.";
    }
    if (canonical_value && legacy_value &&
        *canonical_value != *legacy_value) {
        return std::string(layer_name) +
            " eps conflicts with legacy epsilon. Keep one effective value.";
    }
    const double value = canonical_value.value_or(
        legacy_value.value_or(static_cast<double>(epsilon)));
    if (value <= 0.0) {
        return std::string(layer_name) + " eps must be a finite number > 0.";
    }
    epsilon = static_cast<float>(value);
    return std::nullopt;
}

} // namespace normalization_regularization_configuration_policy_detail

inline std::optional<std::string>
ResolveNormalizationRegularizationConfiguration(
    gui::NodeType node_type,
    const std::map<std::string, std::string>& parameters,
    NormalizationRegularizationConfiguration& configuration) {
    using namespace normalization_regularization_configuration_policy_detail;
    configuration = {};

    if (node_type == gui::NodeType::Dropout) {
        if (const std::string* text = FindNonEmpty(parameters, "rate")) {
            const auto value = ParseFiniteDouble(*text);
            if (!value || *value < 0.0 || *value >= 1.0) {
                return std::string(
                    "Dropout rate must be a finite number in [0, 1).");
            }
            configuration.dropout_rate = static_cast<float>(*value);
        }
        return std::nullopt;
    }

    if (node_type == gui::NodeType::BatchNorm) {
        if (const auto error = ResolveEpsilon(
                parameters, "BatchNorm", configuration.epsilon)) {
            return error;
        }
        if (const std::string* text = FindNonEmpty(parameters, "momentum")) {
            const auto value = ParseFiniteDouble(*text);
            if (!value || *value < 0.0 || *value > 1.0) {
                return std::string(
                    "BatchNorm momentum must be a finite number in [0, 1].");
            }
            configuration.momentum = static_cast<float>(*value);
        }
        return std::nullopt;
    }

    if (node_type != gui::NodeType::LayerNorm) {
        return std::nullopt;
    }

    if (const auto error = ResolveEpsilon(
            parameters, "LayerNorm", configuration.epsilon)) {
        return error;
    }
    if (const std::string* text = FindNonEmpty(parameters, "normalized_shape")) {
        const auto shape = ParseNormalizedShape(*text);
        if (!shape || shape->empty()) {
            return std::string(
                "LayerNorm normalized_shape must be empty for automatic width "
                "or a comma-separated list of positive integers.");
        }
        configuration.automatic_normalized_shape = false;
        configuration.normalized_shape = *shape;
    }
    if (const std::string* text =
            FindNonEmpty(parameters, "elementwise_affine")) {
        const auto value = ParseBool(*text);
        if (!value) {
            return std::string(
                "LayerNorm elementwise_affine must be true or false.");
        }
        configuration.elementwise_affine = *value;
    }
    return std::nullopt;
}

inline std::optional<std::string>
ResolveInvalidNormalizationRegularizationConfigurationReason(
    gui::NodeType node_type,
    const std::map<std::string, std::string>& parameters) {
    NormalizationRegularizationConfiguration configuration;
    return ResolveNormalizationRegularizationConfiguration(
        node_type, parameters, configuration);
}

} // namespace cyxwiz
