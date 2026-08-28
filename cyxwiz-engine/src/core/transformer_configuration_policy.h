#pragma once

#include "../gui/node_editor.h"

#include <cerrno>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <string>

namespace cyxwiz {
namespace transformer_configuration_policy_detail {

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

inline bool ParseBool(const std::string& text, bool& value) {
    if (text == "true" || text == "1") {
        value = true;
        return true;
    }
    if (text == "false" || text == "0") {
        value = false;
        return true;
    }
    return false;
}

inline std::optional<std::string> ValidatePositiveIntegerAliases(
    const std::map<std::string, std::string>& parameters,
    const char* layer_name,
    const char* canonical,
    std::initializer_list<const char*> aliases,
    long long default_value,
    long long& resolved) {
    resolved = default_value;
    std::optional<long long> observed;
    const auto inspect = [&](const char* key) -> std::optional<std::string> {
        const std::string* text = FindNonEmpty(parameters, key);
        if (!text) return std::nullopt;
        const auto parsed = ParseInteger(*text);
        if (!parsed || *parsed < 1 ||
            *parsed > static_cast<long long>(std::numeric_limits<int>::max())) {
            return std::string(layer_name) + " " + key +
                " must be an integer >= 1.";
        }
        if (observed && *observed != *parsed) {
            return std::string(layer_name) + " " + canonical +
                " conflicts with legacy " + key + ". Keep one effective value.";
        }
        observed = *parsed;
        return std::nullopt;
    };

    if (const auto error = inspect(canonical)) return error;
    for (const char* alias : aliases) {
        if (const auto error = inspect(alias)) return error;
    }
    if (observed) resolved = *observed;
    return std::nullopt;
}

inline std::optional<std::string> ResolveDropoutAliases(
    const std::map<std::string, std::string>& parameters,
    const char* layer_name,
    double default_value,
    double& resolved) {
    resolved = default_value;
    std::optional<double> observed;
    for (const char* key : {"dropout", "dropout_rate"}) {
        const std::string* text = FindNonEmpty(parameters, key);
        if (!text) continue;
        const auto parsed = ParseFiniteDouble(*text);
        if (!parsed || *parsed < 0.0 || *parsed >= 1.0) {
            return std::string(layer_name) + " " + key +
                " must be a finite value in [0, 1).";
        }
        if (observed && *observed != *parsed) {
            return std::string(layer_name) +
                " dropout conflicts with legacy dropout_rate.";
        }
        observed = *parsed;
    }
    if (observed) resolved = *observed;
    return std::nullopt;
}

inline std::optional<std::string> ResolveBoolParameter(
    const std::map<std::string, std::string>& parameters,
    const char* layer_name,
    const char* key,
    bool default_value,
    bool& resolved) {
    resolved = default_value;
    const std::string* text = FindNonEmpty(parameters, key);
    if (!text) return std::nullopt;
    if (!ParseBool(*text, resolved)) {
        return std::string(layer_name) + " " + key +
            " must be true, false, 1, or 0.";
    }
    return std::nullopt;
}

} // namespace transformer_configuration_policy_detail

struct TransformerConfiguration {
    size_t model_width = 512;
    size_t num_heads = 8;
    size_t feedforward_width = 2048;
    size_t max_sequence_length = 5000;
    float dropout = 0.1f;
    bool use_bias = true;
    bool norm_first = false;
};

// Shared fail-closed policy for the executable unary transformer path. Legacy
// aliases remain readable only when they agree with the canonical field.
inline std::optional<std::string> ResolveTransformerConfiguration(
    gui::NodeType node_type,
    const std::map<std::string, std::string>& parameters,
    TransformerConfiguration& configuration) {
    using namespace transformer_configuration_policy_detail;

    const bool is_attention = node_type == gui::NodeType::MultiHeadAttention;
    const bool is_encoder = node_type == gui::NodeType::TransformerEncoder;
    const bool is_decoder = node_type == gui::NodeType::TransformerDecoder;
    const bool is_positional = node_type == gui::NodeType::PositionalEncoding;
    if (!is_attention && !is_encoder && !is_decoder && !is_positional) {
        return std::nullopt;
    }

    configuration = TransformerConfiguration{};

    const char* layer_name = is_attention
        ? "MultiHeadAttention"
        : (is_encoder ? "TransformerEncoder"
                      : (is_decoder ? "TransformerDecoder"
                                    : "PositionalEncoding"));

    long long model_width = 512;
    if (const auto error = ValidatePositiveIntegerAliases(
            parameters, layer_name,
            is_attention ? "embed_dim" : "d_model",
            is_attention ? std::initializer_list<const char*>{"d_model"}
                         : std::initializer_list<const char*>{"embed_dim"},
            512, model_width)) {
        return error;
    }
    configuration.model_width = static_cast<size_t>(model_width);

    if (is_positional) {
        long long maximum_length = 5000;
        if (const auto error = ValidatePositiveIntegerAliases(
            parameters, layer_name, "max_sequence_length",
            {"max_len", "max_length", "max_seq_len"}, 5000,
            maximum_length)) {
            return error;
        }
        configuration.max_sequence_length =
            static_cast<size_t>(maximum_length);
        configuration.dropout = 0.0f;
        return std::nullopt;
    }

    long long num_heads = 8;
    if (const auto error = ValidatePositiveIntegerAliases(
            parameters, layer_name, "num_heads",
            is_attention ? std::initializer_list<const char*>{"heads"}
                         : std::initializer_list<const char*>{"nhead"},
            8, num_heads)) {
        return error;
    }
    configuration.num_heads = static_cast<size_t>(num_heads);
    if (model_width % num_heads != 0) {
        return std::string(layer_name) +
            " model width must be divisible by num_heads; silent one-head "
            "fallback is not an executable configuration contract.";
    }
    double dropout = is_attention ? 0.0 : 0.1;
    if (const auto error = ResolveDropoutAliases(
            parameters, layer_name, dropout, dropout)) {
        return error;
    }
    configuration.dropout = static_cast<float>(dropout);

    if (is_attention) {
        if (const auto error = ResolveBoolParameter(
                parameters, layer_name, "use_bias", true,
                configuration.use_bias)) {
            return error;
        }
        if (const std::string* batch_first =
                FindNonEmpty(parameters, "batch_first")) {
            bool value = false;
            if (!ParseBool(*batch_first, value) || !value) {
                return std::string(
                    "MultiHeadAttention supports only batch_first=true "
                    "[batch, sequence, features] input.");
            }
        }
        return std::nullopt;
    }

    long long feedforward_width = model_width * 4;
    if (const auto error = ValidatePositiveIntegerAliases(
            parameters, layer_name, "dim_feedforward",
            {"ff_dim", "d_ff"}, feedforward_width, feedforward_width)) {
        return error;
    }
    configuration.feedforward_width =
        static_cast<size_t>(feedforward_width);
    if (const auto error = ResolveBoolParameter(
            parameters, layer_name, "norm_first", false,
            configuration.norm_first)) {
        return error;
    }
    if (const std::string* layers = FindNonEmpty(parameters, "num_layers")) {
        const auto parsed = ParseInteger(*layers);
        if (!parsed || *parsed != 1) {
            return std::string(layer_name) +
                " represents exactly one block. Stack multiple nodes instead "
                "of setting num_layers.";
        }
    }
    return std::nullopt;
}

inline std::optional<std::string> ResolveInvalidTransformerConfigurationReason(
    gui::NodeType node_type,
    const std::map<std::string, std::string>& parameters) {
    TransformerConfiguration configuration;
    return ResolveTransformerConfiguration(node_type, parameters,
                                           configuration);
}

} // namespace cyxwiz
