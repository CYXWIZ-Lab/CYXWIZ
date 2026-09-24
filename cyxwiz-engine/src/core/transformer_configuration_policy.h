#pragma once

#include <vector>
#include <cctype>
#include <algorithm>
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
    float ffn_dropout = 0.0f;
    bool use_bias = true;
    bool norm_first = false;
    // Decoder block choices (tofix112). Defaults are the classic block.
    std::string norm_type = "layer_norm";     // layer_norm | rms_norm
    float norm_eps = 1e-5f;
    std::string ffn_type = "mlp";             // mlp | gated
    std::string ffn_activation = "relu";      // see kTransformerFfnActivations
    bool ffn_bias = true;
    std::string position_encoding = "external";  // external | rope
    float rope_base = 10000.0f;

    bool HasClassicBlock() const {
        return norm_type == "layer_norm" && norm_eps == 1e-5f && ffn_type == "mlp" &&
               ffn_activation == "relu" && ffn_bias && position_encoding == "external";
    }
};

// Feed-forward activations a decoder block accepts. "gelu" is the tanh
// approximation (torch GELU(approximate="tanh")); "silu" is Swish. With
// ffn_type=gated: sigmoid=GLU, relu=ReGLU, gelu=GEGLU, silu=SwiGLU.
inline const std::vector<std::string>& TransformerFfnActivations() {
    static const std::vector<std::string> values = {
        "relu", "gelu", "silu", "mish", "elu", "selu", "leaky_relu", "sigmoid", "tanh", "hardswish"};
    return values;
}

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

    if (const std::string* text = FindNonEmpty(parameters, "ffn_dropout")) {
        const auto probability = ParseFiniteDouble(*text);
        if (!probability || *probability < 0.0 || *probability >= 1.0 ||
            static_cast<float>(*probability) >= 1.0f) {
            return std::string(layer_name) + " ffn_dropout must be a finite value in [0,1).";
        }
        configuration.ffn_dropout = static_cast<float>(*probability);
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
    // Block choices. Only the decoder implements them; the encoder must stay
    // classic until it gains the same options (never silently ignored).
    const auto choose = [&](const char* key, const std::vector<std::string>& allowed,
                            std::string& out) -> std::optional<std::string> {
        const std::string* text = FindNonEmpty(parameters, key);
        if (!text) return std::nullopt;
        std::string value;
        for (const char c : *text) value.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        if (value == "swish") value = "silu";
        if (value == "layernorm") value = "layer_norm";
        if (value == "rmsnorm") value = "rms_norm";
        if (std::find(allowed.begin(), allowed.end(), value) == allowed.end()) {
            std::string list;
            for (const auto& v : allowed) list += (list.empty() ? "" : ", ") + v;
            return std::string(layer_name) + " " + key + " must be one of: " + list + ".";
        }
        out = value;
        return std::nullopt;
    };
    if (const auto error = choose("norm_type", {"layer_norm", "rms_norm"}, configuration.norm_type)) return error;
    if (const auto error = choose("ffn_type", {"mlp", "gated"}, configuration.ffn_type)) return error;
    if (const auto error = choose("ffn_activation", TransformerFfnActivations(), configuration.ffn_activation)) return error;
    if (const std::string* text = FindNonEmpty(parameters, "norm_eps")) {
        const auto eps = ParseFiniteDouble(*text);
        if (!eps || *eps <= 0.0 || *eps >= 1.0 || static_cast<float>(*eps) <= 0.0f) {
            return std::string(layer_name) + " norm_eps must be a finite value in (0,1).";
        }
        configuration.norm_eps = static_cast<float>(*eps);
    }
    if (const auto error = ResolveBoolParameter(parameters, layer_name, "ffn_bias", true,
                                                configuration.ffn_bias)) {
        return error;
    }
    if (const auto error = choose("position_encoding", {"external", "rope"},
                                  configuration.position_encoding)) {
        return error;
    }
    if (const std::string* text = FindNonEmpty(parameters, "rope_base")) {
        const auto base = ParseFiniteDouble(*text);
        if (!base || *base <= 1.0) {
            return std::string(layer_name) + " rope_base must be a finite value greater than 1.";
        }
        configuration.rope_base = static_cast<float>(*base);
    }
    if (configuration.position_encoding == "rope" &&
        (configuration.model_width / configuration.num_heads) % 2 != 0) {
        return std::string(layer_name) +
            " position_encoding=rope needs an even head width (d_model / num_heads).";
    }
    if (is_encoder && !configuration.HasClassicBlock()) {
        return std::string(layer_name) +
            " supports only the classic block (layer_norm, mlp, relu, ffn_bias=true, external positions) for now; "
            "norm_type/ffn_type/ffn_activation/ffn_bias/norm_eps/position_encoding are implemented on TransformerDecoder.";
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
