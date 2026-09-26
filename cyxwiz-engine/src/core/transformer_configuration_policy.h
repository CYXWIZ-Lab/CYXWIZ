#pragma once

#include <vector>
#include <cctype>
#include <algorithm>
#include "graph_model.h"

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
    std::string position_encoding = "external";  // decoder: external | rope | alibi; attention node: none | rope | alibi
    float rope_base = 10000.0f;
    float rope_fraction = 1.0f;               // partial RoPE
    bool attention_bias = true;               // bias in attention Q/K/V/output projections
    bool qk_norm = false;                     // per-head RMSNorm of queries and keys
    std::string architecture_preset = "custom";  // custom | classic | gpt2_style | llama_style
    std::string block_layout = "sequential";  // sequential | parallel
    bool sandwich_norm = false;
    float residual_init_scale = 1.0f;
    float attn_logit_softcap = 0.0f;          // 0 = off
    size_t sliding_window = 0;                // 0 = full causal attention
    size_t num_kv_heads = 0;                  // 0 = num_heads
    bool causal = false;                      // MultiHeadAttention node only
    std::string encoding_type = "sinusoidal"; // PositionalEncoding node: sinusoidal | learned

    bool HasClassicBlock() const {
        return norm_type == "layer_norm" && norm_eps == 1e-5f && ffn_type == "mlp" &&
               ffn_activation == "relu" && ffn_bias && position_encoding == "external" &&
               attention_bias && !qk_norm && rope_fraction == 1.0f && block_layout == "sequential" &&
               !sandwich_norm && residual_init_scale == 1.0f && attn_logit_softcap == 0.0f &&
               sliding_window == 0 && num_kv_heads == 0;
    }
};

// Decoder architecture presets (tofix112 phase 4). A preset only fills block
// fields; widths, heads, dropout, norm_eps and rope_base stay the user's.
// Values are the property strings the Properties panel writes.
inline const std::map<std::string, std::map<std::string, std::string>>& TransformerArchitecturePresets() {
    static const std::map<std::string, std::map<std::string, std::string>> presets = {
        {"classic", {{"norm_first", "false"}, {"norm_type", "layer_norm"}, {"ffn_type", "mlp"},
                     {"ffn_activation", "relu"}, {"ffn_bias", "true"}, {"position_encoding", "external"},
                     {"attention_bias", "true"}, {"qk_norm", "false"}, {"block_layout", "sequential"},
                     {"sandwich_norm", "false"}}},
        // GPT-2 block; pair it with a Positional Encoding node set to learned.
        {"gpt2_style", {{"norm_first", "true"}, {"norm_type", "layer_norm"}, {"ffn_type", "mlp"},
                        {"ffn_activation", "gelu"}, {"ffn_bias", "true"}, {"position_encoding", "external"},
                        {"attention_bias", "true"}, {"qk_norm", "false"}, {"block_layout", "sequential"},
                        {"sandwich_norm", "false"}}},
        {"llama_style", {{"norm_first", "true"}, {"norm_type", "rms_norm"}, {"ffn_type", "gated"},
                         {"ffn_activation", "silu"}, {"ffn_bias", "false"}, {"position_encoding", "rope"},
                         {"attention_bias", "false"}, {"qk_norm", "false"}, {"block_layout", "sequential"},
                         {"sandwich_norm", "false"}}},
    };
    return presets;
}

// Feed-forward activations a decoder block accepts. "gelu" is the tanh
// approximation (torch GELU(approximate="tanh")); "silu" is Swish. With
// ffn_type=gated: sigmoid=GLU, relu=ReGLU, gelu=GEGLU, silu=SwiGLU.
inline const std::vector<std::string>& TransformerFfnActivations() {
    static const std::vector<std::string> values = {
        "relu", "gelu", "gelu_exact", "silu", "mish", "elu", "selu", "leaky_relu", "sigmoid", "tanh",
        "hardswish", "squared_relu"};
    return values;
}

namespace transformer_configuration_policy_detail {

inline std::optional<std::string> ResolveNonNegativeInteger(
    const std::map<std::string, std::string>& parameters, const char* layer_name,
    const char* key, size_t& out) {
    if (const std::string* text = FindNonEmpty(parameters, key)) {
        const auto value = ParseInteger(*text);
        if (!value || *value < 0 || *value > 1000000000LL) {
            return std::string(layer_name) + " " + key + " must be a non-negative integer.";
        }
        out = static_cast<size_t>(*value);
    }
    return std::nullopt;
}

// Attention-level options shared by TransformerDecoder self-attention and the
// MultiHeadAttention node: rope_base/rope_fraction, logit soft-cap, sliding
// window, key/value heads. position_encoding must already be resolved.
inline std::optional<std::string> ResolveAttentionOptions(
    const std::map<std::string, std::string>& parameters, const char* layer_name,
    TransformerConfiguration& configuration) {
    if (const std::string* text = FindNonEmpty(parameters, "rope_base")) {
        const auto base = ParseFiniteDouble(*text);
        if (!base || *base <= 1.0) {
            return std::string(layer_name) + " rope_base must be a finite value greater than 1.";
        }
        configuration.rope_base = static_cast<float>(*base);
    }
    if (const std::string* text = FindNonEmpty(parameters, "rope_fraction")) {
        const auto fraction = ParseFiniteDouble(*text);
        if (!fraction || *fraction <= 0.0 || *fraction > 1.0) {
            return std::string(layer_name) + " rope_fraction must be in (0,1].";
        }
        configuration.rope_fraction = static_cast<float>(*fraction);
    }
    const size_t head_width = configuration.model_width / configuration.num_heads;
    if (configuration.position_encoding == "rope") {
        const size_t rotary = static_cast<size_t>(configuration.rope_fraction * head_width / 2.0f + 1e-6f) * 2;
        if (configuration.rope_fraction == 1.0f && head_width % 2 != 0) {
            return std::string(layer_name) +
                " position_encoding=rope needs an even head width (d_model / num_heads).";
        }
        if (rotary < 2) {
            return std::string(layer_name) + " rope_fraction leaves fewer than two rotary features per head.";
        }
    } else if (configuration.rope_fraction != 1.0f) {
        return std::string(layer_name) + " rope_fraction applies to position_encoding=rope only.";
    }
    if (const std::string* text = FindNonEmpty(parameters, "attn_logit_softcap")) {
        const auto cap = ParseFiniteDouble(*text);
        if (!cap || *cap < 0.0) {
            return std::string(layer_name) + " attn_logit_softcap must be a finite value >= 0 (0 = off).";
        }
        configuration.attn_logit_softcap = static_cast<float>(*cap);
    }
    if (const auto error = ResolveNonNegativeInteger(parameters, layer_name, "sliding_window",
                                                     configuration.sliding_window)) {
        return error;
    }
    if (const auto error = ResolveNonNegativeInteger(parameters, layer_name, "num_kv_heads",
                                                     configuration.num_kv_heads)) {
        return error;
    }
    if (configuration.num_kv_heads > 0 &&
        (configuration.num_kv_heads > configuration.num_heads ||
         configuration.num_heads % configuration.num_kv_heads != 0)) {
        return std::string(layer_name) + " num_kv_heads must divide num_heads (0 = same as num_heads).";
    }
    return std::nullopt;
}

} // namespace transformer_configuration_policy_detail

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
        if (const std::string* type = FindNonEmpty(parameters, "encoding_type")) {
            if (*type != "sinusoidal" && *type != "learned") {
                return std::string("PositionalEncoding encoding_type must be sinusoidal or learned.");
            }
            configuration.encoding_type = *type;
        }
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
        // Attention options (tofix112): same meaning as on TransformerDecoder.
        if (const auto error = ResolveBoolParameter(parameters, layer_name, "causal", false,
                                                    configuration.causal)) {
            return error;
        }
        if (const auto error = ResolveBoolParameter(parameters, layer_name, "qk_norm", false,
                                                    configuration.qk_norm)) {
            return error;
        }
        configuration.position_encoding = "none";
        if (const std::string* text = FindNonEmpty(parameters, "position_encoding")) {
            if (*text != "none" && *text != "rope" && *text != "alibi") {
                return std::string("MultiHeadAttention position_encoding must be one of: none, rope, alibi.");
            }
            configuration.position_encoding = *text;
        }
        if (const auto error = ResolveAttentionOptions(parameters, layer_name, configuration)) {
            return error;
        }
        if (!configuration.causal &&
            (configuration.position_encoding == "alibi" || configuration.sliding_window > 0)) {
            return std::string("MultiHeadAttention alibi and sliding_window need causal=true.");
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
    if (const auto error = choose("position_encoding", {"external", "rope", "alibi"},
                                  configuration.position_encoding)) {
        return error;
    }
    if (const auto error = ResolveAttentionOptions(parameters, layer_name, configuration)) {
        return error;
    }
    if (const auto error = choose("block_layout", {"sequential", "parallel"}, configuration.block_layout)) {
        return error;
    }
    if (const auto error = ResolveBoolParameter(parameters, layer_name, "sandwich_norm", false,
                                                configuration.sandwich_norm)) {
        return error;
    }
    if ((configuration.block_layout == "parallel" || configuration.sandwich_norm) && !configuration.norm_first) {
        return std::string(layer_name) + " block_layout=parallel and sandwich_norm need norm_first=true (pre-norm).";
    }
    if (configuration.block_layout == "parallel" && configuration.sandwich_norm) {
        return std::string(layer_name) + " block_layout=parallel cannot be combined with sandwich_norm.";
    }
    if (const std::string* text = FindNonEmpty(parameters, "residual_init_scale")) {
        const auto scale = ParseFiniteDouble(*text);
        if (!scale || *scale <= 0.0 || *scale > 10.0) {
            return std::string(layer_name) + " residual_init_scale must be in (0,10].";
        }
        configuration.residual_init_scale = static_cast<float>(*scale);
    }
    if (const auto error = ResolveBoolParameter(parameters, layer_name, "attention_bias", true,
                                                configuration.attention_bias)) {
        return error;
    }
    if (const auto error = ResolveBoolParameter(parameters, layer_name, "qk_norm", false,
                                                configuration.qk_norm)) {
        return error;
    }
    if (const auto error = choose("architecture_preset", {"custom", "classic", "gpt2_style", "llama_style"},
                                  configuration.architecture_preset)) {
        return error;
    }
    if (configuration.architecture_preset != "custom") {
        // A saved preset must describe the block it builds (the Properties
        // panel switches to custom on any edit; this catches hand-edited files).
        const TransformerConfiguration& c = configuration;
        const std::map<std::string, std::string> resolved = {
            {"norm_first", c.norm_first ? "true" : "false"}, {"norm_type", c.norm_type},
            {"ffn_type", c.ffn_type}, {"ffn_activation", c.ffn_activation},
            {"ffn_bias", c.ffn_bias ? "true" : "false"}, {"position_encoding", c.position_encoding},
            {"attention_bias", c.attention_bias ? "true" : "false"}, {"qk_norm", c.qk_norm ? "true" : "false"},
            {"block_layout", c.block_layout}, {"sandwich_norm", c.sandwich_norm ? "true" : "false"}};
        for (const auto& [key, value] : TransformerArchitecturePresets().at(c.architecture_preset)) {
            if (resolved.at(key) != value) {
                return std::string(layer_name) + " architecture_preset=" + c.architecture_preset +
                    " expects " + key + "=" + value + " but the block has " + key + "=" +
                    resolved.at(key) + "; choose the preset again or set architecture_preset=custom.";
            }
        }
    }
    if (is_encoder && !configuration.HasClassicBlock()) {
        return std::string(layer_name) +
            " supports only the classic block (layer_norm, mlp, relu, ffn_bias=true, external positions, "
            "attention_bias=true, qk_norm=false) for now; the block options are implemented on TransformerDecoder.";
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

// Properties-panel side effect of editing `changed_key` on a TransformerDecoder:
// choosing a preset writes its fields; editing a block field while a preset is
// selected switches the preset to custom when the block no longer matches it.
// Returns true when other parameters were changed.
inline bool ApplyTransformerPresetEdit(gui::NodeType node_type,
                                       std::map<std::string, std::string>& parameters,
                                       const std::string& changed_key) {
    if (node_type != gui::NodeType::TransformerDecoder) return false;
    const auto preset_it = parameters.find("architecture_preset");
    const std::string preset = preset_it == parameters.end() ? "custom" : preset_it->second;
    const auto& presets = TransformerArchitecturePresets();
    const auto found = presets.find(preset);
    if (found == presets.end()) return false;  // custom or invalid (policy reports it)
    if (changed_key == "architecture_preset") {
        bool changed = false;
        for (const auto& [key, value] : found->second) {
            if (parameters[key] != value) {
                parameters[key] = value;
                changed = true;
            }
        }
        return changed;
    }
    if (found->second.count(changed_key) == 0) return false;
    TransformerConfiguration configuration;
    if (!ResolveTransformerConfiguration(node_type, parameters, configuration)) return false;  // still matches
    parameters["architecture_preset"] = "custom";
    return true;
}

inline std::optional<std::string> ResolveInvalidTransformerConfigurationReason(
    gui::NodeType node_type,
    const std::map<std::string, std::string>& parameters) {
    TransformerConfiguration configuration;
    return ResolveTransformerConfiguration(node_type, parameters,
                                           configuration);
}

} // namespace cyxwiz
