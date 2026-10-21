#include "debug_gradient_health_analyzer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace cyxwiz {

namespace {

constexpr double kNormDenominatorFloor = 1.0e-12;
constexpr double kZeroUpdateNorm = 1.0e-12;

std::optional<double> FiniteNonNegativeNumber(
    const nlohmann::json& payload,
    const char* key) {
    const auto it = payload.find(key);
    if (it == payload.end() || !it->is_number()) {
        return std::nullopt;
    }
    const double value = it->get<double>();
    if (!std::isfinite(value) || value < 0.0) {
        return std::nullopt;
    }
    return value;
}

bool PayloadBool(const nlohmann::json& payload,
                 const char* key,
                 bool fallback = false) {
    const auto it = payload.find(key);
    return it != payload.end() && it->is_boolean()
        ? it->get<bool>()
        : fallback;
}

std::string PayloadString(const nlohmann::json& payload,
                          const char* key) {
    const auto it = payload.find(key);
    return it != payload.end() && it->is_string()
        ? it->get<std::string>()
        : std::string{};
}

int PayloadInt(const nlohmann::json& payload,
               const char* key,
               int fallback = -1) {
    const auto it = payload.find(key);
    if (it == payload.end() ||
        (!it->is_number_integer() && !it->is_number_unsigned())) {
        return fallback;
    }
    if (it->is_number_unsigned()) {
        const auto value = it->get<uint64_t>();
        return value <= static_cast<uint64_t>(
                    std::numeric_limits<int>::max())
            ? static_cast<int>(value)
            : fallback;
    }
    const auto value = it->get<int64_t>();
    if (value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        return fallback;
    }
    return static_cast<int>(value);
}

std::string ParameterLayerPrefix(const std::string& parameter_name) {
    const size_t dot = parameter_name.rfind('.');
    return dot == std::string::npos
        ? parameter_name
        : parameter_name.substr(0, dot);
}

std::string LayerKey(const DebugTraceRecord& trace,
                     const std::string& parameter_name,
                     int module_index,
                     int compiled_layer_index) {
    if (trace.node_id >= 0) {
        return "node:" + std::to_string(trace.node_id);
    }
    if (module_index >= 0) {
        return "module:" + std::to_string(module_index);
    }
    if (compiled_layer_index >= 0) {
        return "compiled:" + std::to_string(compiled_layer_index);
    }
    const std::string prefix = ParameterLayerPrefix(parameter_name);
    return prefix.empty() ? "unresolved:" + parameter_name
                          : "parameter_prefix:" + prefix;
}

struct NormAccumulator {
    long double square_sum = 0.0L;
    size_t observed_count = 0;
    bool overflowed = false;

    void Add(double value) {
        const long double wide = static_cast<long double>(value);
        const long double square = wide * wide;
        if (!std::isfinite(square) ||
            square_sum > std::numeric_limits<long double>::max() - square) {
            overflowed = true;
            return;
        }
        square_sum += square;
        ++observed_count;
    }

    std::optional<double> Norm() const {
        if (observed_count == 0 || overflowed) {
            return std::nullopt;
        }
        const long double value = std::sqrt(square_sum);
        if (!std::isfinite(value) ||
            value > std::numeric_limits<double>::max()) {
            return std::nullopt;
        }
        return static_cast<double>(value);
    }
};

struct LayerAccumulator {
    std::string key;
    int node_id = -1;
    std::string node_name;
    int module_index = -1;
    int compiled_layer_index = -1;
    size_t parameter_count = 0;
    size_t gradient_count = 0;
    size_t missing_gradient_count = 0;
    size_t zero_gradient_count = 0;
    size_t update_count = 0;
    bool has_nan = false;
    bool has_inf = false;
    NormAccumulator parameter_norm;
    NormAccumulator gradient_norm;
    NormAccumulator update_norm;
    nlohmann::json parameter_names = nlohmann::json::array();
    nlohmann::json missing_reasons = nlohmann::json::array();
};

void AddName(LayerAccumulator& layer, const std::string& parameter_name) {
    if (layer.parameter_names.size() <
        DebugGradientHealthAnalyzer::kMaxParameterNamesPerLayer) {
        layer.parameter_names.push_back(parameter_name);
    }
}

void AddMissingReason(LayerAccumulator& layer,
                      const std::string& parameter_name,
                      const std::string& reason) {
    if (layer.missing_reasons.size() <
        DebugGradientHealthAnalyzer::kMaxMissingReasonsPerLayer) {
        layer.missing_reasons.push_back({
            {"parameter_name", parameter_name},
            {"reason", reason.empty()
                ? "No matching gradient tensor was observed after backward."
                : reason},
        });
    }
}

std::string LayerStatus(const LayerAccumulator& layer,
                        const std::optional<double>& update_norm,
                        bool parameter_complete,
                        bool gradient_complete,
                        bool update_complete) {
    if (layer.has_nan || layer.has_inf) {
        return "non_finite";
    }
    if (layer.missing_gradient_count > 0) {
        return "missing_gradient";
    }
    if (layer.gradient_count == 0) {
        return "unobserved";
    }
    if (layer.zero_gradient_count == layer.gradient_count) {
        return "zero_gradient";
    }
    if (layer.zero_gradient_count > 0) {
        return "partial_zero_gradient";
    }
    if (!parameter_complete || !gradient_complete || !update_complete) {
        return "partial_evidence";
    }
    if (update_norm && *update_norm <= kZeroUpdateNorm) {
        return "zero_update";
    }
    return "healthy";
}

} // namespace

DebugTraceRecord DebugGradientHealthAnalyzer::BuildTrace(
    const std::string& run_id,
    const std::vector<DebugTraceRecord>& traces) const {
    std::vector<LayerAccumulator> layers;
    std::unordered_map<std::string, size_t> layer_indices;
    size_t parameter_trace_count = 0;

    for (const auto& source : traces) {
        if (source.run_id != run_id ||
            source.phase != "Backward" ||
            source.role != DebugTraceRole::Gradient) {
            continue;
        }
        const std::string parameter_name =
            PayloadString(source.payload, "parameter_name");
        if (parameter_name.empty()) {
            continue;
        }
        ++parameter_trace_count;

        const int module_index = PayloadInt(source.payload, "module_index");
        const int compiled_layer_index = PayloadInt(
            source.payload, "compiled_layer_index");
        const std::string key = LayerKey(
            source, parameter_name, module_index, compiled_layer_index);

        auto [it, inserted] = layer_indices.emplace(key, layers.size());
        if (inserted) {
            LayerAccumulator layer;
            layer.key = key;
            layer.node_id = source.node_id;
            layer.node_name = source.node_name;
            layer.module_index = module_index;
            layer.compiled_layer_index = compiled_layer_index;
            layers.push_back(std::move(layer));
        }
        LayerAccumulator& layer = layers[it->second];
        ++layer.parameter_count;
        AddName(layer, parameter_name);

        if (const auto value = FiniteNonNegativeNumber(
                source.payload, "parameter_l2_norm")) {
            layer.parameter_norm.Add(*value);
        }

        const bool has_gradient = PayloadBool(
            source.payload, "has_gradient", false);
        if (!has_gradient) {
            ++layer.missing_gradient_count;
            AddMissingReason(
                layer,
                parameter_name,
                PayloadString(source.payload, "missing_gradient_reason"));
        } else {
            ++layer.gradient_count;
            if (PayloadBool(source.payload, "is_zero", false)) {
                ++layer.zero_gradient_count;
            }
            layer.has_nan = layer.has_nan ||
                PayloadBool(source.payload, "is_nan", false);
            layer.has_inf = layer.has_inf ||
                PayloadBool(source.payload, "is_inf", false);
            auto gradient_norm = FiniteNonNegativeNumber(
                source.payload, "gradient_l2_norm");
            if (!gradient_norm) {
                gradient_norm = FiniteNonNegativeNumber(
                    source.payload, "l2_norm");
            }
            if (gradient_norm) {
                layer.gradient_norm.Add(*gradient_norm);
            }
        }

        if (PayloadBool(source.payload, "update_observed", false)) {
            if (const auto value = FiniteNonNegativeNumber(
                    source.payload, "update_l2_norm")) {
                layer.update_norm.Add(*value);
                ++layer.update_count;
            }
        }
    }

    nlohmann::json rows = nlohmann::json::array();
    size_t healthy_count = 0;
    size_t attention_count = 0;
    size_t unobserved_count = 0;
    size_t retained_count = 0;

    for (const auto& layer : layers) {
        const auto parameter_norm = layer.parameter_norm.Norm();
        const auto gradient_norm = layer.gradient_norm.Norm();
        const auto update_norm = layer.update_norm.Norm();
        const bool parameter_complete =
            layer.parameter_norm.observed_count == layer.parameter_count &&
            parameter_norm.has_value();
        const bool gradient_complete =
            layer.missing_gradient_count == 0 &&
            layer.gradient_norm.observed_count == layer.gradient_count &&
            gradient_norm.has_value();
        const bool update_complete =
            layer.update_count == layer.parameter_count &&
            update_norm.has_value();
        const bool ratio_observed = parameter_norm && gradient_norm;
        const bool update_ratio_observed = parameter_norm && update_norm;
        const std::string status = LayerStatus(
            layer,
            update_norm,
            parameter_complete,
            gradient_complete,
            update_complete);

        if (status == "healthy") {
            ++healthy_count;
        } else if (status == "unobserved" ||
                   status == "partial_evidence") {
            ++unobserved_count;
        } else {
            ++attention_count;
        }

        if (retained_count >= kMaxLayerRows) {
            continue;
        }
        nlohmann::json row = {
            {"layer_key", layer.key},
            {"node_id", layer.node_id},
            {"node_name", layer.node_name},
            {"module_index", layer.module_index},
            {"compiled_layer_index", layer.compiled_layer_index},
            {"status", status},
            {"parameter_tensor_count", layer.parameter_count},
            {"gradient_tensor_count", layer.gradient_count},
            {"missing_gradient_count", layer.missing_gradient_count},
            {"zero_gradient_count", layer.zero_gradient_count},
            {"update_observation_count", layer.update_count},
            {"has_nan", layer.has_nan},
            {"has_inf", layer.has_inf},
            {"all_observed_gradients_zero",
             layer.gradient_count > 0 &&
                 layer.zero_gradient_count == layer.gradient_count},
            {"some_gradients_zero", layer.zero_gradient_count > 0},
            {"parameter_norm_complete", parameter_complete},
            {"gradient_norm_complete", gradient_complete},
            {"update_norm_complete", update_complete},
            {"parameter_names", layer.parameter_names},
            {"parameter_names_truncated",
             layer.parameter_count > layer.parameter_names.size()},
            {"missing_gradient_explanations", layer.missing_reasons},
            {"missing_explanations_truncated",
             layer.missing_gradient_count > layer.missing_reasons.size()},
        };
        if (parameter_norm) {
            row["parameter_l2_norm"] = *parameter_norm;
        }
        if (gradient_norm) {
            row["gradient_l2_norm"] = *gradient_norm;
        }
        if (update_norm) {
            row["update_l2_norm"] = *update_norm;
        }
        row["grad_parameter_ratio_observed"] = ratio_observed;
        row["grad_parameter_ratio_complete"] = ratio_observed &&
            parameter_complete && gradient_complete;
        if (ratio_observed) {
            row["grad_parameter_ratio"] = *gradient_norm /
                std::max(std::abs(*parameter_norm), kNormDenominatorFloor);
        }
        row["update_parameter_ratio_observed"] = update_ratio_observed;
        row["update_parameter_ratio_complete"] = update_ratio_observed &&
            parameter_complete && update_complete;
        if (update_ratio_observed) {
            row["update_parameter_ratio"] = *update_norm /
                std::max(std::abs(*parameter_norm), kNormDenominatorFloor);
        }
        row["zero_update_observed"] = update_complete && update_norm &&
            *update_norm <= kZeroUpdateNorm;
        rows.push_back(std::move(row));
        ++retained_count;
    }

    const bool observed = !layers.empty();
    const bool success = observed && attention_count == 0 &&
        unobserved_count == 0;
    DebugTraceRecord result = DebugNodeTraceContract::Make(
        run_id,
        -1,
        "Gradient Health",
        "TrainingDiagnostics",
        "GradientHealth",
        DebugTraceRole::Gradient,
        {}, {}, "gradient_metadata", "canonical_debug_evidence",
        !observed ? "unobserved"
                  : (attention_count > 0
                      ? "needs_attention"
                      : (unobserved_count > 0
                          ? "partial_evidence"
                          : "captured")));
    auto& payload = result.payload;
    payload["gradient_health_schema"] = kSchema;
    payload["trace_producer"] = "DebugGradientHealthAnalyzer";
    payload["observation_scope"] =
        "derived_from_local_debug_parameter_gradient_traces";
    payload["layer_count"] = layers.size();
    payload["retained_layer_count"] = retained_count;
    payload["layer_limit"] = kMaxLayerRows;
    payload["layers_truncated"] = layers.size() > retained_count;
    payload["parameter_trace_count"] = parameter_trace_count;
    payload["healthy_layer_count"] = healthy_count;
    payload["attention_layer_count"] = attention_count;
    payload["unobserved_layer_count"] = unobserved_count;
    payload["has_attention"] = attention_count > 0;
    if (observed) {
        payload["success"] = success;
    }
    payload["relative_norm_denominator_floor"] = kNormDenominatorFloor;
    payload["zero_update_norm_threshold"] = kZeroUpdateNorm;
    payload["tensor_reads_added"] = false;
    payload["raw_tensor_values_included"] = false;
    payload["layers"] = std::move(rows);
    payload["scope_note"] =
        "Layer metrics aggregate bounded Local Debug scalar evidence. They "
        "describe one synthetic debug step, not a long-run learning trend.";
    DebugNodeTraceContract::AttachDiagnosticContext(
        result,
        "gradient_health",
        "DebugGradientHealthAnalyzer",
        "cyxwiz-engine/src/core/debug_gradient_health_analyzer.cpp",
        "cyxwiz::DebugGradientHealthAnalyzer::BuildTrace");
    return result;
}

} // namespace cyxwiz
