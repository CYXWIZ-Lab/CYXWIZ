#include "regression_target_transform.h"

#include "preprocessing_state.h"

#include <cmath>

namespace cyxwiz {

bool ResolveRegressionTargetTransform(
    RegressionTargetTransform& transform,
    std::string& error) {
    error.clear();
    transform.resolved = false;
    transform.offsets.clear();
    transform.scales.clear();

    if (!transform.enabled) {
        return true;
    }
    if (transform.operator_name != "StandardScaler") {
        error = "unsupported regression target transform operator '" +
                transform.operator_name + "'";
        return false;
    }
    if (transform.target_columns.empty()) {
        error = "regression target transform has no target columns";
        return false;
    }

    FittedPreprocessingState state;
    if (!LoadFittedPreprocessingState(
            transform.state_path, transform.operator_name, state, error)) {
        return false;
    }
    if (state.features.size() != transform.target_columns.size()) {
        error = "regression target state contains " +
                std::to_string(state.features.size()) +
                " column(s), but the compiled target width is " +
                std::to_string(transform.target_columns.size());
        return false;
    }

    transform.offsets.reserve(state.features.size());
    transform.scales.reserve(state.features.size());
    for (size_t i = 0; i < state.features.size(); ++i) {
        const auto& feature = state.features[i];
        const auto& expected = transform.target_columns[i];
        if (feature.name != expected) {
            error = "regression target state column " + std::to_string(i) +
                    " is '" + feature.name + "', expected '" + expected +
                    "'. Refit the artifact with the exact ordered target columns.";
            return false;
        }
        const auto mean = feature.numeric_values.find("mean");
        const auto scale = feature.numeric_values.find("scale");
        if (mean == feature.numeric_values.end() ||
            scale == feature.numeric_values.end() ||
            !std::isfinite(mean->second) || !std::isfinite(scale->second) ||
            scale->second <= 0.0) {
            error = "regression target state for column '" + feature.name +
                    "' has invalid mean/scale values";
            return false;
        }
        transform.offsets.push_back(mean->second);
        transform.scales.push_back(scale->second);
    }

    transform.resolved = true;
    return true;
}

}  // namespace cyxwiz
