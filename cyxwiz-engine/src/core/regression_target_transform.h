#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

// Declares the fitted transform used by a continuous training target. The
// materializer applies the forward transform; training and Test use this
// contract to report regression metrics in the original target units.
struct RegressionTargetTransform {
    bool enabled = false;
    bool resolved = false;
    int node_id = -1;
    std::string node_name;
    std::string operator_name;
    std::string state_path;
    std::vector<std::string> target_columns;
    std::vector<double> offsets;
    std::vector<double> scales;

    bool IsResolvedForWidth(size_t width) const {
        return !enabled ||
               (resolved && width > 0 && target_columns.size() == width &&
                offsets.size() == width && scales.size() == width);
    }

    double InverseValue(double value, size_t horizon) const {
        if (!enabled || !resolved || horizon >= scales.size()) {
            return value;
        }
        return value * scales[horizon] + offsets[horizon];
    }
};

// Loads and validates the fitted preprocessing state named by the compiled
// contract. Disabled contracts are accepted without I/O.
bool ResolveRegressionTargetTransform(
    RegressionTargetTransform& transform,
    std::string& error);

class RegressionMetricAccumulator {
public:
    explicit RegressionMetricAccumulator(
        const RegressionTargetTransform* transform = nullptr)
        : transform_(transform) {}

    void SetTargetTransform(const RegressionTargetTransform* transform) {
        transform_ = transform;
    }

    void Reset() {
        absolute_error_sum = 0.0;
        squared_error_sum = 0.0;
        value_count = 0;
    }

    void Add(const float* predictions,
             const float* targets,
             size_t count,
             size_t output_width = 0) {
        if (!predictions || !targets) return;

        const bool restore_original_units =
            transform_ && transform_->enabled && transform_->resolved &&
            output_width > 0 && transform_->scales.size() == output_width;
        for (size_t i = 0; i < count; ++i) {
            double error = static_cast<double>(predictions[i]) -
                           static_cast<double>(targets[i]);
            if (restore_original_units) {
                error *= transform_->scales[i % output_width];
            }
            absolute_error_sum += std::abs(error);
            squared_error_sum += error * error;
        }
        value_count += count;
    }

    float Mae() const {
        return value_count == 0
            ? 0.0f
            : static_cast<float>(
                  absolute_error_sum / static_cast<double>(value_count));
    }

    float Rmse() const {
        return value_count == 0
            ? 0.0f
            : static_cast<float>(std::sqrt(
                  squared_error_sum / static_cast<double>(value_count)));
    }

    double absolute_error_sum = 0.0;
    double squared_error_sum = 0.0;
    size_t value_count = 0;

private:
    const RegressionTargetTransform* transform_ = nullptr;
};

}  // namespace cyxwiz
