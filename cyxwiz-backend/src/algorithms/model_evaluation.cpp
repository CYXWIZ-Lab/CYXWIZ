// Prevent Windows min/max macros from interfering with std::numeric_limits and af::max/min
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "cyxwiz/model_evaluation.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>
#include <set>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

// Ensure Windows min/max macros are undefined after all includes
#ifdef _WIN32
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

namespace cyxwiz {

RegressionMetrics ModelEvaluation::ComputeRegressionMetrics(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred) {
    RegressionMetrics result;
    if (y_true.empty() || y_true.size() != y_pred.size()) {
        result.error_message =
            "Regression metrics require non-empty arrays of equal length";
        return result;
    }

    double squared_error_sum = 0.0;
    double absolute_error_sum = 0.0;
    double absolute_percentage_error_sum = 0.0;
    double maximum_error = 0.0;
    double target_sum = 0.0;
    constexpr double kMapeEpsilon =
        std::numeric_limits<double>::epsilon();

    for (size_t index = 0; index < y_true.size(); ++index) {
        const double target = y_true[index];
        const double prediction = y_pred[index];
        if (!std::isfinite(target) || !std::isfinite(prediction)) {
            result.error_message =
                "Regression metrics require finite targets and predictions";
            return result;
        }
        const double error = target - prediction;
        const double absolute_error = std::abs(error);
        squared_error_sum += error * error;
        absolute_error_sum += absolute_error;
        maximum_error = std::max(maximum_error, absolute_error);
        absolute_percentage_error_sum +=
            absolute_error / std::max(std::abs(target), kMapeEpsilon);
        target_sum += target;
    }

    const double sample_count = static_cast<double>(y_true.size());
    result.mse = squared_error_sum / sample_count;
    result.rmse = std::sqrt(result.mse);
    result.mae = absolute_error_sum / sample_count;
    result.mape = absolute_percentage_error_sum / sample_count;
    result.max_error = maximum_error;

    if (y_true.size() < 2) {
        result.r_squared = std::numeric_limits<double>::quiet_NaN();
    } else {
        const double target_mean = target_sum / sample_count;
        double total_square_sum = 0.0;
        for (const double target : y_true) {
            const double centered = target - target_mean;
            total_square_sum += centered * centered;
        }
        if (total_square_sum == 0.0) {
            result.r_squared = squared_error_sum == 0.0 ? 1.0 : 0.0;
        } else {
            result.r_squared = 1.0 - squared_error_sum / total_square_sum;
        }
    }

    result.success = true;
    return result;
}

} // namespace cyxwiz



