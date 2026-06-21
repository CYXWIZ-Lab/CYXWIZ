// Prevent Windows min/max macros from interfering with std::min/std::max
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <cyxwiz/time_series.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

// Windowing for ML
// ============================================================================

TimeSeries::WindowResult TimeSeries::CreateWindows(
    const std::vector<double>& data,
    const WindowConfig& config) {

    WindowResult result;

    int n = static_cast<int>(data.size());
    int required = config.window_size + config.forecast_horizon;

    if (n < required) {
        result.error_message = "Data too short: need at least " + std::to_string(required) +
                               " samples, got " + std::to_string(n);
        return result;
    }

    // Optionally add engineered features
    std::vector<std::vector<double>> features;
    if (!config.lag_values.empty() || !config.rolling_windows.empty() || config.add_diff_features) {
        features = AddFeatures(data, config.lag_values, config.rolling_windows, config.add_diff_features);
    } else {
        // Single feature: just the raw data
        features.resize(n);
        for (int i = 0; i < n; i++) {
            features[i] = {data[i]};
        }
    }

    int feat_n = static_cast<int>(features.size());
    int num_features = static_cast<int>(features[0].size());

    // Create windows
    for (int i = 0; i + config.window_size + config.forecast_horizon - 1 < feat_n; i += config.stride) {
        // Input: window_size timesteps * num_features
        std::vector<double> x;
        x.reserve(config.window_size * num_features);
        for (int t = i; t < i + config.window_size; t++) {
            for (int f = 0; f < num_features; f++) {
                x.push_back(features[t][f]);
            }
        }

        // Target: forecast_horizon values from the original data (first feature = original)
        std::vector<double> y;
        y.reserve(config.forecast_horizon);
        for (int h = 0; h < config.forecast_horizon; h++) {
            y.push_back(features[i + config.window_size + h][0]);
        }

        result.X.push_back(std::move(x));
        result.y.push_back(std::move(y));
    }

    result.num_windows = result.X.size();
    result.input_features = config.window_size * num_features;
    result.target_features = config.forecast_horizon;
    result.success = !result.X.empty();

    if (!result.success) {
        result.error_message = "No windows could be created with given parameters";
    }

    return result;
}

TimeSeries::WindowResult TimeSeries::CreateMultivariateWindows(
    const std::vector<std::vector<double>>& data,
    int target_col,
    const WindowConfig& config) {

    WindowResult result;

    int n = static_cast<int>(data.size());
    if (n == 0) {
        result.error_message = "Empty data";
        return result;
    }

    int num_features = static_cast<int>(data[0].size());
    if (target_col < 0 || target_col >= num_features) {
        result.error_message = "Invalid target column: " + std::to_string(target_col);
        return result;
    }

    int required = config.window_size + config.forecast_horizon;
    if (n < required) {
        result.error_message = "Data too short: need " + std::to_string(required) + ", got " + std::to_string(n);
        return result;
    }

    for (int i = 0; i + config.window_size + config.forecast_horizon - 1 < n; i += config.stride) {
        std::vector<double> x;
        x.reserve(config.window_size * num_features);
        for (int t = i; t < i + config.window_size; t++) {
            for (int f = 0; f < num_features; f++) {
                x.push_back(data[t][f]);
            }
        }

        std::vector<double> y;
        y.reserve(config.forecast_horizon);
        for (int h = 0; h < config.forecast_horizon; h++) {
            y.push_back(data[i + config.window_size + h][target_col]);
        }

        result.X.push_back(std::move(x));
        result.y.push_back(std::move(y));
    }

    result.num_windows = result.X.size();
    result.input_features = config.window_size * num_features;
    result.target_features = config.forecast_horizon;
    result.success = !result.X.empty();

    return result;
}

std::vector<std::vector<double>> TimeSeries::AddFeatures(
    const std::vector<double>& data,
    const std::vector<int>& lag_values,
    const std::vector<int>& rolling_windows,
    bool add_diff) {

    int n = static_cast<int>(data.size());

    // Determine the maximum lookback needed
    int max_lookback = 0;
    for (int lag : lag_values) max_lookback = std::max(max_lookback, lag);
    for (int w : rolling_windows) max_lookback = std::max(max_lookback, w);
    if (add_diff && max_lookback < 1) max_lookback = 1;

    // Build feature matrix starting from max_lookback
    int valid_start = max_lookback;
    int valid_n = n - valid_start;

    if (valid_n <= 0) {
        return {{data.back()}};
    }

    std::vector<std::vector<double>> result(valid_n);

    // Pre-compute rolling stats
    std::vector<std::vector<double>> roll_means, roll_stds;
    for (int w : rolling_windows) {
        roll_means.push_back(RollingMean(data, w));
        roll_stds.push_back(RollingStd(data, w));
    }

    for (int i = 0; i < valid_n; i++) {
        int idx = valid_start + i;
        auto& row = result[i];

        // Original value
        row.push_back(data[idx]);

        // Lag features
        for (int lag : lag_values) {
            row.push_back(data[idx - lag]);
        }

        // Rolling mean/std features
        for (size_t ri = 0; ri < rolling_windows.size(); ri++) {
            int w = rolling_windows[ri];
            // rolling_mean has length n - w + 1, index maps to data[w-1 + j]
            int rm_idx = idx - w + 1;
            if (rm_idx >= 0 && rm_idx < static_cast<int>(roll_means[ri].size())) {
                row.push_back(roll_means[ri][rm_idx]);
                row.push_back(roll_stds[ri][rm_idx]);
            } else {
                row.push_back(data[idx]);
                row.push_back(0.0);
            }
        }

        // Difference feature
        if (add_diff) {
            row.push_back(data[idx] - data[idx - 1]);
        }
    }

    return result;
}

std::pair<size_t, size_t> TimeSeries::ChronologicalSplit(
    size_t num_samples, double train_ratio, double val_ratio) {

    // Need at least 3 samples for train/val/test
    if (num_samples < 3) {
        // Degenerate case: put everything in train
        return {num_samples, num_samples};
    }

    size_t train_end = static_cast<size_t>(num_samples * train_ratio);
    size_t val_end = train_end + static_cast<size_t>(num_samples * val_ratio);

    // Ensure at least 1 sample in each split
    if (train_end == 0) train_end = 1;
    if (val_end <= train_end) val_end = train_end + 1;
    if (val_end >= num_samples) val_end = num_samples > 1 ? num_samples - 1 : num_samples;

    return {train_end, val_end};
}


} // namespace cyxwiz
