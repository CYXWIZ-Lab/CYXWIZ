#pragma once

#include "../data_registry.h"
#include <cyxwiz/time_series.h>
#include <string>
#include <vector>

namespace cyxwiz {

/**
 * Configuration for loading time-series data as a windowed ML dataset
 */
struct TimeSeriesDatasetConfig {
    // Data source
    std::string value_column = "value";       // Column containing the time series values
    std::string time_column = "";             // Optional time/date column (for ordering)
    std::vector<std::string> feature_columns; // Additional feature columns (multivariate)

    // Windowing
    int window_size = 10;            // Lookback window
    int forecast_horizon = 1;        // Predict ahead
    int stride = 1;                  // Step between windows

    // Feature engineering
    std::vector<int> lag_values;     // Lag features (e.g., {1, 7, 30})
    std::vector<int> rolling_windows; // Rolling stat windows (e.g., {7, 30})
    bool add_diff_features = false;  // Add first-difference

    // Split
    double train_ratio = 0.7;
    double val_ratio = 0.15;
    // test_ratio = 1.0 - train - val
};

/**
 * TimeSeriesDataset - Loads CSV/tabular time-series data and creates
 * sliding window samples for supervised ML training.
 *
 * Each sample is a (window, target) pair where:
 *   - window: flattened vector of [window_size * num_features] floats
 *   - target: encoded as integer class (for classification) or index (for regression)
 *
 * For regression tasks, use GetRegressionTarget() to get the float target.
 */
class TimeSeriesDataset : public Dataset {
public:
    /**
     * Load time-series data from a CSV file
     * @param filepath Path to CSV file
     * @param config Configuration for windowing and features
     */
    TimeSeriesDataset(const std::string& filepath, const TimeSeriesDatasetConfig& config);

    // Dataset interface
    size_t Size() const override;
    std::pair<std::vector<float>, int> GetItem(size_t index) const override;
    DatasetInfo GetInfo() const override;

    // Regression target access
    float GetRegressionTarget(size_t index) const;
    const std::vector<float>& GetAllTargets() const { return targets_; }

    // Column info
    std::vector<std::string> GetColumnNames() const override { return column_names_; }
    bool HasFloatLabels() const override { return true; }
    float GetFloatLabel(size_t index) const override { return GetRegressionTarget(index); }

    // Accessors
    int GetWindowSize() const { return config_.window_size; }
    int GetForecastHorizon() const { return config_.forecast_horizon; }
    int GetNumFeatures() const { return num_features_per_step_; }
    size_t GetNumWindows() const { return windows_.size(); }

private:
    TimeSeriesDatasetConfig config_;
    std::vector<std::vector<float>> windows_;  // [num_windows][window_size * features]
    std::vector<float> targets_;               // [num_windows] (first horizon step)
    std::vector<std::string> column_names_;
    int num_features_per_step_ = 1;
    size_t raw_data_points_ = 0;

    void LoadCSV(const std::string& filepath);
};

} // namespace cyxwiz
