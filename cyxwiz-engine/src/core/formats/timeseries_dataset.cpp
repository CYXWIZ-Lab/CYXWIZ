#include "timeseries_dataset.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace cyxwiz {

TimeSeriesDataset::TimeSeriesDataset(const std::string& filepath,
                                     const TimeSeriesDatasetConfig& config)
    : config_(config) {
    LoadCSV(filepath);
}

void TimeSeriesDataset::LoadCSV(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("TimeSeriesDataset: cannot open file: " + filepath);
    }

    // Read header
    std::string header_line;
    std::getline(file, header_line);

    // Parse column names
    std::vector<std::string> headers;
    {
        std::istringstream ss(header_line);
        std::string col;
        while (std::getline(ss, col, ',')) {
            // Trim whitespace
            col.erase(0, col.find_first_not_of(" \t\r\n"));
            col.erase(col.find_last_not_of(" \t\r\n") + 1);
            headers.push_back(col);
        }
    }
    column_names_ = headers;

    // Find value column index
    int value_idx = -1;
    for (int i = 0; i < static_cast<int>(headers.size()); i++) {
        if (headers[i] == config_.value_column) {
            value_idx = i;
            break;
        }
    }

    // If not found by name, try last numeric column
    if (value_idx < 0) {
        value_idx = static_cast<int>(headers.size()) - 1;
        spdlog::warn("TimeSeriesDataset: column '{}' not found, using column {} ('{}')",
                     config_.value_column, value_idx, headers[value_idx]);
    }

    // Find additional feature column indices
    std::vector<int> feature_indices;
    for (const auto& feat_col : config_.feature_columns) {
        for (int i = 0; i < static_cast<int>(headers.size()); i++) {
            if (headers[i] == feat_col) {
                feature_indices.push_back(i);
                break;
            }
        }
    }

    // Read all data rows
    std::vector<double> values;
    std::vector<std::vector<double>> multivariate_data;
    bool is_multivariate = !feature_indices.empty();

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(ss, cell, ',')) {
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
            cells.push_back(cell);
        }

        if (value_idx >= static_cast<int>(cells.size())) continue;

        try {
            double val = std::stod(cells[value_idx]);
            values.push_back(val);

            if (is_multivariate) {
                std::vector<double> row;
                row.push_back(val);  // Target column first
                for (int fi : feature_indices) {
                    if (fi < static_cast<int>(cells.size())) {
                        row.push_back(std::stod(cells[fi]));
                    } else {
                        row.push_back(0.0);
                    }
                }
                multivariate_data.push_back(std::move(row));
            }
        } catch (...) {
            continue;  // Skip non-numeric rows
        }
    }

    raw_data_points_ = values.size();
    spdlog::info("TimeSeriesDataset: loaded {} data points from '{}'", raw_data_points_, filepath);

    if (values.empty()) {
        throw std::runtime_error("TimeSeriesDataset: no valid numeric data found");
    }

    // Create windows using backend TimeSeries API
    TimeSeries::WindowConfig wconfig;
    wconfig.window_size = config_.window_size;
    wconfig.forecast_horizon = config_.forecast_horizon;
    wconfig.stride = config_.stride;
    wconfig.lag_values = config_.lag_values;
    wconfig.rolling_windows = config_.rolling_windows;
    wconfig.add_diff_features = config_.add_diff_features;

    TimeSeries::WindowResult wresult;
    if (is_multivariate) {
        wresult = TimeSeries::CreateMultivariateWindows(multivariate_data, 0, wconfig);
    } else {
        wresult = TimeSeries::CreateWindows(values, wconfig);
    }

    if (!wresult.success) {
        throw std::runtime_error("TimeSeriesDataset: windowing failed: " + wresult.error_message);
    }

    // Convert to float storage
    windows_.resize(wresult.num_windows);
    targets_.resize(wresult.num_windows);

    for (size_t i = 0; i < wresult.num_windows; i++) {
        windows_[i].resize(wresult.X[i].size());
        for (size_t j = 0; j < wresult.X[i].size(); j++) {
            windows_[i][j] = static_cast<float>(wresult.X[i][j]);
        }
        targets_[i] = static_cast<float>(wresult.y[i][0]);
    }

    num_features_per_step_ = static_cast<int>(wresult.input_features / config_.window_size);

    // Apply chronological split
    auto [train_end, val_end] = TimeSeries::ChronologicalSplit(
        wresult.num_windows, config_.train_ratio, config_.val_ratio);

    all_indices_.resize(wresult.num_windows);
    for (size_t i = 0; i < wresult.num_windows; i++) all_indices_[i] = i;

    train_indices_.assign(all_indices_.begin(), all_indices_.begin() + train_end);
    val_indices_.assign(all_indices_.begin() + train_end, all_indices_.begin() + val_end);
    test_indices_.assign(all_indices_.begin() + val_end, all_indices_.end());

    spdlog::info("TimeSeriesDataset: {} windows (train: {}, val: {}, test: {}), {} features/step",
                 wresult.num_windows, train_indices_.size(), val_indices_.size(),
                 test_indices_.size(), num_features_per_step_);
}

size_t TimeSeriesDataset::Size() const {
    return windows_.size();
}

std::pair<std::vector<float>, int> TimeSeriesDataset::GetItem(size_t index) const {
    if (index >= windows_.size()) {
        return {{}, -1};
    }
    // For regression, label is quantized target (bucket index). Use GetRegressionTarget for float.
    return {windows_[index], static_cast<int>(index)};
}

float TimeSeriesDataset::GetRegressionTarget(size_t index) const {
    if (index >= targets_.size()) return 0.0f;
    return targets_[index];
}

DatasetInfo TimeSeriesDataset::GetInfo() const {
    DatasetInfo info;
    info.name = "TimeSeries";
    info.num_samples = windows_.size();
    size_t feat_size = windows_.empty() ? 0 : windows_[0].size();
    info.shape = {feat_size};
    info.num_classes = 0;  // Regression task
    return info;
}

} // namespace cyxwiz
