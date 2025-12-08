#pragma once

#include <cyxwiz/time_series.h>
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>

namespace cyxwiz {

/**
 * Stationarity Testing Panel
 *
 * Tests time series for stationarity using statistical tests.
 *
 * Features:
 * - Augmented Dickey-Fuller (ADF) test
 * - KPSS test
 * - Rolling mean/std visualization
 * - Automatic differencing suggestion
 * - Before/after differencing comparison
 */
class StationarityPanel {
public:
    StationarityPanel();
    ~StationarityPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

private:
    void RenderToolbar();
    void RenderInputPanel();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderTimeSeriesPlot();
    void RenderRollingStats();
    void RenderTestResults();
    void RenderDifferencedPlot();

    void GenerateData();
    void TestAsync();
    void ApplyDifferencing();

    bool visible_ = false;

    // Data generation
    enum class SignalType { RandomWalk, TrendNoise, Stationary, AR1 };
    SignalType signal_type_ = SignalType::RandomWalk;

    int num_samples_ = 200;
    double noise_std_ = 1.0;
    double trend_slope_ = 0.05;

    // Test parameters
    int max_lags_ = -1;  // -1 = auto

    // Differencing
    int diff_order_ = 1;
    std::vector<double> differenced_data_;

    // Data
    std::vector<double> time_series_;
    std::vector<double> time_axis_;

    // Results
    StationarityResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Async
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;
};

} // namespace cyxwiz
