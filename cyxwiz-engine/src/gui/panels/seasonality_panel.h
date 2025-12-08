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
 * Seasonality Detection Panel
 *
 * Detects and analyzes seasonal patterns in time series.
 *
 * Features:
 * - Periodogram (spectral density)
 * - Peak frequency detection
 * - Multiple seasonality support
 * - Strength measurement
 * - ACF-based confirmation
 */
class SeasonalityPanel {
public:
    SeasonalityPanel();
    ~SeasonalityPanel();

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
    void RenderPeriodogramPlot();
    void RenderACFPlot();
    void RenderSummary();

    void GenerateData();
    void DetectAsync();

    bool visible_ = false;

    // Data generation
    enum class SignalType { Seasonal, MultiSeasonal, TrendSeasonal, NoSeason };
    SignalType signal_type_ = SignalType::Seasonal;

    int num_samples_ = 200;
    double noise_std_ = 0.3;
    int primary_period_ = 12;
    int secondary_period_ = 4;
    double seasonal_amplitude_ = 1.5;

    // Detection parameters
    int min_period_ = 2;
    int max_period_ = -1;  // -1 = auto

    // Data
    std::vector<double> time_series_;
    std::vector<double> time_axis_;

    // Results
    SeasonalityResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Async
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;
};

} // namespace cyxwiz
