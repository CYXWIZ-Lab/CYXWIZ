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
 * Time Series Decomposition Panel
 *
 * Decomposes time series into trend, seasonal, and residual components.
 *
 * Features:
 * - Classical decomposition (additive/multiplicative)
 * - STL decomposition
 * - 4-panel plot visualization
 * - Component strength metrics
 * - Synthetic data generation
 */
class DecompositionPanel {
public:
    DecompositionPanel();
    ~DecompositionPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

private:
    void RenderToolbar();
    void RenderInputPanel();
    void RenderGenerateOptions();
    void RenderDecompositionOptions();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderOriginalPlot();
    void RenderTrendPlot();
    void RenderSeasonalPlot();
    void RenderResidualPlot();
    void RenderStatistics();

    void GenerateData();
    void DecomposeAsync();

    bool visible_ = false;

    // Data generation
    enum class DataType { Generate, Manual };
    DataType data_type_ = DataType::Generate;

    enum class SignalType {
        TrendSeasonal,
        RandomWalk,
        WhiteNoise,
        AR2,
        Seasonal
    };
    SignalType signal_type_ = SignalType::TrendSeasonal;

    // Generation parameters
    int num_samples_ = 200;
    double trend_slope_ = 0.1;
    double seasonal_amplitude_ = 1.5;
    int seasonal_period_ = 12;
    double noise_std_ = 0.3;

    // Decomposition parameters
    enum class DecompMethod { Classical, STL };
    DecompMethod decomp_method_ = DecompMethod::Classical;
    std::string seasonal_type_ = "additive";  // "additive" or "multiplicative"
    int period_ = 12;
    int stl_seasonal_window_ = 7;
    int stl_trend_window_ = -1;  // -1 = auto

    // Data
    std::vector<double> time_series_;
    std::vector<double> time_axis_;

    // Results
    DecompositionResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;
};

} // namespace cyxwiz
