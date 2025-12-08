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
 * ACF/PACF (Correlogram) Panel
 *
 * Computes and visualizes autocorrelation and partial autocorrelation.
 *
 * Features:
 * - ACF computation with confidence bounds
 * - PACF using Durbin-Levinson recursion
 * - Significance testing
 * - Model order suggestions (AR, MA)
 * - Ljung-Box test for white noise
 */
class ACFPACFPanel {
public:
    ACFPACFPanel();
    ~ACFPACFPanel();

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
    void RenderACFPlot();
    void RenderPACFPlot();
    void RenderBothPlot();
    void RenderAnalysis();

    void GenerateData();
    void ComputeAsync();

    bool visible_ = false;

    // Data generation
    enum class SignalType { AR1, AR2, MA1, MA2, ARMA, WhiteNoise, RandomWalk };
    SignalType signal_type_ = SignalType::AR2;

    int num_samples_ = 200;
    double noise_std_ = 1.0;
    double ar1_coeff_ = 0.7;
    double ar2_coeff_ = -0.2;
    double ma1_coeff_ = 0.5;

    // Computation parameters
    int max_lag_ = 40;
    double confidence_level_ = 0.95;

    // Data
    std::vector<double> time_series_;
    std::vector<double> time_axis_;

    // Results
    AutocorrelationResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Async
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;
};

} // namespace cyxwiz
