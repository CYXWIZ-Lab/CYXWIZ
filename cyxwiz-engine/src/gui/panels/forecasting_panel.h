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
 * Forecasting Panel
 *
 * Time series forecasting with multiple methods.
 *
 * Features:
 * - Simple Exponential Smoothing
 * - Holt Linear (with/without damping)
 * - Holt-Winters (additive/multiplicative seasonal)
 * - ARIMA
 * - Point forecasts with prediction intervals
 * - Accuracy metrics (MSE, MAE, MAPE)
 * - Train/test split validation
 */
class ForecastingPanel {
public:
    ForecastingPanel();
    ~ForecastingPanel();

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
    void RenderForecastPlot();
    void RenderMetrics();
    void RenderParameters();

    void GenerateData();
    void ForecastAsync();

    bool visible_ = false;

    // Data generation
    enum class SignalType { TrendSeasonal, Seasonal, Trend, Random };
    SignalType signal_type_ = SignalType::TrendSeasonal;

    int num_samples_ = 120;
    double noise_std_ = 0.3;
    double trend_slope_ = 0.05;
    double seasonal_amplitude_ = 1.5;
    int seasonal_period_ = 12;

    // Forecasting method
    enum class Method { SimpleES, HoltLinear, HoltWinters, ARIMA };
    Method method_ = Method::HoltWinters;

    // Method parameters
    int horizon_ = 24;
    double alpha_ = -1;  // -1 = auto optimize
    double beta_ = -1;
    double gamma_ = -1;
    bool damped_ = false;
    int hw_period_ = -1;  // -1 = auto detect
    bool multiplicative_ = false;

    // ARIMA parameters
    int arima_p_ = -1;
    int arima_d_ = -1;
    int arima_q_ = -1;

    // Train/test split
    float train_ratio_ = 0.8f;
    bool use_test_split_ = true;

    // Data
    std::vector<double> time_series_;
    std::vector<double> time_axis_;
    std::vector<double> train_data_;
    std::vector<double> test_data_;

    // Results
    ForecastResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Async
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;
};

} // namespace cyxwiz
