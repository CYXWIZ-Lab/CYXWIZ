#include "forecasting_panel.h"
#include "../icons.h"
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

ForecastingPanel::ForecastingPanel() {
    GenerateData();
    spdlog::info("ForecastingPanel initialized");
}

ForecastingPanel::~ForecastingPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void ForecastingPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(1000, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CHART_LINE " Forecasting###ForecastingPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            float panel_width = ImGui::GetContentRegionAvail().x;

            ImGui::BeginChild("InputPanel", ImVec2(panel_width * 0.25f, 0), true);
            RenderInputPanel();
            ImGui::EndChild();

            ImGui::SameLine();

            ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
            RenderResults();
            ImGui::EndChild();
        }
    }
    ImGui::End();
}

void ForecastingPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Forecast")) {
        ForecastAsync();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ROTATE " Generate")) {
        GenerateData();
        has_result_ = false;
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        has_result_ = false;
        error_message_.clear();
    }

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    ImGui::Text("Samples: %d", static_cast<int>(time_series_.size()));

    if (has_result_ && result_.success) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                           "| MAPE: %.2f%%", result_.mape);
    }
}

void ForecastingPanel::RenderInputPanel() {
    ImGui::Text(ICON_FA_DATABASE " Data Source");
    ImGui::Separator();

    const char* signal_types[] = { "Trend+Seasonal", "Seasonal Only", "Trend Only", "Random" };
    int signal_idx = static_cast<int>(signal_type_);
    if (ImGui::Combo("Type", &signal_idx, signal_types, IM_ARRAYSIZE(signal_types))) {
        signal_type_ = static_cast<SignalType>(signal_idx);
    }

    ImGui::DragInt("Samples", &num_samples_, 1.0f, 50, 500);

    static const double kSlopeMin = -0.5, kSlopeMax = 0.5;
    static const double kAmpMin = 0.1, kAmpMax = 5.0;
    static const double kNoiseMin = 0.0, kNoiseMax = 2.0;

    if (signal_type_ == SignalType::TrendSeasonal || signal_type_ == SignalType::Trend) {
        ImGui::SliderScalar("Trend Slope", ImGuiDataType_Double, &trend_slope_, &kSlopeMin, &kSlopeMax, "%.3f");
    }

    if (signal_type_ == SignalType::TrendSeasonal || signal_type_ == SignalType::Seasonal) {
        ImGui::SliderScalar("Season Amp", ImGuiDataType_Double, &seasonal_amplitude_, &kAmpMin, &kAmpMax, "%.2f");
        ImGui::DragInt("Period", &seasonal_period_, 1.0f, 2, 52);
    }

    ImGui::SliderScalar("Noise Std", ImGuiDataType_Double, &noise_std_, &kNoiseMin, &kNoiseMax, "%.2f");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text(ICON_FA_SLIDERS " Forecast Settings");
    ImGui::Separator();

    const char* methods[] = { "Simple ES", "Holt Linear", "Holt-Winters", "ARIMA" };
    int method_idx = static_cast<int>(method_);
    if (ImGui::Combo("Method", &method_idx, methods, IM_ARRAYSIZE(methods))) {
        method_ = static_cast<Method>(method_idx);
    }

    ImGui::DragInt("Horizon", &horizon_, 1.0f, 1, 100);

    // Method-specific parameters
    static const double kParamMin = -1.0, kParamMax = 1.0;

    if (method_ == Method::SimpleES) {
        ImGui::Text("Alpha (-1=auto):");
        ImGui::SliderScalar("##alpha", ImGuiDataType_Double, &alpha_, &kParamMin, &kParamMax, "%.2f");
    }
    else if (method_ == Method::HoltLinear) {
        ImGui::Text("Parameters (-1=auto):");
        ImGui::SliderScalar("Alpha", ImGuiDataType_Double, &alpha_, &kParamMin, &kParamMax, "%.2f");
        ImGui::SliderScalar("Beta", ImGuiDataType_Double, &beta_, &kParamMin, &kParamMax, "%.2f");
        ImGui::Checkbox("Damped Trend", &damped_);
    }
    else if (method_ == Method::HoltWinters) {
        ImGui::Text("Parameters (-1=auto):");
        ImGui::SliderScalar("Alpha", ImGuiDataType_Double, &alpha_, &kParamMin, &kParamMax, "%.2f");
        ImGui::SliderScalar("Beta", ImGuiDataType_Double, &beta_, &kParamMin, &kParamMax, "%.2f");
        ImGui::SliderScalar("Gamma", ImGuiDataType_Double, &gamma_, &kParamMin, &kParamMax, "%.2f");
        ImGui::DragInt("Period", &hw_period_, 1.0f, -1, 52);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("-1 = auto detect");
        }
        ImGui::Checkbox("Multiplicative", &multiplicative_);
    }
    else if (method_ == Method::ARIMA) {
        ImGui::Text("ARIMA(p,d,q) (-1=auto):");
        ImGui::DragInt("p (AR)", &arima_p_, 1.0f, -1, 10);
        ImGui::DragInt("d (Diff)", &arima_d_, 1.0f, -1, 3);
        ImGui::DragInt("q (MA)", &arima_q_, 1.0f, -1, 10);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text(ICON_FA_SCISSORS " Train/Test Split");
    ImGui::Separator();

    ImGui::Checkbox("Use Test Split", &use_test_split_);
    if (use_test_split_) {
        ImGui::SliderFloat("Train %", &train_ratio_, 0.5f, 0.95f, "%.0f%%");
    }
}

void ForecastingPanel::RenderLoadingIndicator() {
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() / 2 - 20);
    float width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX(width / 2 - 80);
    ImGui::Text(ICON_FA_SPINNER " Forecasting...");
}

void ForecastingPanel::RenderResults() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        // Show time series plot
        if (!time_series_.empty()) {
            ImGui::Text("Time Series:");
            if (ImPlot::BeginPlot("##TimeSeries", ImVec2(-1, -1))) {
                ImPlot::SetupAxes("Time", "Value");

                if (use_test_split_ && !train_data_.empty()) {
                    // Show train/test split
                    std::vector<double> train_time(train_data_.size());
                    for (size_t i = 0; i < train_time.size(); i++) {
                        train_time[i] = static_cast<double>(i);
                    }
                    ImPlot::PlotLine("Train", train_time.data(), train_data_.data(),
                                    static_cast<int>(train_data_.size()));

                    std::vector<double> test_time(test_data_.size());
                    for (size_t i = 0; i < test_time.size(); i++) {
                        test_time[i] = static_cast<double>(i + train_data_.size());
                    }
                    ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.5f, 0.0f, 1.0f));
                    ImPlot::PlotLine("Test", test_time.data(), test_data_.data(),
                                    static_cast<int>(test_data_.size()));
                } else {
                    ImPlot::PlotLine("Series", time_axis_.data(), time_series_.data(),
                                    static_cast<int>(time_series_.size()));
                }

                ImPlot::EndPlot();
            }
        }
        return;
    }

    if (ImGui::BeginTabBar("ForecastTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Forecast")) {
            RenderForecastPlot();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_TABLE " Metrics")) {
            RenderMetrics();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_SLIDERS " Parameters")) {
            RenderParameters();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void ForecastingPanel::RenderForecastPlot() {
    if (result_.forecast.empty()) {
        ImGui::TextDisabled("No forecast data");
        return;
    }

    if (ImPlot::BeginPlot("##ForecastPlot", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Time", "Value");

        // Historical data
        size_t hist_size = use_test_split_ ? train_data_.size() : time_series_.size();
        std::vector<double> hist_time(hist_size);
        for (size_t i = 0; i < hist_size; i++) {
            hist_time[i] = static_cast<double>(i);
        }

        if (use_test_split_) {
            ImPlot::PlotLine("Training", hist_time.data(), train_data_.data(),
                            static_cast<int>(train_data_.size()));

            // Actual test data
            if (!test_data_.empty()) {
                std::vector<double> test_time(test_data_.size());
                for (size_t i = 0; i < test_time.size(); i++) {
                    test_time[i] = static_cast<double>(i + train_data_.size());
                }
                ImPlot::SetNextLineStyle(ImVec4(0.0f, 0.7f, 0.0f, 1.0f));
                ImPlot::PlotLine("Actual", test_time.data(), test_data_.data(),
                                static_cast<int>(test_data_.size()));
            }
        } else {
            ImPlot::PlotLine("Historical", hist_time.data(), time_series_.data(),
                            static_cast<int>(time_series_.size()));
        }

        // Forecast
        std::vector<double> forecast_time(result_.forecast.size());
        for (size_t i = 0; i < forecast_time.size(); i++) {
            forecast_time[i] = static_cast<double>(i + hist_size);
        }

        // Prediction intervals
        if (!result_.lower_bound.empty() && !result_.upper_bound.empty()) {
            ImPlot::SetNextFillStyle(ImVec4(1.0f, 0.5f, 0.0f, 0.2f));
            ImPlot::PlotShaded("95% PI", forecast_time.data(),
                              result_.lower_bound.data(),
                              result_.upper_bound.data(),
                              static_cast<int>(forecast_time.size()));
        }

        // Point forecast
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.3f, 0.0f, 1.0f), 2.0f);
        ImPlot::PlotLine("Forecast", forecast_time.data(), result_.forecast.data(),
                        static_cast<int>(result_.forecast.size()));

        // Fitted values
        if (!result_.fitted_values.empty()) {
            std::vector<double> fitted_time(result_.fitted_values.size());
            for (size_t i = 0; i < fitted_time.size(); i++) {
                fitted_time[i] = static_cast<double>(i);
            }
            ImPlot::SetNextLineStyle(ImVec4(0.5f, 0.5f, 1.0f, 0.7f));
            ImPlot::PlotLine("Fitted", fitted_time.data(), result_.fitted_values.data(),
                            static_cast<int>(result_.fitted_values.size()));
        }

        ImPlot::EndPlot();
    }
}

void ForecastingPanel::RenderMetrics() {
    ImGui::Text(ICON_FA_CHART_BAR " Forecast Accuracy Metrics");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Columns(2, "metrics_cols", false);

    ImGui::Text("MSE (Mean Squared Error):");
    ImGui::NextColumn();
    ImGui::Text("%.6f", result_.mse);
    ImGui::NextColumn();

    ImGui::Text("MAE (Mean Absolute Error):");
    ImGui::NextColumn();
    ImGui::Text("%.6f", result_.mae);
    ImGui::NextColumn();

    ImGui::Text("MAPE (Mean Abs % Error):");
    ImGui::NextColumn();
    ImGui::Text("%.2f%%", result_.mape);
    ImGui::NextColumn();

    ImGui::Text("RMSE (Root MSE):");
    ImGui::NextColumn();
    ImGui::Text("%.6f", std::sqrt(result_.mse));
    ImGui::NextColumn();

    ImGui::Columns(1);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Forecast horizon info
    ImGui::Text(ICON_FA_CLOCK " Forecast Horizon: %d periods", static_cast<int>(result_.forecast.size()));

    if (use_test_split_) {
        ImGui::Text(ICON_FA_SCISSORS " Train samples: %d, Test samples: %d",
                    static_cast<int>(train_data_.size()),
                    static_cast<int>(test_data_.size()));
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Interpretation
    ImGui::Text(ICON_FA_CIRCLE_INFO " Interpretation:");
    ImGui::Indent();

    if (result_.mape < 10.0) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                           "Highly accurate forecast (MAPE < 10%%)");
    } else if (result_.mape < 20.0) {
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f),
                           "Good forecast (MAPE < 20%%)");
    } else if (result_.mape < 50.0) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                           "Reasonable forecast (MAPE < 50%%)");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                           "Poor forecast (MAPE > 50%%)");
    }

    ImGui::Unindent();
}

void ForecastingPanel::RenderParameters() {
    ImGui::Text(ICON_FA_SLIDERS " Model Parameters");
    ImGui::Separator();
    ImGui::Spacing();

    // Method name
    const char* method_names[] = { "Simple Exponential Smoothing", "Holt Linear", "Holt-Winters", "ARIMA" };
    ImGui::Text("Method: %s", method_names[static_cast<int>(method_)]);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Display optimized parameters from result
    if (!result_.model_summary.empty()) {
        ImGui::Text(ICON_FA_COG " Model Summary:");
        ImGui::Indent();
        ImGui::TextWrapped("%s", result_.model_summary.c_str());
        ImGui::Unindent();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // User-specified parameters
    ImGui::Text(ICON_FA_USER " User Settings:");
    ImGui::Indent();

    ImGui::Columns(2, "param_cols", false);

    ImGui::Text("Horizon:");
    ImGui::NextColumn();
    ImGui::Text("%d", horizon_);
    ImGui::NextColumn();

    if (method_ == Method::SimpleES) {
        ImGui::Text("Alpha:");
        ImGui::NextColumn();
        if (alpha_ < 0) ImGui::Text("auto");
        else ImGui::Text("%.2f", alpha_);
        ImGui::NextColumn();
    }
    else if (method_ == Method::HoltLinear) {
        ImGui::Text("Alpha:");
        ImGui::NextColumn();
        if (alpha_ < 0) ImGui::Text("auto");
        else ImGui::Text("%.2f", alpha_);
        ImGui::NextColumn();

        ImGui::Text("Beta:");
        ImGui::NextColumn();
        if (beta_ < 0) ImGui::Text("auto");
        else ImGui::Text("%.2f", beta_);
        ImGui::NextColumn();

        ImGui::Text("Damped:");
        ImGui::NextColumn();
        ImGui::Text("%s", damped_ ? "Yes" : "No");
        ImGui::NextColumn();
    }
    else if (method_ == Method::HoltWinters) {
        ImGui::Text("Alpha:");
        ImGui::NextColumn();
        if (alpha_ < 0) ImGui::Text("auto");
        else ImGui::Text("%.2f", alpha_);
        ImGui::NextColumn();

        ImGui::Text("Beta:");
        ImGui::NextColumn();
        if (beta_ < 0) ImGui::Text("auto");
        else ImGui::Text("%.2f", beta_);
        ImGui::NextColumn();

        ImGui::Text("Gamma:");
        ImGui::NextColumn();
        if (gamma_ < 0) ImGui::Text("auto");
        else ImGui::Text("%.2f", gamma_);
        ImGui::NextColumn();

        ImGui::Text("Period:");
        ImGui::NextColumn();
        if (hw_period_ < 0) ImGui::Text("auto");
        else ImGui::Text("%d", hw_period_);
        ImGui::NextColumn();

        ImGui::Text("Seasonal:");
        ImGui::NextColumn();
        ImGui::Text("%s", multiplicative_ ? "Multiplicative" : "Additive");
        ImGui::NextColumn();
    }
    else if (method_ == Method::ARIMA) {
        ImGui::Text("p (AR):");
        ImGui::NextColumn();
        if (arima_p_ < 0) ImGui::Text("auto");
        else ImGui::Text("%d", arima_p_);
        ImGui::NextColumn();

        ImGui::Text("d (Diff):");
        ImGui::NextColumn();
        if (arima_d_ < 0) ImGui::Text("auto");
        else ImGui::Text("%d", arima_d_);
        ImGui::NextColumn();

        ImGui::Text("q (MA):");
        ImGui::NextColumn();
        if (arima_q_ < 0) ImGui::Text("auto");
        else ImGui::Text("%d", arima_q_);
        ImGui::NextColumn();
    }

    ImGui::Columns(1);
    ImGui::Unindent();
}

void ForecastingPanel::GenerateData() {
    time_series_.clear();
    constexpr double TWO_PI = 6.28318530718;

    switch (signal_type_) {
        case SignalType::TrendSeasonal:
            time_series_ = TimeSeries::GenerateTrendSeasonal(
                num_samples_, trend_slope_, seasonal_amplitude_, seasonal_period_, noise_std_);
            break;

        case SignalType::Seasonal:
            time_series_ = TimeSeries::GenerateTrendSeasonal(
                num_samples_, 0.0, seasonal_amplitude_, seasonal_period_, noise_std_);
            break;

        case SignalType::Trend:
            time_series_ = TimeSeries::GenerateTrendSeasonal(
                num_samples_, trend_slope_, 0.0, seasonal_period_, noise_std_);
            break;

        case SignalType::Random:
            time_series_ = TimeSeries::GenerateWhiteNoise(num_samples_, 0.0, noise_std_ > 0 ? noise_std_ : 1.0);
            break;
    }

    time_axis_.resize(num_samples_);
    for (int i = 0; i < num_samples_; i++) {
        time_axis_[i] = static_cast<double>(i);
    }

    // Create train/test split
    if (use_test_split_) {
        int train_size = static_cast<int>(time_series_.size() * train_ratio_);
        train_data_.assign(time_series_.begin(), time_series_.begin() + train_size);
        test_data_.assign(time_series_.begin() + train_size, time_series_.end());
    } else {
        train_data_ = time_series_;
        test_data_.clear();
    }
}

void ForecastingPanel::ForecastAsync() {
    if (is_computing_.load()) return;
    if (time_series_.empty()) {
        error_message_ = "No data";
        return;
    }

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_result_ = false;
    error_message_.clear();

    compute_thread_ = std::make_unique<std::thread>([this]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            const auto& data = use_test_split_ ? train_data_ : time_series_;
            int forecast_horizon = use_test_split_ ? static_cast<int>(test_data_.size()) : horizon_;

            switch (method_) {
                case Method::SimpleES:
                    result_ = TimeSeries::SimpleES(data, forecast_horizon, alpha_);
                    break;

                case Method::HoltLinear:
                    result_ = TimeSeries::HoltLinear(data, forecast_horizon, alpha_, beta_, damped_);
                    break;

                case Method::HoltWinters: {
                    std::string seasonal_type = multiplicative_ ? "multiplicative" : "additive";
                    result_ = TimeSeries::HoltWinters(data, forecast_horizon, hw_period_,
                                                     seasonal_type, alpha_, beta_, gamma_);
                    break;
                }

                case Method::ARIMA:
                    result_ = TimeSeries::ARIMA(data, forecast_horizon, arima_p_, arima_d_, arima_q_);
                    break;
            }

            if (result_.success) {
                // If using test split, compute actual vs forecast metrics
                if (use_test_split_ && !test_data_.empty() && !result_.forecast.empty()) {
                    size_t n = std::min(test_data_.size(), result_.forecast.size());
                    double mse_sum = 0, mae_sum = 0, mape_sum = 0;
                    int mape_count = 0;

                    for (size_t i = 0; i < n; i++) {
                        double err = test_data_[i] - result_.forecast[i];
                        mse_sum += err * err;
                        mae_sum += std::abs(err);
                        if (std::abs(test_data_[i]) > 1e-10) {
                            mape_sum += std::abs(err / test_data_[i]);
                            mape_count++;
                        }
                    }

                    result_.mse = mse_sum / n;
                    result_.mae = mae_sum / n;
                    result_.mape = mape_count > 0 ? (mape_sum / mape_count) * 100 : 0;
                }

                has_result_ = true;
                spdlog::info("Forecasting complete: MSE={:.4f}, MAE={:.4f}, MAPE={:.2f}%",
                            result_.mse, result_.mae, result_.mape);
            } else {
                error_message_ = result_.error_message;
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

} // namespace cyxwiz
