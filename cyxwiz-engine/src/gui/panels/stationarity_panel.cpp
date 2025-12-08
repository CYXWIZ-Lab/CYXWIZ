#include "stationarity_panel.h"
#include "../icons.h"
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

StationarityPanel::StationarityPanel() {
    GenerateData();
    spdlog::info("StationarityPanel initialized");
}

StationarityPanel::~StationarityPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void StationarityPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 650), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_SCALE_BALANCED " Stationarity Testing###StationarityPanel", &visible_)) {
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

void StationarityPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Test Stationarity")) {
        TestAsync();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ROTATE " Generate")) {
        GenerateData();
        has_result_ = false;
        differenced_data_.clear();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        has_result_ = false;
        error_message_.clear();
        differenced_data_.clear();
    }

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    ImGui::Text("Samples: %d", static_cast<int>(time_series_.size()));

    if (has_result_) {
        ImGui::SameLine();
        if (result_.is_stationary) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "| Stationary");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.2f, 1.0f), "| Non-stationary (d=%d)", result_.suggested_differencing);
        }
    }
}

void StationarityPanel::RenderInputPanel() {
    ImGui::Text(ICON_FA_DATABASE " Data Source");
    ImGui::Separator();

    const char* signal_types[] = { "Random Walk", "Trend + Noise", "Stationary", "AR(1)" };
    int signal_idx = static_cast<int>(signal_type_);
    if (ImGui::Combo("Type", &signal_idx, signal_types, IM_ARRAYSIZE(signal_types))) {
        signal_type_ = static_cast<SignalType>(signal_idx);
    }

    ImGui::DragInt("Samples", &num_samples_, 1.0f, 50, 1000);
    static const double kNoiseMin = 0.1, kNoiseMax = 5.0;
    ImGui::SliderScalar("Noise Std", ImGuiDataType_Double, &noise_std_, &kNoiseMin, &kNoiseMax, "%.2f");

    if (signal_type_ == SignalType::TrendNoise) {
        static const double kSlopeMin = -0.5, kSlopeMax = 0.5;
        ImGui::SliderScalar("Trend Slope", ImGuiDataType_Double, &trend_slope_, &kSlopeMin, &kSlopeMax, "%.3f");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text(ICON_FA_SLIDERS " Test Settings");
    ImGui::Separator();

    ImGui::DragInt("Max Lags", &max_lags_, 1.0f, -1, 50);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("-1 = automatic selection");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text(ICON_FA_ARROW_DOWN " Differencing");
    ImGui::Separator();

    ImGui::DragInt("Order", &diff_order_, 1.0f, 1, 3);

    if (ImGui::Button(ICON_FA_PLAY " Apply Differencing")) {
        ApplyDifferencing();
    }
}

void StationarityPanel::RenderLoadingIndicator() {
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() / 2 - 20);
    float width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX(width / 2 - 100);
    ImGui::Text(ICON_FA_SPINNER " Testing stationarity...");
}

void StationarityPanel::RenderResults() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        RenderTimeSeriesPlot();
        return;
    }

    if (ImGui::BeginTabBar("StationarityTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Time Series")) {
            RenderTimeSeriesPlot();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CHART_AREA " Rolling Stats")) {
            RenderRollingStats();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CIRCLE_CHECK " Test Results")) {
            RenderTestResults();
            ImGui::EndTabItem();
        }
        if (!differenced_data_.empty() && ImGui::BeginTabItem(ICON_FA_ARROW_DOWN " Differenced")) {
            RenderDifferencedPlot();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void StationarityPanel::RenderTimeSeriesPlot() {
    if (time_series_.empty()) return;

    ImGui::Text("Original Time Series:");

    if (ImPlot::BeginPlot("##TimeSeries", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Time", "Value");
        ImPlot::PlotLine("Series", time_axis_.data(), time_series_.data(),
                        static_cast<int>(time_series_.size()));
        ImPlot::EndPlot();
    }
}

void StationarityPanel::RenderRollingStats() {
    if (!has_result_ || result_.rolling_mean.empty()) {
        ImGui::TextDisabled("Run test to see rolling statistics");
        return;
    }

    float height = (ImGui::GetContentRegionAvail().y - 40) / 2;

    ImGui::Text("Rolling Mean (window=%d):", result_.rolling_window);
    if (ImPlot::BeginPlot("##RollingMean", ImVec2(-1, height))) {
        ImPlot::SetupAxes("Time", "Mean");

        // Create time axis for rolling stats
        std::vector<double> roll_time(result_.rolling_mean.size());
        for (size_t i = 0; i < roll_time.size(); i++) {
            roll_time[i] = static_cast<double>(i + result_.rolling_window - 1);
        }

        ImPlot::PlotLine("Rolling Mean", roll_time.data(), result_.rolling_mean.data(),
                        static_cast<int>(result_.rolling_mean.size()));
        ImPlot::EndPlot();
    }

    ImGui::Text("Rolling Std:");
    if (ImPlot::BeginPlot("##RollingStd", ImVec2(-1, height))) {
        ImPlot::SetupAxes("Time", "Std");

        std::vector<double> roll_time(result_.rolling_std.size());
        for (size_t i = 0; i < roll_time.size(); i++) {
            roll_time[i] = static_cast<double>(i + result_.rolling_window - 1);
        }

        ImPlot::PlotLine("Rolling Std", roll_time.data(), result_.rolling_std.data(),
                        static_cast<int>(result_.rolling_std.size()));
        ImPlot::EndPlot();
    }
}

void StationarityPanel::RenderTestResults() {
    ImGui::Text(ICON_FA_FLASK " Statistical Tests");
    ImGui::Separator();
    ImGui::Spacing();

    // ADF Test Results
    ImGui::Text(ICON_FA_CHART_LINE " Augmented Dickey-Fuller Test");
    ImGui::Indent();

    ImGui::Columns(2, "adf_cols", false);
    ImGui::Text("Test Statistic:"); ImGui::NextColumn();
    ImGui::Text("%.4f", result_.adf_statistic); ImGui::NextColumn();
    ImGui::Text("p-value:"); ImGui::NextColumn();
    ImGui::Text("%.4f", result_.adf_pvalue); ImGui::NextColumn();

    ImGui::Text("Critical Values:"); ImGui::NextColumn();
    ImGui::Text("1%%: %.2f, 5%%: %.2f, 10%%: %.2f",
                result_.adf_critical.count("1%") ? result_.adf_critical.at("1%") : -3.43,
                result_.adf_critical.count("5%") ? result_.adf_critical.at("5%") : -2.86,
                result_.adf_critical.count("10%") ? result_.adf_critical.at("10%") : -2.57);
    ImGui::NextColumn();

    ImGui::Text("Result:"); ImGui::NextColumn();
    if (result_.adf_stationary) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Reject H0 (Stationary)");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.2f, 1.0f), "Fail to reject H0 (Non-stationary)");
    }
    ImGui::NextColumn();
    ImGui::Columns(1);
    ImGui::Unindent();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // KPSS Test Results
    ImGui::Text(ICON_FA_CHART_BAR " KPSS Test");
    ImGui::Indent();

    ImGui::Columns(2, "kpss_cols", false);
    ImGui::Text("Test Statistic:"); ImGui::NextColumn();
    ImGui::Text("%.4f", result_.kpss_statistic); ImGui::NextColumn();

    ImGui::Text("Result:"); ImGui::NextColumn();
    if (result_.kpss_stationary) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Fail to reject H0 (Stationary)");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.2f, 1.0f), "Reject H0 (Non-stationary)");
    }
    ImGui::NextColumn();
    ImGui::Columns(1);
    ImGui::Unindent();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Combined Result
    ImGui::Text(ICON_FA_SCALE_BALANCED " Combined Conclusion");
    ImGui::Indent();

    if (result_.is_stationary) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                           ICON_FA_CHECK " Series is STATIONARY");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.2f, 1.0f),
                           ICON_FA_XMARK " Series is NON-STATIONARY");
        ImGui::Text("Suggested differencing order: d = %d", result_.suggested_differencing);
    }

    ImGui::Spacing();
    ImGui::TextWrapped("%s", result_.analysis.c_str());
    ImGui::Unindent();
}

void StationarityPanel::RenderDifferencedPlot() {
    if (differenced_data_.empty()) return;

    ImGui::Text("Differenced Series (order=%d):", diff_order_);

    std::vector<double> diff_time(differenced_data_.size());
    for (size_t i = 0; i < diff_time.size(); i++) {
        diff_time[i] = static_cast<double>(i);
    }

    if (ImPlot::BeginPlot("##Differenced", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Time", "Value");
        ImPlot::PlotLine("Differenced", diff_time.data(), differenced_data_.data(),
                        static_cast<int>(differenced_data_.size()));

        // Zero line
        double zero_x[] = {0, static_cast<double>(differenced_data_.size())};
        double zero_y[] = {0, 0};
        ImPlot::SetNextLineStyle(ImVec4(1, 0, 0, 0.5f));
        ImPlot::PlotLine("##zero", zero_x, zero_y, 2);

        ImPlot::EndPlot();
    }
}

void StationarityPanel::GenerateData() {
    time_series_.clear();

    switch (signal_type_) {
        case SignalType::RandomWalk:
            time_series_ = TimeSeries::GenerateRandomWalk(num_samples_, 0, noise_std_);
            break;
        case SignalType::TrendNoise: {
            auto noise = TimeSeries::GenerateWhiteNoise(num_samples_, 0, noise_std_);
            time_series_.resize(num_samples_);
            for (int i = 0; i < num_samples_; i++) {
                time_series_[i] = trend_slope_ * i + noise[i];
            }
            break;
        }
        case SignalType::Stationary:
            time_series_ = TimeSeries::GenerateWhiteNoise(num_samples_, 0, noise_std_);
            break;
        case SignalType::AR1:
            time_series_ = TimeSeries::GenerateAR(num_samples_, {0.8}, noise_std_);
            break;
    }

    time_axis_.resize(num_samples_);
    for (int i = 0; i < num_samples_; i++) {
        time_axis_[i] = static_cast<double>(i);
    }
}

void StationarityPanel::TestAsync() {
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
            result_ = TimeSeries::TestStationarity(time_series_, max_lags_);
            if (result_.success) {
                has_result_ = true;
                spdlog::info("Stationarity test: ADF={}, KPSS={}, stationary={}",
                            result_.adf_stationary, result_.kpss_stationary, result_.is_stationary);
            } else {
                error_message_ = result_.error_message;
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void StationarityPanel::ApplyDifferencing() {
    if (time_series_.empty()) return;
    differenced_data_ = TimeSeries::Difference(time_series_, diff_order_);
    spdlog::info("Applied differencing of order {}", diff_order_);
}

} // namespace cyxwiz
