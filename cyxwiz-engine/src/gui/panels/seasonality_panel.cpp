#include "seasonality_panel.h"
#include "../icons.h"
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

SeasonalityPanel::SeasonalityPanel() {
    GenerateData();
    spdlog::info("SeasonalityPanel initialized");
}

SeasonalityPanel::~SeasonalityPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void SeasonalityPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 650), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CALENDAR " Seasonality Detection###SeasonalityPanel", &visible_)) {
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

void SeasonalityPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS " Detect")) {
        DetectAsync();
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

    if (has_result_) {
        ImGui::SameLine();
        if (result_.has_seasonality) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                               "| Period: %d (%.0f%%)",
                               result_.detected_period,
                               result_.strength * 100);
        } else {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "| No seasonality");
        }
    }
}

void SeasonalityPanel::RenderInputPanel() {
    ImGui::Text(ICON_FA_DATABASE " Data Source");
    ImGui::Separator();

    const char* signal_types[] = { "Seasonal", "Multi-Seasonal", "Trend+Seasonal", "No Seasonality" };
    int signal_idx = static_cast<int>(signal_type_);
    if (ImGui::Combo("Type", &signal_idx, signal_types, IM_ARRAYSIZE(signal_types))) {
        signal_type_ = static_cast<SignalType>(signal_idx);
    }

    ImGui::DragInt("Samples", &num_samples_, 1.0f, 50, 1000);

    if (signal_type_ != SignalType::NoSeason) {
        ImGui::DragInt("Period", &primary_period_, 1.0f, 2, 52);
        static const double kAmpMin = 0.1, kAmpMax = 10.0;
        ImGui::SliderScalar("Amplitude", ImGuiDataType_Double, &seasonal_amplitude_, &kAmpMin, &kAmpMax, "%.2f");
    }

    if (signal_type_ == SignalType::MultiSeasonal) {
        ImGui::DragInt("Period 2", &secondary_period_, 1.0f, 2, 52);
    }

    static const double kNoiseMin = 0.0, kNoiseMax = 5.0;
    ImGui::SliderScalar("Noise Std", ImGuiDataType_Double, &noise_std_, &kNoiseMin, &kNoiseMax, "%.2f");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text(ICON_FA_SLIDERS " Detection Settings");
    ImGui::Separator();

    ImGui::DragInt("Min Period", &min_period_, 1.0f, 2, 20);
    ImGui::DragInt("Max Period", &max_period_, 1.0f, -1, 100);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("-1 = automatic (n/2)");
    }
}

void SeasonalityPanel::RenderLoadingIndicator() {
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() / 2 - 20);
    float width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX(width / 2 - 100);
    ImGui::Text(ICON_FA_SPINNER " Detecting seasonality...");
}

void SeasonalityPanel::RenderResults() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        // Show time series
        if (!time_series_.empty()) {
            ImGui::Text("Time Series:");
            if (ImPlot::BeginPlot("##TimeSeries", ImVec2(-1, -1))) {
                ImPlot::SetupAxes("Time", "Value");
                ImPlot::PlotLine("Series", time_axis_.data(), time_series_.data(),
                                static_cast<int>(time_series_.size()));
                ImPlot::EndPlot();
            }
        }
        return;
    }

    if (ImGui::BeginTabBar("SeasonalityTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CHART_AREA " Periodogram")) {
            RenderPeriodogramPlot();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " ACF Peaks")) {
            RenderACFPlot();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CIRCLE_INFO " Summary")) {
            RenderSummary();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void SeasonalityPanel::RenderPeriodogramPlot() {
    ImGui::Text("Periodogram (Spectral Density):");

    if (result_.periodogram.empty()) {
        ImGui::TextDisabled("No periodogram data");
        return;
    }

    // Convert frequencies to periods for display
    std::vector<double> periods;
    std::vector<double> power;

    for (size_t i = 1; i < result_.frequencies.size(); i++) {
        if (result_.frequencies[i] > 1e-10) {
            double period = 1.0 / result_.frequencies[i];
            if (period >= min_period_ && (max_period_ < 0 || period <= max_period_)) {
                periods.push_back(period);
                power.push_back(result_.periodogram[i]);
            }
        }
    }

    if (ImPlot::BeginPlot("##Periodogram", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Period", "Spectral Power");

        if (!periods.empty()) {
            ImPlot::PlotLine("Power", periods.data(), power.data(),
                            static_cast<int>(periods.size()));

            // Mark detected period
            if (result_.has_seasonality && result_.detected_period > 0) {
                double peak_x[] = {static_cast<double>(result_.detected_period)};
                double peak_y[] = {*std::max_element(power.begin(), power.end())};
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 10, ImVec4(1, 0, 0, 1));
                ImPlot::PlotScatter("Peak", peak_x, peak_y, 1);
            }
        }

        ImPlot::EndPlot();
    }
}

void SeasonalityPanel::RenderACFPlot() {
    ImGui::Text("ACF with Seasonal Peaks:");

    // Compute ACF for display
    auto acf = TimeSeries::ComputeACF(time_series_, max_period_ > 0 ? max_period_ : 50);

    if (!acf.success || acf.acf.empty()) {
        ImGui::TextDisabled("Failed to compute ACF");
        return;
    }

    if (ImPlot::BeginPlot("##ACFSeasonal", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Lag", "ACF");
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.1, 1.1);

        // Confidence bounds
        if (!acf.confidence_upper.empty()) {
            ImPlot::SetNextFillStyle(ImVec4(0.5f, 0.5f, 1.0f, 0.2f));
            ImPlot::PlotShaded("95% CI", acf.lags.data(),
                              acf.confidence_upper.data(),
                              acf.confidence_lower.data(),
                              static_cast<int>(acf.lags.size()));
        }

        // ACF
        ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.6f, 1.0f, 0.8f));
        ImPlot::PlotBars("ACF", acf.lags.data(), acf.acf.data(),
                        static_cast<int>(acf.acf.size()), 0.6);

        // Mark detected seasonal peaks
        if (!result_.acf_peaks.empty()) {
            std::vector<double> peak_x, peak_y;
            for (int peak : result_.acf_peaks) {
                if (peak < static_cast<int>(acf.acf.size())) {
                    peak_x.push_back(static_cast<double>(peak));
                    peak_y.push_back(acf.acf[peak]);
                }
            }
            if (!peak_x.empty()) {
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 8, ImVec4(1, 0.3f, 0, 1));
                ImPlot::PlotScatter("Seasonal Peaks", peak_x.data(), peak_y.data(),
                                   static_cast<int>(peak_x.size()));
            }
        }

        ImPlot::EndPlot();
    }
}

void SeasonalityPanel::RenderSummary() {
    ImGui::Text(ICON_FA_CIRCLE_INFO " Seasonality Analysis Summary");
    ImGui::Separator();
    ImGui::Spacing();

    // Main result
    if (result_.has_seasonality) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                           ICON_FA_CHECK " Seasonality DETECTED");
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                           ICON_FA_XMARK " No significant seasonality detected");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Columns(2, "summary_cols", false);

    ImGui::Text("Primary Period:");
    ImGui::NextColumn();
    ImGui::Text("%d", result_.detected_period);
    ImGui::NextColumn();

    ImGui::Text("Strength:");
    ImGui::NextColumn();
    ImGui::ProgressBar(static_cast<float>(result_.strength), ImVec2(-1, 0),
                       (std::to_string(static_cast<int>(result_.strength * 100)) + "%").c_str());
    ImGui::NextColumn();

    ImGui::Columns(1);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Candidate periods
    if (!result_.candidate_periods.empty()) {
        ImGui::Text(ICON_FA_LIST " Detected Periods (ranked by strength):");
        ImGui::Spacing();

        ImGui::Columns(3, "periods_table", true);
        ImGui::Text("Rank"); ImGui::NextColumn();
        ImGui::Text("Period"); ImGui::NextColumn();
        ImGui::Text("Strength"); ImGui::NextColumn();
        ImGui::Separator();

        for (size_t i = 0; i < result_.candidate_periods.size() && i < 5; i++) {
            ImGui::Text("%d", static_cast<int>(i + 1));
            ImGui::NextColumn();
            ImGui::Text("%d", result_.candidate_periods[i]);
            ImGui::NextColumn();
            if (i < result_.candidate_strengths.size()) {
                ImGui::Text("%.2f%%", result_.candidate_strengths[i] * 100);
            }
            ImGui::NextColumn();
        }
        ImGui::Columns(1);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ACF peaks
    if (!result_.acf_peaks.empty()) {
        ImGui::Text(ICON_FA_CHART_BAR " ACF Peak Lags: ");
        ImGui::SameLine();
        for (size_t i = 0; i < result_.acf_peaks.size() && i < 8; i++) {
            ImGui::Text("%d", result_.acf_peaks[i]);
            if (i < result_.acf_peaks.size() - 1 && i < 7) {
                ImGui::SameLine();
                ImGui::Text(", ");
                ImGui::SameLine();
            }
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextWrapped("%s", result_.analysis.c_str());
}

void SeasonalityPanel::GenerateData() {
    time_series_.clear();
    constexpr double TWO_PI = 6.28318530718;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0, noise_std_);

    time_series_.resize(num_samples_);

    switch (signal_type_) {
        case SignalType::Seasonal:
            for (int i = 0; i < num_samples_; i++) {
                time_series_[i] = seasonal_amplitude_ * std::sin(TWO_PI * i / primary_period_) + noise(gen);
            }
            break;

        case SignalType::MultiSeasonal:
            for (int i = 0; i < num_samples_; i++) {
                time_series_[i] = seasonal_amplitude_ * std::sin(TWO_PI * i / primary_period_) +
                                  0.5 * seasonal_amplitude_ * std::sin(TWO_PI * i / secondary_period_) +
                                  noise(gen);
            }
            break;

        case SignalType::TrendSeasonal:
            for (int i = 0; i < num_samples_; i++) {
                time_series_[i] = 0.05 * i +
                                  seasonal_amplitude_ * std::sin(TWO_PI * i / primary_period_) +
                                  noise(gen);
            }
            break;

        case SignalType::NoSeason:
            time_series_ = TimeSeries::GenerateWhiteNoise(num_samples_, 0, noise_std_ > 0 ? noise_std_ : 1.0);
            break;
    }

    time_axis_.resize(num_samples_);
    for (int i = 0; i < num_samples_; i++) {
        time_axis_[i] = static_cast<double>(i);
    }
}

void SeasonalityPanel::DetectAsync() {
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
            result_ = TimeSeries::DetectSeasonality(time_series_, min_period_, max_period_);
            if (result_.success) {
                has_result_ = true;
                spdlog::info("Seasonality detection: has_seasonality={}, period={}, strength={:.2f}",
                            result_.has_seasonality, result_.detected_period, result_.strength);
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
