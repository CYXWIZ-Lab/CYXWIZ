#include "decomposition_panel.h"
#include "../icons.h"
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

DecompositionPanel::DecompositionPanel() {
    // Generate initial data
    GenerateData();
    spdlog::info("DecompositionPanel initialized");
}

DecompositionPanel::~DecompositionPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void DecompositionPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(1000, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CHART_LINE " Time Series Decomposition###DecompositionPanel", &visible_)) {
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

void DecompositionPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Decompose")) {
        DecomposeAsync();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Decompose time series into components");
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_ROTATE " Generate")) {
        GenerateData();
        has_result_ = false;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Generate new synthetic data");
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
        ImGui::Text("| Period: %d | Method: %s",
                    result_.period,
                    result_.method.c_str());
    }
}

void DecompositionPanel::RenderInputPanel() {
    ImGui::Text(ICON_FA_DATABASE " Data Source");
    ImGui::Separator();

    const char* data_types[] = { "Generate", "Manual" };
    int data_type_idx = static_cast<int>(data_type_);
    if (ImGui::Combo("Source", &data_type_idx, data_types, IM_ARRAYSIZE(data_types))) {
        data_type_ = static_cast<DataType>(data_type_idx);
    }

    if (data_type_ == DataType::Generate) {
        RenderGenerateOptions();
    }

    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::Text(ICON_FA_SLIDERS " Decomposition Settings");
    ImGui::Separator();
    RenderDecompositionOptions();
}

void DecompositionPanel::RenderGenerateOptions() {
    ImGui::Spacing();

    const char* signal_types[] = {
        "Trend + Seasonal",
        "Random Walk",
        "White Noise",
        "AR(2)",
        "Pure Seasonal"
    };
    int signal_idx = static_cast<int>(signal_type_);
    if (ImGui::Combo("Signal Type", &signal_idx, signal_types, IM_ARRAYSIZE(signal_types))) {
        signal_type_ = static_cast<SignalType>(signal_idx);
    }

    ImGui::DragInt("Samples (n)", &num_samples_, 1.0f, 50, 1000);

    if (signal_type_ == SignalType::TrendSeasonal ||
        signal_type_ == SignalType::Seasonal) {
        static const double kSlopeMin = -1.0, kSlopeMax = 1.0;
        static const double kAmpMin = 0.0, kAmpMax = 10.0;
        ImGui::SliderScalar("Trend Slope", ImGuiDataType_Double, &trend_slope_, &kSlopeMin, &kSlopeMax, "%.3f");
        ImGui::SliderScalar("Seasonal Amp", ImGuiDataType_Double, &seasonal_amplitude_, &kAmpMin, &kAmpMax, "%.2f");
        ImGui::DragInt("Period", &seasonal_period_, 1.0f, 2, 52);
    }

    static const double kNoiseMin = 0.0, kNoiseMax = 5.0;
    ImGui::SliderScalar("Noise Std", ImGuiDataType_Double, &noise_std_, &kNoiseMin, &kNoiseMax, "%.2f");

    if (ImGui::Button(ICON_FA_ROTATE " Regenerate")) {
        GenerateData();
        has_result_ = false;
    }
}

void DecompositionPanel::RenderDecompositionOptions() {
    const char* methods[] = { "Classical", "STL" };
    int method_idx = static_cast<int>(decomp_method_);
    if (ImGui::Combo("Method", &method_idx, methods, IM_ARRAYSIZE(methods))) {
        decomp_method_ = static_cast<DecompMethod>(method_idx);
    }

    if (decomp_method_ == DecompMethod::Classical) {
        const char* types[] = { "Additive", "Multiplicative" };
        int type_idx = (seasonal_type_ == "multiplicative") ? 1 : 0;
        if (ImGui::Combo("Type", &type_idx, types, IM_ARRAYSIZE(types))) {
            seasonal_type_ = (type_idx == 1) ? "multiplicative" : "additive";
        }
    }

    ImGui::DragInt("Period", &period_, 1.0f, 2, 52);

    if (decomp_method_ == DecompMethod::STL) {
        ImGui::DragInt("Seasonal Window", &stl_seasonal_window_, 1.0f, 3, 51);
        if (stl_seasonal_window_ % 2 == 0) stl_seasonal_window_++;

        ImGui::DragInt("Trend Window", &stl_trend_window_, 1.0f, -1, 101);
        if (stl_trend_window_ > 0 && stl_trend_window_ % 2 == 0) stl_trend_window_++;
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("-1 = auto");
        }
    }
}

void DecompositionPanel::RenderLoadingIndicator() {
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() / 2 - 20);
    float width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX(width / 2 - 100);

    ImGui::Text(ICON_FA_SPINNER " Decomposing...");

    ImGui::SetCursorPosX(width / 2 - 100);
    ImGui::ProgressBar(-1.0f * static_cast<float>(ImGui::GetTime()), ImVec2(200, 0));
}

void DecompositionPanel::RenderResults() {
    ImGui::Text(ICON_FA_CHART_AREA " Decomposition Results");
    ImGui::Separator();

    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_ && time_series_.empty()) {
        ImGui::TextDisabled("Generate data and click 'Decompose' to see results");
        return;
    }

    // Show original data even without decomposition
    if (!has_result_) {
        RenderOriginalPlot();
        return;
    }

    if (ImGui::BeginTabBar("DecompTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " All Components")) {
            float plot_height = (ImGui::GetContentRegionAvail().y - 80) / 4.0f;

            if (ImPlot::BeginPlot("##Original", ImVec2(-1, plot_height))) {
                ImPlot::SetupAxes("Time", "Original");
                ImPlot::PlotLine("Original", time_axis_.data(), result_.original.data(),
                                static_cast<int>(result_.original.size()));
                ImPlot::EndPlot();
            }

            if (ImPlot::BeginPlot("##Trend", ImVec2(-1, plot_height))) {
                ImPlot::SetupAxes("Time", "Trend");
                ImPlot::PlotLine("Trend", time_axis_.data(), result_.trend.data(),
                                static_cast<int>(result_.trend.size()));
                ImPlot::EndPlot();
            }

            if (ImPlot::BeginPlot("##Seasonal", ImVec2(-1, plot_height))) {
                ImPlot::SetupAxes("Time", "Seasonal");
                ImPlot::PlotLine("Seasonal", time_axis_.data(), result_.seasonal.data(),
                                static_cast<int>(result_.seasonal.size()));
                ImPlot::EndPlot();
            }

            if (ImPlot::BeginPlot("##Residual", ImVec2(-1, plot_height))) {
                ImPlot::SetupAxes("Time", "Residual");
                ImPlot::PlotLine("Residual", time_axis_.data(), result_.residual.data(),
                                static_cast<int>(result_.residual.size()));
                ImPlot::EndPlot();
            }

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_CHART_SIMPLE " Original")) {
            RenderOriginalPlot();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_ARROW_TREND_UP " Trend")) {
            RenderTrendPlot();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_CALENDAR " Seasonal")) {
            RenderSeasonalPlot();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_SHUFFLE " Residual")) {
            RenderResidualPlot();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_CIRCLE_INFO " Statistics")) {
            RenderStatistics();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
}

void DecompositionPanel::RenderOriginalPlot() {
    if (time_series_.empty()) return;

    ImGui::Text("Original Time Series:");

    if (ImPlot::BeginPlot("##OriginalFull", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Time", "Value");
        ImPlot::PlotLine("Original", time_axis_.data(), time_series_.data(),
                        static_cast<int>(time_series_.size()));
        ImPlot::EndPlot();
    }
}

void DecompositionPanel::RenderTrendPlot() {
    if (!has_result_) return;

    ImGui::Text("Trend Component (strength: %.2f):", result_.trend_strength);

    if (ImPlot::BeginPlot("##TrendFull", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Time", "Trend");

        // Show original in background
        ImPlot::SetNextLineStyle(ImVec4(0.5f, 0.5f, 0.5f, 0.3f));
        ImPlot::PlotLine("Original", time_axis_.data(), result_.original.data(),
                        static_cast<int>(result_.original.size()));

        ImPlot::SetNextLineStyle(ImVec4(0.2f, 0.6f, 1.0f, 1.0f), 2.0f);
        ImPlot::PlotLine("Trend", time_axis_.data(), result_.trend.data(),
                        static_cast<int>(result_.trend.size()));
        ImPlot::EndPlot();
    }
}

void DecompositionPanel::RenderSeasonalPlot() {
    if (!has_result_) return;

    ImGui::Text("Seasonal Component (strength: %.2f, period: %d):",
                result_.seasonal_strength, result_.period);

    float height = ImGui::GetContentRegionAvail().y / 2 - 20;

    // Full seasonal
    if (ImPlot::BeginPlot("##SeasonalFull", ImVec2(-1, height))) {
        ImPlot::SetupAxes("Time", "Seasonal");
        ImPlot::PlotLine("Seasonal", time_axis_.data(), result_.seasonal.data(),
                        static_cast<int>(result_.seasonal.size()));
        ImPlot::EndPlot();
    }

    // One period
    ImGui::Text("Single Period Pattern:");
    if (result_.period > 0 && static_cast<int>(result_.seasonal.size()) >= result_.period) {
        std::vector<double> period_x(result_.period);
        std::vector<double> period_y(result_.period);
        for (int i = 0; i < result_.period; i++) {
            period_x[i] = i + 1;
            period_y[i] = result_.seasonal[i];
        }

        if (ImPlot::BeginPlot("##SeasonalPeriod", ImVec2(-1, height))) {
            ImPlot::SetupAxes("Period Index", "Seasonal Value");
            ImPlot::PlotBars("Pattern", period_x.data(), period_y.data(), result_.period, 0.6);
            ImPlot::EndPlot();
        }
    }
}

void DecompositionPanel::RenderResidualPlot() {
    if (!has_result_) return;

    ImGui::Text("Residual Component (variance: %.4f):", result_.residual_variance);

    float height = ImGui::GetContentRegionAvail().y / 2 - 20;

    // Residual time series
    if (ImPlot::BeginPlot("##ResidualFull", ImVec2(-1, height))) {
        ImPlot::SetupAxes("Time", "Residual");
        ImPlot::PlotLine("Residual", time_axis_.data(), result_.residual.data(),
                        static_cast<int>(result_.residual.size()));

        // Zero line
        double zero_x[] = {time_axis_.front(), time_axis_.back()};
        double zero_y[] = {0.0, 0.0};
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.0f, 0.0f, 0.5f), 1.0f);
        ImPlot::PlotLine("##zero", zero_x, zero_y, 2);

        ImPlot::EndPlot();
    }

    // Residual histogram
    ImGui::Text("Residual Distribution:");
    if (ImPlot::BeginPlot("##ResidualHist", ImVec2(-1, height))) {
        ImPlot::SetupAxes("Value", "Count");
        ImPlot::PlotHistogram("Residuals", result_.residual.data(),
                              static_cast<int>(result_.residual.size()), 30);
        ImPlot::EndPlot();
    }
}

void DecompositionPanel::RenderStatistics() {
    if (!has_result_) {
        ImGui::TextDisabled("Decompose data to see statistics");
        return;
    }

    ImGui::Text(ICON_FA_CIRCLE_INFO " Decomposition Statistics");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Columns(2, "stats_cols", false);

    ImGui::Text("Method:");
    ImGui::NextColumn();
    ImGui::Text("%s", result_.method.c_str());
    ImGui::NextColumn();

    ImGui::Text("Period:");
    ImGui::NextColumn();
    ImGui::Text("%d", result_.period);
    ImGui::NextColumn();

    ImGui::Text("Sample Size:");
    ImGui::NextColumn();
    ImGui::Text("%d", static_cast<int>(result_.original.size()));
    ImGui::NextColumn();

    ImGui::Separator();

    ImGui::Text("Trend Strength:");
    ImGui::NextColumn();
    ImGui::ProgressBar(static_cast<float>(result_.trend_strength), ImVec2(-1, 0),
                       (std::to_string(static_cast<int>(result_.trend_strength * 100)) + "%").c_str());
    ImGui::NextColumn();

    ImGui::Text("Seasonal Strength:");
    ImGui::NextColumn();
    ImGui::ProgressBar(static_cast<float>(result_.seasonal_strength), ImVec2(-1, 0),
                       (std::to_string(static_cast<int>(result_.seasonal_strength * 100)) + "%").c_str());
    ImGui::NextColumn();

    ImGui::Text("Residual Variance:");
    ImGui::NextColumn();
    ImGui::Text("%.6f", result_.residual_variance);
    ImGui::NextColumn();

    ImGui::Columns(1);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Interpretation
    ImGui::Text(ICON_FA_LIGHTBULB " Interpretation:");
    ImGui::Spacing();

    if (result_.trend_strength > 0.7) {
        ImGui::BulletText("Strong trend component detected");
    } else if (result_.trend_strength > 0.3) {
        ImGui::BulletText("Moderate trend present");
    } else {
        ImGui::BulletText("Weak or no trend");
    }

    if (result_.seasonal_strength > 0.7) {
        ImGui::BulletText("Strong seasonality with period %d", result_.period);
    } else if (result_.seasonal_strength > 0.3) {
        ImGui::BulletText("Moderate seasonality detected");
    } else {
        ImGui::BulletText("Weak or no seasonality");
    }

    if (result_.residual_variance < 0.1) {
        ImGui::BulletText("Good fit - low residual variance");
    } else if (result_.residual_variance < 0.5) {
        ImGui::BulletText("Acceptable fit");
    } else {
        ImGui::BulletText("High residual variance - consider different model");
    }
}

void DecompositionPanel::GenerateData() {
    time_series_.clear();
    time_axis_.clear();

    switch (signal_type_) {
        case SignalType::TrendSeasonal:
            time_series_ = TimeSeries::GenerateTrendSeasonal(
                num_samples_, trend_slope_, seasonal_amplitude_,
                seasonal_period_, noise_std_
            );
            period_ = seasonal_period_;
            break;

        case SignalType::RandomWalk:
            time_series_ = TimeSeries::GenerateRandomWalk(num_samples_, 0.0, noise_std_);
            break;

        case SignalType::WhiteNoise:
            time_series_ = TimeSeries::GenerateWhiteNoise(num_samples_, 0.0, noise_std_);
            break;

        case SignalType::AR2:
            time_series_ = TimeSeries::GenerateAR(num_samples_, {0.6, 0.2}, noise_std_);
            break;

        case SignalType::Seasonal:
            time_series_ = TimeSeries::GenerateTrendSeasonal(
                num_samples_, 0.0, seasonal_amplitude_,
                seasonal_period_, noise_std_
            );
            period_ = seasonal_period_;
            break;
    }

    time_axis_.resize(num_samples_);
    for (int i = 0; i < num_samples_; i++) {
        time_axis_[i] = static_cast<double>(i);
    }

    spdlog::info("Generated {} samples of type {}",
                 num_samples_, static_cast<int>(signal_type_));
}

void DecompositionPanel::DecomposeAsync() {
    if (is_computing_.load()) return;

    if (time_series_.empty()) {
        error_message_ = "No data to decompose";
        return;
    }

    if (static_cast<int>(time_series_.size()) < 2 * period_) {
        error_message_ = "Data length must be at least 2 * period";
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
            if (decomp_method_ == DecompMethod::Classical) {
                result_ = TimeSeries::Decompose(time_series_, period_, seasonal_type_);
            } else {
                result_ = TimeSeries::STLDecompose(
                    time_series_, period_, stl_seasonal_window_, stl_trend_window_
                );
            }

            if (result_.success) {
                has_result_ = true;
                spdlog::info("Decomposition complete: trend_strength={:.2f}, seasonal_strength={:.2f}",
                            result_.trend_strength, result_.seasonal_strength);
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
