#include "acf_pacf_panel.h"
#include "../icons.h"
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

ACFPACFPanel::ACFPACFPanel() {
    GenerateData();
    spdlog::info("ACFPACFPanel initialized");
}

ACFPACFPanel::~ACFPACFPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void ACFPACFPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 650), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CHART_BAR " ACF/PACF (Correlogram)###ACFPACFPanel", &visible_)) {
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

void ACFPACFPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Compute")) {
        ComputeAsync();
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

    ImGui::Text("Samples: %d | Max Lag: %d",
                static_cast<int>(time_series_.size()), max_lag_);

    if (has_result_) {
        ImGui::SameLine();
        ImGui::Text("| AR(p): %d, MA(q): %d",
                    result_.suggested_ar_order, result_.suggested_ma_order);
    }
}

void ACFPACFPanel::RenderInputPanel() {
    ImGui::Text(ICON_FA_DATABASE " Data Source");
    ImGui::Separator();

    const char* signal_types[] = {
        "AR(1)", "AR(2)", "MA(1)", "MA(2)", "ARMA(1,1)",
        "White Noise", "Random Walk"
    };
    int signal_idx = static_cast<int>(signal_type_);
    if (ImGui::Combo("Model", &signal_idx, signal_types, IM_ARRAYSIZE(signal_types))) {
        signal_type_ = static_cast<SignalType>(signal_idx);
    }

    ImGui::DragInt("Samples", &num_samples_, 1.0f, 50, 1000);
    static const double kNoiseMin = 0.1, kNoiseMax = 5.0;
    ImGui::SliderScalar("Noise Std", ImGuiDataType_Double, &noise_std_, &kNoiseMin, &kNoiseMax, "%.2f");

    static const double kCoeffMin = -0.99, kCoeffMax = 0.99;
    if (signal_type_ == SignalType::AR1 || signal_type_ == SignalType::AR2 ||
        signal_type_ == SignalType::ARMA) {
        ImGui::SliderScalar("AR1 Coeff", ImGuiDataType_Double, &ar1_coeff_, &kCoeffMin, &kCoeffMax, "%.2f");
    }

    if (signal_type_ == SignalType::AR2) {
        ImGui::SliderScalar("AR2 Coeff", ImGuiDataType_Double, &ar2_coeff_, &kCoeffMin, &kCoeffMax, "%.2f");
    }

    if (signal_type_ == SignalType::MA1 || signal_type_ == SignalType::MA2 ||
        signal_type_ == SignalType::ARMA) {
        ImGui::SliderScalar("MA1 Coeff", ImGuiDataType_Double, &ma1_coeff_, &kCoeffMin, &kCoeffMax, "%.2f");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text(ICON_FA_SLIDERS " ACF Settings");
    ImGui::Separator();

    ImGui::DragInt("Max Lag", &max_lag_, 1.0f, 5, 100);
    ImGui::SliderFloat("Confidence", reinterpret_cast<float*>(&confidence_level_), 0.90f, 0.99f, "%.2f");
}

void ACFPACFPanel::RenderLoadingIndicator() {
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() / 2 - 20);
    float width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX(width / 2 - 100);
    ImGui::Text(ICON_FA_SPINNER " Computing ACF/PACF...");
}

void ACFPACFPanel::RenderResults() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        ImGui::TextDisabled("Click 'Compute' to calculate ACF/PACF");
        return;
    }

    if (ImGui::BeginTabBar("ACFTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " ACF")) {
            RenderACFPlot();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " PACF")) {
            RenderPACFPlot();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_TABLE_COLUMNS " Both")) {
            RenderBothPlot();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CIRCLE_INFO " Analysis")) {
            RenderAnalysis();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void ACFPACFPanel::RenderACFPlot() {
    ImGui::Text("Autocorrelation Function (ACF):");

    if (ImPlot::BeginPlot("##ACF", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Lag", "ACF");
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.1, 1.1);

        // Confidence bounds (shaded)
        if (!result_.confidence_upper.empty()) {
            ImPlot::SetNextFillStyle(ImVec4(0.5f, 0.5f, 1.0f, 0.2f));
            std::vector<double> upper = result_.confidence_upper;
            std::vector<double> lower = result_.confidence_lower;
            ImPlot::PlotShaded("95% CI", result_.lags.data(), upper.data(), lower.data(),
                              static_cast<int>(result_.lags.size()));
        }

        // ACF bars
        ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.6f, 1.0f, 0.8f));
        ImPlot::PlotBars("ACF", result_.lags.data(), result_.acf.data(),
                        static_cast<int>(result_.acf.size()), 0.6);

        // Zero line
        double zero_x[] = {0, static_cast<double>(max_lag_)};
        double zero_y[] = {0, 0};
        ImPlot::SetNextLineStyle(ImVec4(1, 0, 0, 0.5f));
        ImPlot::PlotLine("##zero", zero_x, zero_y, 2);

        ImPlot::EndPlot();
    }
}

void ACFPACFPanel::RenderPACFPlot() {
    ImGui::Text("Partial Autocorrelation Function (PACF):");

    if (ImPlot::BeginPlot("##PACF", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Lag", "PACF");
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.1, 1.1);

        // Confidence bounds
        if (!result_.confidence_upper.empty()) {
            ImPlot::SetNextFillStyle(ImVec4(0.5f, 0.5f, 1.0f, 0.2f));
            ImPlot::PlotShaded("95% CI", result_.lags.data(),
                              result_.confidence_upper.data(),
                              result_.confidence_lower.data(),
                              static_cast<int>(result_.lags.size()));
        }

        // PACF bars
        ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.8f, 0.4f, 0.8f));
        ImPlot::PlotBars("PACF", result_.lags.data(), result_.pacf.data(),
                        static_cast<int>(result_.pacf.size()), 0.6);

        // Zero line
        double zero_x[] = {0, static_cast<double>(max_lag_)};
        double zero_y[] = {0, 0};
        ImPlot::SetNextLineStyle(ImVec4(1, 0, 0, 0.5f));
        ImPlot::PlotLine("##zero", zero_x, zero_y, 2);

        ImPlot::EndPlot();
    }
}

void ACFPACFPanel::RenderBothPlot() {
    float height = (ImGui::GetContentRegionAvail().y - 20) / 2;

    ImGui::Text("ACF:");
    if (ImPlot::BeginPlot("##ACFSmall", ImVec2(-1, height))) {
        ImPlot::SetupAxes("Lag", "ACF");
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.1, 1.1);

        if (!result_.confidence_upper.empty()) {
            ImPlot::SetNextFillStyle(ImVec4(0.5f, 0.5f, 1.0f, 0.2f));
            ImPlot::PlotShaded("##ci", result_.lags.data(),
                              result_.confidence_upper.data(),
                              result_.confidence_lower.data(),
                              static_cast<int>(result_.lags.size()));
        }

        ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.6f, 1.0f, 0.8f));
        ImPlot::PlotBars("ACF", result_.lags.data(), result_.acf.data(),
                        static_cast<int>(result_.acf.size()), 0.6);
        ImPlot::EndPlot();
    }

    ImGui::Text("PACF:");
    if (ImPlot::BeginPlot("##PACFSmall", ImVec2(-1, height))) {
        ImPlot::SetupAxes("Lag", "PACF");
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.1, 1.1);

        if (!result_.confidence_upper.empty()) {
            ImPlot::SetNextFillStyle(ImVec4(0.5f, 0.5f, 1.0f, 0.2f));
            ImPlot::PlotShaded("##ci", result_.lags.data(),
                              result_.confidence_upper.data(),
                              result_.confidence_lower.data(),
                              static_cast<int>(result_.lags.size()));
        }

        ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.8f, 0.4f, 0.8f));
        ImPlot::PlotBars("PACF", result_.lags.data(), result_.pacf.data(),
                        static_cast<int>(result_.pacf.size()), 0.6);
        ImPlot::EndPlot();
    }
}

void ACFPACFPanel::RenderAnalysis() {
    ImGui::Text(ICON_FA_CIRCLE_INFO " Model Identification");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Columns(2, "analysis_cols", false);

    ImGui::Text("Suggested AR order (p):");
    ImGui::NextColumn();
    ImGui::Text("%d", result_.suggested_ar_order);
    ImGui::NextColumn();

    ImGui::Text("Suggested MA order (q):");
    ImGui::NextColumn();
    ImGui::Text("%d", result_.suggested_ma_order);
    ImGui::NextColumn();

    ImGui::Text("Max Lag:");
    ImGui::NextColumn();
    ImGui::Text("%d", result_.max_lag);
    ImGui::NextColumn();

    ImGui::Separator();

    ImGui::Text("Ljung-Box p-value:");
    ImGui::NextColumn();
    ImGui::Text("%.4f", result_.ljung_box_pvalue);
    ImGui::NextColumn();

    ImGui::Columns(1);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text(ICON_FA_LIGHTBULB " Significant Lags:");
    ImGui::Spacing();

    if (!result_.significant_acf_lags.empty()) {
        ImGui::Text("ACF: ");
        ImGui::SameLine();
        for (size_t i = 0; i < result_.significant_acf_lags.size() && i < 10; i++) {
            ImGui::Text("%d", result_.significant_acf_lags[i]);
            if (i < result_.significant_acf_lags.size() - 1 && i < 9) {
                ImGui::SameLine();
                ImGui::Text(", ");
                ImGui::SameLine();
            }
        }
    } else {
        ImGui::TextDisabled("ACF: None significant");
    }

    if (!result_.significant_pacf_lags.empty()) {
        ImGui::Text("PACF: ");
        ImGui::SameLine();
        for (size_t i = 0; i < result_.significant_pacf_lags.size() && i < 10; i++) {
            ImGui::Text("%d", result_.significant_pacf_lags[i]);
            if (i < result_.significant_pacf_lags.size() - 1 && i < 9) {
                ImGui::SameLine();
                ImGui::Text(", ");
                ImGui::SameLine();
            }
        }
    } else {
        ImGui::TextDisabled("PACF: None significant");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text(ICON_FA_BOOK " Interpretation Guide:");
    ImGui::BulletText("AR(p): PACF cuts off after lag p, ACF decays");
    ImGui::BulletText("MA(q): ACF cuts off after lag q, PACF decays");
    ImGui::BulletText("ARMA: Both ACF and PACF decay");
    ImGui::BulletText("White noise: No significant lags");

    if (result_.ljung_box_pvalue > 0.05) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.2f, 0.8f, 0.2f, 1.0f));
        ImGui::Text(ICON_FA_CHECK " Ljung-Box: Cannot reject white noise (p > 0.05)");
        ImGui::PopStyleColor();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.6f, 0.2f, 1.0f));
        ImGui::Text(ICON_FA_TRIANGLE_EXCLAMATION " Ljung-Box: Significant autocorrelation (p < 0.05)");
        ImGui::PopStyleColor();
    }
}

void ACFPACFPanel::GenerateData() {
    time_series_.clear();

    switch (signal_type_) {
        case SignalType::AR1:
            time_series_ = TimeSeries::GenerateAR(num_samples_, {ar1_coeff_}, noise_std_);
            break;
        case SignalType::AR2:
            time_series_ = TimeSeries::GenerateAR(num_samples_, {ar1_coeff_, ar2_coeff_}, noise_std_);
            break;
        case SignalType::MA1:
            time_series_ = TimeSeries::GenerateMA(num_samples_, {ma1_coeff_}, noise_std_);
            break;
        case SignalType::MA2:
            time_series_ = TimeSeries::GenerateMA(num_samples_, {ma1_coeff_, 0.3}, noise_std_);
            break;
        case SignalType::ARMA:
            time_series_ = TimeSeries::GenerateARIMA(num_samples_, {ar1_coeff_}, {ma1_coeff_}, 0, noise_std_);
            break;
        case SignalType::WhiteNoise:
            time_series_ = TimeSeries::GenerateWhiteNoise(num_samples_, 0, noise_std_);
            break;
        case SignalType::RandomWalk:
            time_series_ = TimeSeries::GenerateRandomWalk(num_samples_, 0, noise_std_);
            break;
    }

    time_axis_.resize(num_samples_);
    for (int i = 0; i < num_samples_; i++) {
        time_axis_[i] = static_cast<double>(i);
    }
}

void ACFPACFPanel::ComputeAsync() {
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
            result_ = TimeSeries::ComputeACFPACF(time_series_, max_lag_);
            if (result_.success) {
                has_result_ = true;
                spdlog::info("ACF/PACF computed: AR(p)={}, MA(q)={}",
                            result_.suggested_ar_order, result_.suggested_ma_order);
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
