#include "wavelet_panel.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace cyxwiz {

// Static constants for slider ranges
static const double kNoiseMin = 0.0;
static const double kNoiseMax = 1.0;
static const double kThresholdMin = 0.01;
static const double kThresholdMax = 1.0;

WaveletPanel::WaveletPanel() {
    GenerateSignal();
}

WaveletPanel::~WaveletPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        compute_thread_->join();
    }
}

void WaveletPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(950, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_WATER " Wavelet Transform###WaveletPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            float panel_width = ImGui::GetContentRegionAvail().x;

            ImGui::BeginChild("SettingsPanel", ImVec2(panel_width * 0.28f, 0), true);
            RenderSignalInput();
            ImGui::Separator();
            RenderWaveletSettings();
            ImGui::EndChild();

            ImGui::SameLine();

            ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
            RenderResults();
            ImGui::EndChild();
        }
    }
    ImGui::End();
}

void WaveletPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Decompose")) {
        DecomposeAsync();
    }

    ImGui::SameLine();

    if (has_result_ && ImGui::Button(ICON_FA_ROTATE_LEFT " Reconstruct")) {
        ReconstructSignal();
    }

    ImGui::SameLine();

    if (has_result_ && ImGui::Button(ICON_FA_BRUSH " Denoise")) {
        ApplyDenoising();
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        has_result_ = false;
        has_reconstruction_ = false;
        has_denoised_ = false;
        error_message_.clear();
    }
}

void WaveletPanel::RenderSignalInput() {
    ImGui::Text(ICON_FA_SIGNAL " Signal");
    ImGui::Spacing();

    ImGui::Text("Type:");
    const char* signal_types[] = { "Sine", "Step", "Noisy Sine", "Chirp", "Custom" };
    int current_type = static_cast<int>(signal_type_);
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##SignalType", &current_type, signal_types, IM_ARRAYSIZE(signal_types))) {
        signal_type_ = static_cast<SignalType>(current_type);
        if (signal_type_ != SignalType::Custom) {
            GenerateSignal();
            has_result_ = false;
        }
    }

    if (signal_type_ != SignalType::Custom) {
        ImGui::Text("Length:");
        ImGui::SetNextItemWidth(-1);
        const char* lengths[] = { "64", "128", "256", "512", "1024" };
        int len_idx = 2;
        if (signal_length_ == 64) len_idx = 0;
        else if (signal_length_ == 128) len_idx = 1;
        else if (signal_length_ == 256) len_idx = 2;
        else if (signal_length_ == 512) len_idx = 3;
        else if (signal_length_ == 1024) len_idx = 4;

        if (ImGui::Combo("##Length", &len_idx, lengths, IM_ARRAYSIZE(lengths))) {
            int lens[] = { 64, 128, 256, 512, 1024 };
            signal_length_ = lens[len_idx];
            GenerateSignal();
            has_result_ = false;
        }

        if (signal_type_ != SignalType::Step) {
            ImGui::Text("Frequency:");
            ImGui::SetNextItemWidth(-1);
            if (ImGui::InputDouble("##Freq", &signal_freq_, 1.0, 5.0, "%.1f")) {
                signal_freq_ = std::clamp(signal_freq_, 1.0, 50.0);
                GenerateSignal();
                has_result_ = false;
            }
        }

        if (signal_type_ == SignalType::Noisy) {
            ImGui::Text("Noise Level:");
            ImGui::SetNextItemWidth(-1);
            if (ImGui::SliderScalar("##Noise", ImGuiDataType_Double, &noise_level_, &kNoiseMin, &kNoiseMax, "%.2f")) {
                GenerateSignal();
                has_result_ = false;
            }
        }
    }

    ImGui::Spacing();

    if (ImGui::Button(ICON_FA_ROTATE " Regenerate", ImVec2(-1, 0))) {
        GenerateSignal();
        has_result_ = false;
        has_reconstruction_ = false;
        has_denoised_ = false;
    }

    // Presets
    ImGui::Spacing();
    ImGui::Text("Presets:");
    if (ImGui::Button("Noisy Signal", ImVec2(-1, 0))) {
        signal_type_ = SignalType::Noisy;
        signal_freq_ = 10.0;
        noise_level_ = 0.3;
        signal_length_ = 256;
        GenerateSignal();
        has_result_ = false;
    }
    if (ImGui::Button("Clean Sine", ImVec2(-1, 0))) {
        signal_type_ = SignalType::Sine;
        signal_freq_ = 5.0;
        signal_length_ = 256;
        GenerateSignal();
        has_result_ = false;
    }
}

void WaveletPanel::RenderWaveletSettings() {
    ImGui::Text(ICON_FA_SLIDERS " Wavelet Settings");
    ImGui::Spacing();

    ImGui::Text("Wavelet Family:");
    const char* wavelets[] = { "Haar", "db1", "db2", "db3", "db4" };
    int current_wavelet = static_cast<int>(wavelet_type_);
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##Wavelet", &current_wavelet, wavelets, IM_ARRAYSIZE(wavelets))) {
        wavelet_type_ = static_cast<WaveletType>(current_wavelet);
        has_result_ = false;
    }

    ImGui::Spacing();

    ImGui::Text("Decomposition Levels:");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::SliderInt("##Levels", &decomp_levels_, 1, 6)) {
        has_result_ = false;
    }

    // Max levels based on signal length
    int max_levels = static_cast<int>(std::log2(signal_length_)) - 2;
    if (decomp_levels_ > max_levels) {
        decomp_levels_ = max_levels;
    }
    ImGui::TextDisabled("(max %d for length %d)", max_levels, signal_length_);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Denoising settings
    ImGui::Text(ICON_FA_BRUSH " Denoising");
    ImGui::Spacing();

    ImGui::Text("Threshold:");
    ImGui::SetNextItemWidth(-1);
    ImGui::SliderScalar("##Threshold", ImGuiDataType_Double, &threshold_, &kThresholdMin, &kThresholdMax, "%.3f");

    ImGui::Text("Type:");
    const char* thresh_types[] = { "Hard", "Soft" };
    int current_thresh = static_cast<int>(threshold_type_);
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##ThreshType", &current_thresh, thresh_types, IM_ARRAYSIZE(thresh_types))) {
        threshold_type_ = static_cast<ThresholdType>(current_thresh);
    }
}

void WaveletPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Computing wavelet transform...", ICON_FA_SPINNER);
}

void WaveletPanel::RenderResults() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (signal_.empty()) {
        ImGui::TextDisabled("No signal data. Generate a signal first.");
        return;
    }

    if (ImGui::BeginTabBar("WaveletTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_LAYER_GROUP " Decomposition")) {
            RenderDecomposition();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_ROTATE_LEFT " Reconstruction")) {
            RenderReconstruction();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_BRUSH " Denoising")) {
            RenderDenoising();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void WaveletPanel::RenderDecomposition() {
    // Original signal
    ImGui::Text("Original Signal:");
    if (ImPlot::BeginPlot("##Original", ImVec2(-1, 150))) {
        ImPlot::SetupAxes("Sample", "Amplitude");
        if (!signal_.empty() && !time_axis_.empty()) {
            ImPlot::PlotLine("Signal", time_axis_.data(), signal_.data(),
                             static_cast<int>(signal_.size()));
        }
        ImPlot::EndPlot();
    }

    if (!has_result_) {
        ImGui::TextDisabled("Click 'Decompose' to compute DWT");
        return;
    }

    ImGui::Spacing();
    ImGui::Text("Wavelet Decomposition (%d levels):", result_.levels);

    // Approximation coefficients
    ImGui::Text("Approximation (cA%d):", result_.levels);
    if (ImPlot::BeginPlot("##Approx", ImVec2(-1, 100))) {
        ImPlot::SetupAxes("Index", "Value");
        if (!result_.approximation.empty()) {
            std::vector<double> indices(result_.approximation.size());
            std::iota(indices.begin(), indices.end(), 0);
            ImPlot::PlotLine("cA", indices.data(), result_.approximation.data(),
                             static_cast<int>(result_.approximation.size()));
        }
        ImPlot::EndPlot();
    }

    // Detail coefficients for each level
    for (int level = 0; level < result_.levels; level++) {
        if (level < static_cast<int>(result_.details.size())) {
            ImGui::Text("Detail (cD%d):", level + 1);
            char plot_id[32];
            snprintf(plot_id, sizeof(plot_id), "##Detail%d", level);
            if (ImPlot::BeginPlot(plot_id, ImVec2(-1, 80))) {
                ImPlot::SetupAxes("Index", "Value");
                const auto& detail = result_.details[level];
                std::vector<double> indices(detail.size());
                std::iota(indices.begin(), indices.end(), 0);
                ImPlot::PlotLine("cD", indices.data(), detail.data(),
                                 static_cast<int>(detail.size()));
                ImPlot::EndPlot();
            }
        }
    }

    // Coefficient statistics
    ImGui::Spacing();
    ImGui::Text("Statistics:");
    ImGui::BulletText("Approximation: %zu coefficients", result_.approximation.size());
    for (int level = 0; level < result_.levels && level < static_cast<int>(result_.details.size()); level++) {
        double energy = 0;
        for (double d : result_.details[level]) {
            energy += d * d;
        }
        ImGui::BulletText("Detail L%d: %zu coeffs, energy: %.4f",
                          level + 1, result_.details[level].size(), energy);
    }
}

void WaveletPanel::RenderReconstruction() {
    if (!has_result_) {
        ImGui::TextDisabled("Decompose the signal first");
        return;
    }

    if (!has_reconstruction_) {
        ImGui::TextDisabled("Click 'Reconstruct' to rebuild signal from coefficients");
        return;
    }

    // Comparison plot
    ImGui::Text("Original vs Reconstructed:");
    if (ImPlot::BeginPlot("##Comparison", ImVec2(-1, 250))) {
        ImPlot::SetupAxes("Sample", "Amplitude");
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);

        if (!signal_.empty() && !time_axis_.empty()) {
            ImPlot::PlotLine("Original", time_axis_.data(), signal_.data(),
                             static_cast<int>(signal_.size()));
        }
        if (!reconstructed_.empty()) {
            ImPlot::PlotLine("Reconstructed", time_axis_.data(), reconstructed_.data(),
                             static_cast<int>(reconstructed_.size()));
        }
        ImPlot::EndPlot();
    }

    // Error plot
    ImGui::Text("Reconstruction Error:");
    if (ImPlot::BeginPlot("##Error", ImVec2(-1, 150))) {
        ImPlot::SetupAxes("Sample", "Error");

        if (signal_.size() == reconstructed_.size()) {
            std::vector<double> error(signal_.size());
            for (size_t i = 0; i < signal_.size(); i++) {
                error[i] = signal_[i] - reconstructed_[i];
            }
            ImPlot::PlotLine("Error", time_axis_.data(), error.data(),
                             static_cast<int>(error.size()));
        }
        ImPlot::EndPlot();
    }

    // Statistics
    ImGui::Text("Reconstruction Error: %.2e (MSE)", reconstruction_error_);
    ImGui::Text("PSNR: %.2f dB", reconstruction_error_ > 0 ? 10 * std::log10(1.0 / reconstruction_error_) : 999.0);
}

void WaveletPanel::RenderDenoising() {
    if (!has_result_) {
        ImGui::TextDisabled("Decompose the signal first");
        return;
    }

    ImGui::Text("Wavelet Denoising:");
    ImGui::TextWrapped("Applies thresholding to detail coefficients to remove noise.");

    ImGui::Spacing();

    if (!has_denoised_) {
        ImGui::TextDisabled("Click 'Denoise' to apply thresholding");

        // Show original for context
        ImGui::Text("Original Signal:");
        if (ImPlot::BeginPlot("##OrigDenoise", ImVec2(-1, 200))) {
            ImPlot::SetupAxes("Sample", "Amplitude");
            if (!signal_.empty() && !time_axis_.empty()) {
                ImPlot::PlotLine("Signal", time_axis_.data(), signal_.data(),
                                 static_cast<int>(signal_.size()));
            }
            ImPlot::EndPlot();
        }
        return;
    }

    // Comparison plot
    ImGui::Text("Original vs Denoised:");
    if (ImPlot::BeginPlot("##Denoised", ImVec2(-1, 250))) {
        ImPlot::SetupAxes("Sample", "Amplitude");
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);

        if (!signal_.empty() && !time_axis_.empty()) {
            ImPlot::PlotLine("Noisy", time_axis_.data(), signal_.data(),
                             static_cast<int>(signal_.size()));
        }
        if (!denoised_.empty()) {
            ImPlot::PlotLine("Denoised", time_axis_.data(), denoised_.data(),
                             static_cast<int>(denoised_.size()));
        }
        ImPlot::EndPlot();
    }

    // Statistics
    double orig_rms = 0, denoised_rms = 0;
    for (double v : signal_) orig_rms += v * v;
    for (double v : denoised_) denoised_rms += v * v;
    orig_rms = std::sqrt(orig_rms / signal_.size());
    denoised_rms = std::sqrt(denoised_rms / denoised_.size());

    ImGui::Text("Original RMS: %.4f", orig_rms);
    ImGui::Text("Denoised RMS: %.4f", denoised_rms);
    ImGui::Text("Threshold: %.4f (%s)", threshold_,
                threshold_type_ == ThresholdType::Soft ? "soft" : "hard");
}

void WaveletPanel::GenerateSignal() {
    signal_.resize(signal_length_);
    time_axis_.resize(signal_length_);

    const double PI = 3.14159265358979323846;
    static std::mt19937 rng(42);
    std::normal_distribution<double> noise_dist(0.0, 1.0);

    for (int i = 0; i < signal_length_; i++) {
        double t = static_cast<double>(i) / signal_length_;
        time_axis_[i] = i;

        switch (signal_type_) {
            case SignalType::Sine:
                signal_[i] = std::sin(2.0 * PI * signal_freq_ * t);
                break;

            case SignalType::Step:
                signal_[i] = (i < signal_length_ / 2) ? 0.0 : 1.0;
                break;

            case SignalType::Noisy:
                signal_[i] = std::sin(2.0 * PI * signal_freq_ * t) +
                             noise_level_ * noise_dist(rng);
                break;

            case SignalType::Chirp: {
                double f = signal_freq_ * (1.0 + t);  // Linear chirp
                signal_[i] = std::sin(2.0 * PI * f * t);
                break;
            }

            case SignalType::Custom:
                // Keep existing
                break;
        }
    }

    has_result_ = false;
    has_reconstruction_ = false;
    has_denoised_ = false;
}

void WaveletPanel::DecomposeAsync() {
    if (is_computing_.load()) return;
    if (signal_.empty()) {
        error_message_ = "No signal data";
        return;
    }

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_result_ = false;
    has_reconstruction_ = false;
    has_denoised_ = false;
    error_message_.clear();

    // Get wavelet name
    std::string wavelet_name;
    switch (wavelet_type_) {
        case WaveletType::Haar: wavelet_name = "haar"; break;
        case WaveletType::DB1: wavelet_name = "db1"; break;
        case WaveletType::DB2: wavelet_name = "db2"; break;
        case WaveletType::DB3: wavelet_name = "db3"; break;
        case WaveletType::DB4: wavelet_name = "db4"; break;
    }

    int levels = decomp_levels_;

    compute_thread_ = std::make_unique<std::thread>([this, wavelet_name, levels]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            result_ = SignalProcessing::DWT(signal_, wavelet_name, levels);
            if (result_.success) {
                has_result_ = true;
                spdlog::info("DWT computed: {} levels", result_.levels);
            } else {
                error_message_ = result_.error_message;
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void WaveletPanel::ReconstructSignal() {
    if (!has_result_) return;

    try {
        reconstructed_ = SignalProcessing::IDWT(result_);

        // Compute MSE
        if (reconstructed_.size() >= signal_.size()) {
            reconstruction_error_ = 0;
            for (size_t i = 0; i < signal_.size(); i++) {
                double diff = signal_[i] - reconstructed_[i];
                reconstruction_error_ += diff * diff;
            }
            reconstruction_error_ /= signal_.size();
        }

        // Trim to original length
        if (reconstructed_.size() > signal_.size()) {
            reconstructed_.resize(signal_.size());
        }

        has_reconstruction_ = true;
        spdlog::info("Signal reconstructed from DWT coefficients");
    } catch (const std::exception& e) {
        error_message_ = std::string("Reconstruction failed: ") + e.what();
    }
}

void WaveletPanel::ApplyDenoising() {
    if (!has_result_) return;

    try {
        // Create a copy of result for thresholding
        WaveletResult thresholded = result_;

        // Apply thresholding to detail coefficients
        for (auto& detail : thresholded.details) {
            for (auto& d : detail) {
                if (threshold_type_ == ThresholdType::Hard) {
                    // Hard thresholding: set to zero if below threshold
                    if (std::abs(d) < threshold_) {
                        d = 0.0;
                    }
                } else {
                    // Soft thresholding: shrink towards zero
                    if (std::abs(d) < threshold_) {
                        d = 0.0;
                    } else if (d > 0) {
                        d -= threshold_;
                    } else {
                        d += threshold_;
                    }
                }
            }
        }

        // Reconstruct from thresholded coefficients
        denoised_ = SignalProcessing::IDWT(thresholded);

        // Trim to original length
        if (denoised_.size() > signal_.size()) {
            denoised_.resize(signal_.size());
        }

        has_denoised_ = true;
        spdlog::info("Denoising applied with threshold {}", threshold_);
    } catch (const std::exception& e) {
        error_message_ = std::string("Denoising failed: ") + e.what();
    }
}

} // namespace cyxwiz
