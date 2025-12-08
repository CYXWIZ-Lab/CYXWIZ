#include "spectrogram_panel.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

SpectrogramPanel::SpectrogramPanel() {
    GenerateSignal();
}

SpectrogramPanel::~SpectrogramPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        compute_thread_->join();
    }
}

void SpectrogramPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(950, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CHART_AREA " Spectrogram###SpectrogramPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            float panel_width = ImGui::GetContentRegionAvail().x;

            ImGui::BeginChild("InputPanel", ImVec2(panel_width * 0.28f, 0), true);
            RenderSignalInput();
            ImGui::Separator();
            RenderParameters();
            ImGui::EndChild();

            ImGui::SameLine();

            ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
            RenderResults();
            ImGui::EndChild();
        }
    }
    ImGui::End();
}

void SpectrogramPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Compute")) {
        ComputeAsync();
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_ROTATE " Generate")) {
        GenerateSignal();
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

    ImGui::Text("Duration: %.2f s", signal_.empty() ? 0.0 : signal_.size() / sample_rate_);
}

void SpectrogramPanel::RenderSignalInput() {
    ImGui::Text(ICON_FA_SIGNAL " Signal Generator");
    ImGui::Spacing();

    ImGui::Text("Signal Type:");
    const char* signal_types[] = { "Linear Chirp", "Exponential Chirp", "Multi-Tone", "AM Signal", "FM Signal" };
    int current_type = static_cast<int>(signal_type_);
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##SignalType", &current_type, signal_types, IM_ARRAYSIZE(signal_types))) {
        signal_type_ = static_cast<SignalType>(current_type);
    }

    ImGui::Spacing();

    // Parameters based on signal type
    switch (signal_type_) {
        case SignalType::ChirpLinear:
        case SignalType::ChirpExponential:
            ImGui::Text("Start Freq (Hz):");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputDouble("##startfreq", &start_freq_, 10.0, 50.0, "%.1f");
            start_freq_ = std::clamp(start_freq_, 10.0, sample_rate_ / 2.0);

            ImGui::Text("End Freq (Hz):");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputDouble("##endfreq", &end_freq_, 10.0, 50.0, "%.1f");
            end_freq_ = std::clamp(end_freq_, 10.0, sample_rate_ / 2.0);
            break;

        case SignalType::MultiTone:
            ImGui::Text("Base Freq (Hz):");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputDouble("##basefreq", &start_freq_, 10.0, 50.0, "%.1f");
            ImGui::TextDisabled("(Harmonics at 2x, 3x, 4x)");
            break;

        case SignalType::AM:
        case SignalType::FM:
            ImGui::Text("Carrier Freq (Hz):");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputDouble("##carrierfreq", &carrier_freq_, 10.0, 50.0, "%.1f");
            carrier_freq_ = std::clamp(carrier_freq_, 10.0, sample_rate_ / 2.0);

            ImGui::Text("Modulation Freq (Hz):");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputDouble("##modfreq", &modulation_freq_, 1.0, 10.0, "%.1f");
            modulation_freq_ = std::clamp(modulation_freq_, 0.1, 100.0);
            break;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Sample Rate (Hz):");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputDouble("##samplerate", &sample_rate_, 500.0, 2000.0, "%.0f")) {
        sample_rate_ = std::clamp(sample_rate_, 1000.0, 48000.0);
    }

    ImGui::Text("Duration (samples):");
    ImGui::SetNextItemWidth(-1);
    const char* durations[] = { "1024", "2048", "4096", "8192", "16384" };
    int dur_idx = 2;
    if (num_samples_ == 1024) dur_idx = 0;
    else if (num_samples_ == 2048) dur_idx = 1;
    else if (num_samples_ == 4096) dur_idx = 2;
    else if (num_samples_ == 8192) dur_idx = 3;
    else if (num_samples_ == 16384) dur_idx = 4;

    if (ImGui::Combo("##numsamples", &dur_idx, durations, IM_ARRAYSIZE(durations))) {
        int sizes[] = { 1024, 2048, 4096, 8192, 16384 };
        num_samples_ = sizes[dur_idx];
    }

    ImGui::Spacing();

    // Presets
    ImGui::Text("Presets:");
    if (ImGui::Button("Sweep 100-1000Hz", ImVec2(-1, 0))) {
        signal_type_ = SignalType::ChirpLinear;
        start_freq_ = 100.0;
        end_freq_ = 1000.0;
        sample_rate_ = 4000.0;
        num_samples_ = 8192;
        GenerateSignal();
    }
    if (ImGui::Button("FM Modulation", ImVec2(-1, 0))) {
        signal_type_ = SignalType::FM;
        carrier_freq_ = 500.0;
        modulation_freq_ = 5.0;
        sample_rate_ = 4000.0;
        num_samples_ = 4096;
        GenerateSignal();
    }
}

void SpectrogramPanel::RenderParameters() {
    ImGui::Text(ICON_FA_SLIDERS " STFT Parameters");
    ImGui::Spacing();

    ImGui::Text("Window Function:");
    const char* window_types[] = { "Hamming", "Hann", "Blackman", "Rectangular" };
    int current_window = static_cast<int>(window_type_);
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##WindowType", &current_window, window_types, IM_ARRAYSIZE(window_types))) {
        window_type_ = static_cast<WindowType>(current_window);
    }

    ImGui::Text("Window Size:");
    ImGui::SetNextItemWidth(-1);
    const char* win_sizes[] = { "64", "128", "256", "512", "1024" };
    int win_idx = 2;
    if (window_size_ == 64) win_idx = 0;
    else if (window_size_ == 128) win_idx = 1;
    else if (window_size_ == 256) win_idx = 2;
    else if (window_size_ == 512) win_idx = 3;
    else if (window_size_ == 1024) win_idx = 4;

    if (ImGui::Combo("##winsize", &win_idx, win_sizes, IM_ARRAYSIZE(win_sizes))) {
        int sizes[] = { 64, 128, 256, 512, 1024 };
        window_size_ = sizes[win_idx];
        hop_size_ = std::min(hop_size_, window_size_);
    }

    ImGui::Text("Hop Size:");
    ImGui::SetNextItemWidth(-1);
    ImGui::SliderInt("##hopsize", &hop_size_, 16, window_size_, "%d");
    hop_size_ = std::clamp(hop_size_, 16, window_size_);

    ImGui::Spacing();

    // Overlap percentage
    float overlap = 100.0f * (1.0f - static_cast<float>(hop_size_) / window_size_);
    ImGui::Text("Overlap: %.1f%%", overlap);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text(ICON_FA_PALETTE " Display");

    ImGui::Checkbox("Log Scale (dB)", &log_scale_);

    if (log_scale_) {
        ImGui::Text("Dynamic Range:");
        ImGui::SetNextItemWidth(-1);
        static const double kDynMin = 20.0, kDynMax = 120.0;
        ImGui::SliderScalar("##dynrange", ImGuiDataType_Double, &dynamic_range_, &kDynMin, &kDynMax, "%.0f dB");
    }

    ImGui::Text("Color Scale:");
    const char* color_scales[] = { "Viridis", "Plasma", "Inferno", "Grayscale" };
    int current_color = static_cast<int>(color_scale_);
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##ColorScale", &current_color, color_scales, IM_ARRAYSIZE(color_scales))) {
        color_scale_ = static_cast<ColorScale>(current_color);
    }
}

void SpectrogramPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Computing spectrogram...", ICON_FA_SPINNER);
}

void SpectrogramPanel::RenderResults() {
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

    if (ImGui::BeginTabBar("SpectrogramTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_IMAGE " Spectrogram")) {
            RenderSpectrogram();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_WAVE_SQUARE " Signal")) {
            RenderSignalPreview();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void SpectrogramPanel::RenderSpectrogram() {
    if (!has_result_) {
        ImGui::TextDisabled("Click 'Compute' to generate spectrogram");
        return;
    }

    // Info
    ImGui::Text("Time frames: %d | Frequency bins: %d", result_.num_frames, result_.num_bins);

    // Prepare data for heatmap
    int rows = result_.num_bins;
    int cols = result_.num_frames;

    // Flatten data for ImPlot heatmap (row-major)
    std::vector<double> flat_data(rows * cols);

    double max_val = -1e10;
    double min_val = 1e10;

    for (int f = 0; f < rows; f++) {
        for (int t = 0; t < cols; t++) {
            double val = result_.spectrogram[t][f];
            if (log_scale_) {
                val = 20.0 * std::log10(std::max(val, 1e-10));
            }
            flat_data[f * cols + t] = val;
            max_val = std::max(max_val, val);
            min_val = std::min(min_val, val);
        }
    }

    // Apply dynamic range limit for log scale
    if (log_scale_) {
        double floor = max_val - dynamic_range_;
        for (auto& v : flat_data) {
            v = std::max(v, floor);
        }
        min_val = floor;
    }

    // Set colormap based on selection
    ImPlotColormap colormap = ImPlotColormap_Viridis;
    switch (color_scale_) {
        case ColorScale::Viridis: colormap = ImPlotColormap_Viridis; break;
        case ColorScale::Plasma: colormap = ImPlotColormap_Plasma; break;
        case ColorScale::Inferno: colormap = ImPlotColormap_Hot; break;  // ImPlot doesn't have Inferno, use Hot
        case ColorScale::Grayscale: colormap = ImPlotColormap_Greys; break;
    }

    ImPlot::PushColormap(colormap);

    // Time and frequency bounds
    double t_min = result_.times.empty() ? 0 : result_.times.front();
    double t_max = result_.times.empty() ? 1 : result_.times.back();
    double f_min = result_.frequencies.empty() ? 0 : result_.frequencies.front();
    double f_max = result_.frequencies.empty() ? sample_rate_ / 2 : result_.frequencies.back();

    if (ImPlot::BeginPlot("##Spectrogram", ImVec2(-1, 400))) {
        ImPlot::SetupAxes("Time (s)", "Frequency (Hz)");
        ImPlot::SetupAxisLimits(ImAxis_X1, t_min, t_max, ImPlotCond_Once);
        ImPlot::SetupAxisLimits(ImAxis_Y1, f_min, f_max, ImPlotCond_Once);

        // Plot heatmap
        ImPlot::PlotHeatmap("##heatmap", flat_data.data(), rows, cols,
                           min_val, max_val,
                           nullptr,  // label format
                           ImPlotPoint(t_min, f_min),
                           ImPlotPoint(t_max, f_max));

        ImPlot::EndPlot();
    }

    // Color scale legend
    ImPlot::ColormapScale(log_scale_ ? "##scale_db" : "##scale_linear",
                          min_val, max_val, ImVec2(60, 400));

    ImPlot::PopColormap();

    // Additional info
    ImGui::Spacing();
    ImGui::Text("Frequency resolution: %.2f Hz", sample_rate_ / window_size_);
    ImGui::Text("Time resolution: %.4f s", static_cast<double>(hop_size_) / sample_rate_);
}

void SpectrogramPanel::RenderSignalPreview() {
    ImGui::Text("Input Signal Preview:");

    if (ImPlot::BeginPlot("##SignalPreview", ImVec2(-1, 300))) {
        ImPlot::SetupAxes("Time (s)", "Amplitude");

        if (!signal_.empty() && !time_axis_.empty()) {
            ImPlot::PlotLine("Signal", time_axis_.data(), signal_.data(),
                             static_cast<int>(signal_.size()));
        }
        ImPlot::EndPlot();
    }

    // Statistics
    if (!signal_.empty()) {
        double sum = 0, sum_sq = 0;
        for (double v : signal_) {
            sum += v;
            sum_sq += v * v;
        }
        double mean = sum / signal_.size();
        double rms = std::sqrt(sum_sq / signal_.size());

        ImGui::Text("Samples: %d | Duration: %.3f s | RMS: %.4f",
                    static_cast<int>(signal_.size()),
                    signal_.size() / sample_rate_,
                    rms);
    }
}

void SpectrogramPanel::GenerateSignal() {
    signal_.clear();
    time_axis_.clear();

    const double PI = 3.14159265358979323846;
    double dt = 1.0 / sample_rate_;
    double duration = num_samples_ * dt;

    for (int i = 0; i < num_samples_; i++) {
        double t = i * dt;
        time_axis_.push_back(t);

        double value = 0.0;

        switch (signal_type_) {
            case SignalType::ChirpLinear: {
                // Linear frequency sweep: f(t) = f0 + (f1 - f0) * t / T
                double k = (end_freq_ - start_freq_) / duration;
                double phase = 2.0 * PI * (start_freq_ * t + 0.5 * k * t * t);
                value = std::sin(phase);
                break;
            }

            case SignalType::ChirpExponential: {
                // Exponential frequency sweep
                double k = std::pow(end_freq_ / start_freq_, 1.0 / duration);
                double phase = 2.0 * PI * start_freq_ * (std::pow(k, t) - 1.0) / std::log(k);
                value = std::sin(phase);
                break;
            }

            case SignalType::MultiTone: {
                // Multiple harmonics
                value = 0.5 * std::sin(2.0 * PI * start_freq_ * t) +
                        0.3 * std::sin(2.0 * PI * start_freq_ * 2 * t) +
                        0.15 * std::sin(2.0 * PI * start_freq_ * 3 * t) +
                        0.05 * std::sin(2.0 * PI * start_freq_ * 4 * t);
                break;
            }

            case SignalType::AM: {
                // Amplitude modulation
                double modulation = 1.0 + 0.5 * std::sin(2.0 * PI * modulation_freq_ * t);
                value = modulation * std::sin(2.0 * PI * carrier_freq_ * t);
                break;
            }

            case SignalType::FM: {
                // Frequency modulation
                double deviation = carrier_freq_ * 0.5;  // Modulation index
                double phase = 2.0 * PI * carrier_freq_ * t +
                               (deviation / modulation_freq_) * std::sin(2.0 * PI * modulation_freq_ * t);
                value = std::sin(phase);
                break;
            }
        }

        signal_.push_back(value);
    }

    spdlog::info("Generated {} samples at {} Hz", num_samples_, sample_rate_);
}

void SpectrogramPanel::ComputeAsync() {
    if (is_computing_.load()) return;
    if (signal_.empty()) {
        error_message_ = "No signal data to process";
        return;
    }

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_result_ = false;
    error_message_.clear();

    // Capture parameters for thread
    int win_size = window_size_;
    int hop = hop_size_;
    double sr = sample_rate_;
    WindowType win_type = window_type_;

    compute_thread_ = std::make_unique<std::thread>([this, win_size, hop, sr, win_type]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            // Get window function name
            std::string window_name;
            switch (win_type) {
                case WindowType::Hamming: window_name = "hamming"; break;
                case WindowType::Hann: window_name = "hann"; break;
                case WindowType::Blackman: window_name = "blackman"; break;
                case WindowType::Rectangular: window_name = "rectangular"; break;
            }

            result_ = SignalProcessing::ComputeSpectrogram(signal_, win_size, hop, sr, window_name);
            if (result_.success) {
                has_result_ = true;
                spdlog::info("Spectrogram computed: {} frames x {} bins", result_.num_frames, result_.num_bins);
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
