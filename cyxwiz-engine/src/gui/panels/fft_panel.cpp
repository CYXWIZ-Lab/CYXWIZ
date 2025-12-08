#include "fft_panel.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace cyxwiz {

FFTPanel::FFTPanel() {
    // Generate default signal
    GenerateSignal();
}

FFTPanel::~FFTPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        compute_thread_->join();
    }
}

void FFTPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_WAVE_SQUARE " FFT (Fast Fourier Transform)###FFTPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            float panel_width = ImGui::GetContentRegionAvail().x;

            ImGui::BeginChild("InputPanel", ImVec2(panel_width * 0.30f, 0), true);
            RenderSignalInput();
            ImGui::EndChild();

            ImGui::SameLine();

            ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
            RenderResults();
            ImGui::EndChild();
        }
    }
    ImGui::End();
}

void FFTPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Compute FFT")) {
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
        peaks_.clear();
        error_message_.clear();
    }

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    ImGui::Text("Samples: %d", static_cast<int>(signal_.size()));

    if (has_result_) {
        ImGui::SameLine();
        ImGui::Text("| Freq bins: %d", result_.n / 2);
    }
}

void FFTPanel::RenderSignalInput() {
    ImGui::Text(ICON_FA_SIGNAL " Signal Input");
    ImGui::Separator();

    // Input mode selection
    if (ImGui::RadioButton("Generate", input_mode_ == InputMode::Generate)) {
        input_mode_ = InputMode::Generate;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Manual", input_mode_ == InputMode::Manual)) {
        input_mode_ = InputMode::Manual;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (input_mode_ == InputMode::Generate) {
        RenderGenerateSignal();
    } else {
        RenderManualInput();
    }
}

void FFTPanel::RenderGenerateSignal() {
    ImGui::Text("Signal Type:");
    const char* signal_types[] = { "Sine", "Square", "Sawtooth", "White Noise", "Composite" };
    int current_type = static_cast<int>(signal_type_);
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##SignalType", &current_type, signal_types, IM_ARRAYSIZE(signal_types))) {
        signal_type_ = static_cast<SignalType>(current_type);
    }

    ImGui::Spacing();

    // Parameters based on signal type
    if (signal_type_ != SignalType::WhiteNoise) {
        ImGui::Text("Frequency 1 (Hz):");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputDouble("##freq1", &frequency1_, 1.0, 10.0, "%.1f");
        frequency1_ = std::clamp(frequency1_, 1.0, sample_rate_ / 2.0);

        ImGui::Text("Amplitude 1:");
        ImGui::SetNextItemWidth(-1);
        static const double kAmpMin = 0.0, kAmpMax = 2.0;
        ImGui::SliderScalar("##amp1", ImGuiDataType_Double, &amplitude1_, &kAmpMin, &kAmpMax, "%.2f");
    }

    if (signal_type_ == SignalType::Composite) {
        ImGui::Spacing();
        ImGui::Text("Frequency 2 (Hz):");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputDouble("##freq2", &frequency2_, 1.0, 10.0, "%.1f");
        frequency2_ = std::clamp(frequency2_, 1.0, sample_rate_ / 2.0);

        ImGui::Text("Amplitude 2:");
        ImGui::SetNextItemWidth(-1);
        static const double kAmp2Min = 0.0, kAmp2Max = 2.0;
        ImGui::SliderScalar("##amp2", ImGuiDataType_Double, &amplitude2_, &kAmp2Min, &kAmp2Max, "%.2f");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Sample Rate (Hz):");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputDouble("##samplerate", &sample_rate_, 100.0, 1000.0, "%.0f")) {
        sample_rate_ = std::clamp(sample_rate_, 100.0, 100000.0);
    }

    ImGui::Text("Number of Samples:");
    ImGui::SetNextItemWidth(-1);
    const char* sample_sizes[] = { "128", "256", "512", "1024", "2048", "4096" };
    int sample_idx = 3; // Default to 1024
    if (num_samples_ == 128) sample_idx = 0;
    else if (num_samples_ == 256) sample_idx = 1;
    else if (num_samples_ == 512) sample_idx = 2;
    else if (num_samples_ == 1024) sample_idx = 3;
    else if (num_samples_ == 2048) sample_idx = 4;
    else if (num_samples_ == 4096) sample_idx = 5;

    if (ImGui::Combo("##numsamples", &sample_idx, sample_sizes, IM_ARRAYSIZE(sample_sizes))) {
        int sizes[] = { 128, 256, 512, 1024, 2048, 4096 };
        num_samples_ = sizes[sample_idx];
    }

    ImGui::Spacing();

    // Noise option
    ImGui::Checkbox("Add Noise", &add_noise_);
    if (add_noise_) {
        ImGui::Text("SNR (dB):");
        ImGui::SetNextItemWidth(-1);
        static const double kSnrMin = 0.0, kSnrMax = 40.0;
        ImGui::SliderScalar("##snr", ImGuiDataType_Double, &noise_snr_, &kSnrMin, &kSnrMax, "%.1f");
    }

    ImGui::Spacing();

    // Quick presets
    ImGui::Text("Presets:");
    if (ImGui::Button("60Hz Hum", ImVec2(-1, 0))) {
        signal_type_ = SignalType::Composite;
        frequency1_ = 60.0;
        frequency2_ = 120.0;
        amplitude1_ = 1.0;
        amplitude2_ = 0.3;
        sample_rate_ = 1000.0;
        num_samples_ = 1024;
        GenerateSignal();
    }
    if (ImGui::Button("A440 Note", ImVec2(-1, 0))) {
        signal_type_ = SignalType::Sine;
        frequency1_ = 440.0;
        amplitude1_ = 1.0;
        sample_rate_ = 8000.0;
        num_samples_ = 2048;
        GenerateSignal();
    }
    if (ImGui::Button("Square Wave", ImVec2(-1, 0))) {
        signal_type_ = SignalType::Square;
        frequency1_ = 100.0;
        amplitude1_ = 1.0;
        sample_rate_ = 2000.0;
        num_samples_ = 1024;
        GenerateSignal();
    }
}

void FFTPanel::RenderManualInput() {
    ImGui::Text("Sample Rate (Hz):");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputDouble("##samplerate_manual", &sample_rate_, 100.0, 1000.0, "%.0f");
    sample_rate_ = std::clamp(sample_rate_, 100.0, 100000.0);

    ImGui::Spacing();

    ImGui::Text("Signal Values (one per line):");

    static char signal_text[8192] = "1.0\n0.707\n0.0\n-0.707\n-1.0\n-0.707\n0.0\n0.707";
    ImGui::InputTextMultiline("##signaltext", signal_text, sizeof(signal_text),
                               ImVec2(-1, 200));

    if (ImGui::Button("Parse Signal", ImVec2(-1, 0))) {
        signal_.clear();
        time_axis_.clear();

        std::string text(signal_text);
        std::istringstream iss(text);
        std::string line;
        double t = 0;
        double dt = 1.0 / sample_rate_;

        while (std::getline(iss, line)) {
            if (line.empty()) continue;
            try {
                double val = std::stod(line);
                signal_.push_back(val);
                time_axis_.push_back(t);
                t += dt;
            } catch (...) {
                // Skip invalid lines
            }
        }

        has_result_ = false;
        spdlog::info("Parsed {} signal samples", signal_.size());
    }

    ImGui::Text("Samples loaded: %d", static_cast<int>(signal_.size()));
}

void FFTPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Computing FFT...", ICON_FA_SPINNER);
}

void FFTPanel::RenderResults() {
    ImGui::Text(ICON_FA_CHART_LINE " Results");
    ImGui::Separator();

    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (signal_.empty()) {
        ImGui::TextDisabled("No signal data. Generate or input a signal first.");
        return;
    }

    if (ImGui::BeginTabBar("FFTTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CLOCK " Time Domain")) {
            RenderTimeDomain();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CHART_AREA " Frequency")) {
            RenderFrequencyDomain();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_ROTATE " Phase")) {
            RenderPhasePlot();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_MOUNTAIN " Peaks")) {
            RenderPeaks();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void FFTPanel::RenderTimeDomain() {
    ImGui::Text("Time Domain Signal:");

    if (ImPlot::BeginPlot("##TimeDomain", ImVec2(-1, 300))) {
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
        double min_val = signal_[0], max_val = signal_[0];
        for (double v : signal_) {
            sum += v;
            sum_sq += v * v;
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
        }
        double mean = sum / signal_.size();
        double variance = sum_sq / signal_.size() - mean * mean;
        double rms = std::sqrt(sum_sq / signal_.size());

        ImGui::Text("Statistics:");
        ImGui::BulletText("Min: %.4f", min_val);
        ImGui::SameLine(150);
        ImGui::BulletText("Max: %.4f", max_val);
        ImGui::BulletText("Mean: %.4f", mean);
        ImGui::SameLine(150);
        ImGui::BulletText("RMS: %.4f", rms);
        ImGui::BulletText("Variance: %.6f", variance);
    }
}

void FFTPanel::RenderFrequencyDomain() {
    if (!has_result_) {
        ImGui::TextDisabled("Click 'Compute FFT' to see frequency spectrum");
        return;
    }

    ImGui::Text("Frequency Spectrum (Magnitude):");

    // Only plot positive frequencies (first half)
    int half_n = static_cast<int>(result_.magnitude.size()) / 2;
    std::vector<double> pos_freq(result_.frequencies.begin(), result_.frequencies.begin() + half_n);
    std::vector<double> pos_mag(result_.magnitude.begin(), result_.magnitude.begin() + half_n);

    // Option for log scale
    static bool log_scale = false;
    ImGui::Checkbox("Log Scale (dB)", &log_scale);

    if (log_scale) {
        std::vector<double> mag_db(pos_mag.size());
        for (size_t i = 0; i < pos_mag.size(); i++) {
            mag_db[i] = 20.0 * std::log10(std::max(pos_mag[i], 1e-10));
        }

        if (ImPlot::BeginPlot("##FreqDomain", ImVec2(-1, 300))) {
            ImPlot::SetupAxes("Frequency (Hz)", "Magnitude (dB)");
            ImPlot::PlotLine("Magnitude", pos_freq.data(), mag_db.data(),
                             static_cast<int>(pos_freq.size()));
            ImPlot::EndPlot();
        }
    } else {
        if (ImPlot::BeginPlot("##FreqDomain", ImVec2(-1, 300))) {
            ImPlot::SetupAxes("Frequency (Hz)", "Magnitude");
            ImPlot::PlotLine("Magnitude", pos_freq.data(), pos_mag.data(),
                             static_cast<int>(pos_freq.size()));
            ImPlot::EndPlot();
        }
    }

    // Nyquist info
    ImGui::Text("Nyquist Frequency: %.1f Hz", result_.sample_rate / 2.0);
    ImGui::Text("Frequency Resolution: %.2f Hz", result_.sample_rate / result_.n);
}

void FFTPanel::RenderPhasePlot() {
    if (!has_result_) {
        ImGui::TextDisabled("Click 'Compute FFT' to see phase spectrum");
        return;
    }

    ImGui::Text("Phase Spectrum:");

    int half_n = static_cast<int>(result_.phase.size()) / 2;
    std::vector<double> pos_freq(result_.frequencies.begin(), result_.frequencies.begin() + half_n);
    std::vector<double> pos_phase(result_.phase.begin(), result_.phase.begin() + half_n);

    // Convert to degrees option
    static bool use_degrees = true;
    ImGui::Checkbox("Degrees", &use_degrees);

    if (use_degrees) {
        std::vector<double> phase_deg(pos_phase.size());
        for (size_t i = 0; i < pos_phase.size(); i++) {
            phase_deg[i] = pos_phase[i] * 180.0 / 3.14159265358979323846;
        }

        if (ImPlot::BeginPlot("##PhaseDomain", ImVec2(-1, 300))) {
            ImPlot::SetupAxes("Frequency (Hz)", "Phase (degrees)");
            ImPlot::PlotLine("Phase", pos_freq.data(), phase_deg.data(),
                             static_cast<int>(pos_freq.size()));
            ImPlot::EndPlot();
        }
    } else {
        if (ImPlot::BeginPlot("##PhaseDomain", ImVec2(-1, 300))) {
            ImPlot::SetupAxes("Frequency (Hz)", "Phase (radians)");
            ImPlot::PlotLine("Phase", pos_freq.data(), pos_phase.data(),
                             static_cast<int>(pos_freq.size()));
            ImPlot::EndPlot();
        }
    }
}

void FFTPanel::RenderPeaks() {
    if (!has_result_) {
        ImGui::TextDisabled("Click 'Compute FFT' to detect peaks");
        return;
    }

    ImGui::Text("Peak Detection:");
    ImGui::Spacing();

    ImGui::Text("Number of Peaks:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::InputInt("##numpeaks", &num_peaks_to_find_);
    num_peaks_to_find_ = std::clamp(num_peaks_to_find_, 1, 20);

    ImGui::SameLine();

    ImGui::Text("Threshold:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    static const double kPeakThreshMin = 0.01;
    static const double kPeakThreshMax = 0.5;
    ImGui::SliderScalar("##peakthresh", ImGuiDataType_Double, &peak_threshold_, &kPeakThreshMin, &kPeakThreshMax, "%.2f");

    ImGui::SameLine();

    if (ImGui::Button("Find Peaks")) {
        FindPeaks();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (peaks_.empty()) {
        ImGui::TextDisabled("No peaks found. Click 'Find Peaks' or adjust threshold.");
        return;
    }

    // Show peaks in table
    if (ImGui::BeginTable("PeaksTable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Rank", ImGuiTableColumnFlags_WidthFixed, 50);
        ImGui::TableSetupColumn("Frequency (Hz)", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Magnitude", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < peaks_.size(); i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%zu", i + 1);
            ImGui::TableNextColumn();
            ImGui::Text("%.2f", peaks_[i].frequency);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", peaks_[i].magnitude);
        }
        ImGui::EndTable();
    }

    // Visualize peaks on spectrum
    ImGui::Spacing();
    ImGui::Text("Spectrum with Peaks:");

    int half_n = static_cast<int>(result_.magnitude.size()) / 2;
    std::vector<double> pos_freq(result_.frequencies.begin(), result_.frequencies.begin() + half_n);
    std::vector<double> pos_mag(result_.magnitude.begin(), result_.magnitude.begin() + half_n);

    if (ImPlot::BeginPlot("##SpectrumPeaks", ImVec2(-1, 250))) {
        ImPlot::SetupAxes("Frequency (Hz)", "Magnitude");
        ImPlot::PlotLine("Spectrum", pos_freq.data(), pos_mag.data(),
                         static_cast<int>(pos_freq.size()));

        // Plot peak markers
        std::vector<double> peak_freqs, peak_mags;
        for (size_t i = 0; i < peaks_.size(); i++) {
            peak_freqs.push_back(peaks_[i].frequency);
            peak_mags.push_back(peaks_[i].magnitude);
        }

        if (!peak_freqs.empty()) {
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 8, ImVec4(1, 0, 0, 1), 2);
            ImPlot::PlotScatter("Peaks", peak_freqs.data(), peak_mags.data(),
                                static_cast<int>(peak_freqs.size()));
        }

        ImPlot::EndPlot();
    }
}

void FFTPanel::GenerateSignal() {
    signal_.clear();
    time_axis_.clear();

    double dt = 1.0 / sample_rate_;

    for (int i = 0; i < num_samples_; i++) {
        double t = i * dt;
        time_axis_.push_back(t);

        double value = 0.0;

        switch (signal_type_) {
            case SignalType::Sine:
                value = amplitude1_ * std::sin(2.0 * 3.14159265358979323846 * frequency1_ * t);
                break;

            case SignalType::Square: {
                double phase = std::fmod(frequency1_ * t, 1.0);
                value = amplitude1_ * (phase < 0.5 ? 1.0 : -1.0);
                break;
            }

            case SignalType::Sawtooth: {
                double phase = std::fmod(frequency1_ * t, 1.0);
                value = amplitude1_ * (2.0 * phase - 1.0);
                break;
            }

            case SignalType::WhiteNoise: {
                value = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                break;
            }

            case SignalType::Composite:
                value = amplitude1_ * std::sin(2.0 * 3.14159265358979323846 * frequency1_ * t) +
                        amplitude2_ * std::sin(2.0 * 3.14159265358979323846 * frequency2_ * t);
                break;
        }

        signal_.push_back(value);
    }

    // Add noise if requested
    if (add_noise_ && signal_type_ != SignalType::WhiteNoise) {
        signal_ = SignalProcessing::AddNoise(signal_, noise_snr_);
    }

    spdlog::info("Generated {} samples at {} Hz", num_samples_, sample_rate_);
}

void FFTPanel::ComputeAsync() {
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
    peaks_.clear();
    error_message_.clear();

    compute_thread_ = std::make_unique<std::thread>([this]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            result_ = SignalProcessing::FFT(signal_, sample_rate_);
            if (result_.success) {
                has_result_ = true;
                spdlog::info("FFT computed: {} frequency bins", result_.n);
            } else {
                error_message_ = result_.error_message;
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void FFTPanel::FindPeaks() {
    peaks_.clear();

    if (!has_result_ || result_.magnitude.empty()) return;

    int half_n = static_cast<int>(result_.magnitude.size()) / 2;

    // Find max magnitude for threshold
    double max_mag = 0;
    for (int i = 1; i < half_n; i++) {  // Skip DC component
        max_mag = std::max(max_mag, result_.magnitude[i]);
    }

    double threshold = max_mag * peak_threshold_;

    // Find local maxima above threshold
    std::vector<Peak> candidates;
    for (int i = 2; i < half_n - 1; i++) {
        double m = result_.magnitude[i];
        if (m > threshold &&
            m > result_.magnitude[i - 1] &&
            m > result_.magnitude[i + 1]) {
            candidates.push_back({result_.frequencies[i], m, i});
        }
    }

    // Sort by magnitude descending
    std::sort(candidates.begin(), candidates.end(),
              [](const Peak& a, const Peak& b) { return a.magnitude > b.magnitude; });

    // Take top N peaks
    int n = std::min(num_peaks_to_find_, static_cast<int>(candidates.size()));
    peaks_.assign(candidates.begin(), candidates.begin() + n);

    // Sort by frequency for display
    std::sort(peaks_.begin(), peaks_.end(),
              [](const Peak& a, const Peak& b) { return a.frequency < b.frequency; });

    spdlog::info("Found {} peaks", peaks_.size());
}

} // namespace cyxwiz
