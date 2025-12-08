#include "filter_designer_panel.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace cyxwiz {

FilterDesignerPanel::FilterDesignerPanel() {
    GenerateTestSignal();
}

FilterDesignerPanel::~FilterDesignerPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        compute_thread_->join();
    }
}

void FilterDesignerPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_FILTER " Filter Designer###FilterDesignerPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            float panel_width = ImGui::GetContentRegionAvail().x;

            ImGui::BeginChild("SettingsPanel", ImVec2(panel_width * 0.30f, 0), true);
            RenderFilterSettings();
            ImGui::EndChild();

            ImGui::SameLine();

            ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
            RenderResults();
            ImGui::EndChild();
        }
    }
    ImGui::End();
}

void FilterDesignerPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Design Filter")) {
        DesignFilterAsync();
    }

    ImGui::SameLine();

    if (has_filter_ && ImGui::Button(ICON_FA_WAND_MAGIC_SPARKLES " Apply to Signal")) {
        ApplyFilter();
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        has_filter_ = false;
        has_filtered_ = false;
        error_message_.clear();
    }
}

void FilterDesignerPanel::RenderFilterSettings() {
    ImGui::Text(ICON_FA_SLIDERS " Filter Settings");
    ImGui::Separator();
    ImGui::Spacing();

    // Filter type
    ImGui::Text("Filter Type:");
    const char* filter_types[] = { "Lowpass", "Highpass", "Bandpass", "Bandstop" };
    int current_type = static_cast<int>(filter_type_);
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##FilterType", &current_type, filter_types, IM_ARRAYSIZE(filter_types))) {
        filter_type_ = static_cast<FilterType>(current_type);
        has_filter_ = false;
    }

    ImGui::Spacing();

    // Sample rate
    ImGui::Text("Sample Rate (Hz):");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputDouble("##samplerate", &sample_rate_, 100.0, 500.0, "%.0f")) {
        sample_rate_ = std::clamp(sample_rate_, 100.0, 48000.0);
        has_filter_ = false;
    }

    ImGui::Spacing();

    // Cutoff frequencies
    double nyquist = sample_rate_ / 2.0;

    ImGui::Text("Cutoff Frequency 1 (Hz):");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputDouble("##cutoff1", &cutoff1_, 10.0, 50.0, "%.1f")) {
        cutoff1_ = std::clamp(cutoff1_, 1.0, nyquist - 1.0);
        has_filter_ = false;
    }

    if (filter_type_ == FilterType::Bandpass || filter_type_ == FilterType::Bandstop) {
        ImGui::Text("Cutoff Frequency 2 (Hz):");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputDouble("##cutoff2", &cutoff2_, 10.0, 50.0, "%.1f")) {
            cutoff2_ = std::clamp(cutoff2_, cutoff1_ + 1.0, nyquist - 1.0);
            has_filter_ = false;
        }
    }

    ImGui::Spacing();

    // Filter order
    ImGui::Text("Filter Order (taps):");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::SliderInt("##order", &filter_order_, 11, 201)) {
        // Keep odd for symmetric FIR
        if (filter_order_ % 2 == 0) filter_order_++;
        has_filter_ = false;
    }
    ImGui::TextDisabled("(Higher = sharper cutoff, more delay)");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Nyquist info
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Nyquist: %.1f Hz", nyquist);

    // Normalized cutoff
    if (filter_type_ == FilterType::Lowpass || filter_type_ == FilterType::Highpass) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Norm. cutoff: %.3f", cutoff1_ / nyquist);
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Norm. cutoffs: %.3f - %.3f",
                          cutoff1_ / nyquist, cutoff2_ / nyquist);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Test signal settings
    ImGui::Text(ICON_FA_WAVE_SQUARE " Test Signal");
    ImGui::Spacing();

    ImGui::Text("Frequency 1 (Hz):");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputDouble("##testfreq1", &test_freq1_, 10.0, 50.0, "%.1f");
    test_freq1_ = std::clamp(test_freq1_, 1.0, sample_rate_ / 2.0);

    ImGui::Text("Frequency 2 (Hz):");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputDouble("##testfreq2", &test_freq2_, 10.0, 50.0, "%.1f");
    test_freq2_ = std::clamp(test_freq2_, 1.0, sample_rate_ / 2.0);

    ImGui::Spacing();

    if (ImGui::Button(ICON_FA_ROTATE " Regenerate Signal", ImVec2(-1, 0))) {
        GenerateTestSignal();
        has_filtered_ = false;
    }

    // Presets
    ImGui::Spacing();
    ImGui::Text("Filter Presets:");

    if (ImGui::Button("LP 200Hz", ImVec2(-1, 0))) {
        filter_type_ = FilterType::Lowpass;
        cutoff1_ = 200.0;
        filter_order_ = 51;
        sample_rate_ = 2000.0;
        has_filter_ = false;
    }
    if (ImGui::Button("HP 300Hz", ImVec2(-1, 0))) {
        filter_type_ = FilterType::Highpass;
        cutoff1_ = 300.0;
        filter_order_ = 51;
        sample_rate_ = 2000.0;
        has_filter_ = false;
    }
    if (ImGui::Button("BP 200-400Hz", ImVec2(-1, 0))) {
        filter_type_ = FilterType::Bandpass;
        cutoff1_ = 200.0;
        cutoff2_ = 400.0;
        filter_order_ = 101;
        sample_rate_ = 2000.0;
        has_filter_ = false;
    }
}

void FilterDesignerPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Designing filter...", ICON_FA_SPINNER);
}

void FilterDesignerPanel::RenderResults() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_filter_) {
        ImGui::TextDisabled("Click 'Design Filter' to create filter coefficients");
        return;
    }

    if (ImGui::BeginTabBar("FilterTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Frequency Response")) {
            RenderFrequencyResponse();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_TABLE " Coefficients")) {
            RenderCoefficients();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_WAVE_SQUARE " Signal Filtering")) {
            RenderSignalFiltering();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void FilterDesignerPanel::RenderFrequencyResponse() {
    ImGui::Text("Frequency Response:");

    // Magnitude plot
    static bool log_scale = true;
    ImGui::Checkbox("dB Scale", &log_scale);

    if (ImPlot::BeginPlot("##MagnitudeResponse", ImVec2(-1, 250))) {
        ImPlot::SetupAxes("Frequency (Hz)", log_scale ? "Magnitude (dB)" : "Magnitude");

        if (!filter_.freq_axis.empty() && !filter_.freq_response_mag.empty()) {
            if (log_scale) {
                std::vector<double> mag_db(filter_.freq_response_mag.size());
                for (size_t i = 0; i < filter_.freq_response_mag.size(); i++) {
                    mag_db[i] = 20.0 * std::log10(std::max(filter_.freq_response_mag[i], 1e-10));
                }
                ImPlot::PlotLine("Magnitude", filter_.freq_axis.data(), mag_db.data(),
                                 static_cast<int>(filter_.freq_axis.size()));

                // Draw -3dB line
                double x_range[2] = { filter_.freq_axis.front(), filter_.freq_axis.back() };
                double y_3db[2] = { -3.0, -3.0 };
                ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1, 0.5, 0, 0.7));
                ImPlot::PlotLine("-3dB", x_range, y_3db, 2);
                ImPlot::PopStyleColor();
            } else {
                ImPlot::PlotLine("Magnitude", filter_.freq_axis.data(), filter_.freq_response_mag.data(),
                                 static_cast<int>(filter_.freq_axis.size()));
            }

            // Draw cutoff frequency lines
            double y_min = log_scale ? -60.0 : 0.0;
            double y_max = log_scale ? 5.0 : 1.2;

            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1, 0, 0, 0.5));
            double cutoff_line_x[2] = { cutoff1_, cutoff1_ };
            double cutoff_line_y[2] = { y_min, y_max };
            ImPlot::PlotLine("##cutoff1", cutoff_line_x, cutoff_line_y, 2);

            if (filter_type_ == FilterType::Bandpass || filter_type_ == FilterType::Bandstop) {
                cutoff_line_x[0] = cutoff_line_x[1] = cutoff2_;
                ImPlot::PlotLine("##cutoff2", cutoff_line_x, cutoff_line_y, 2);
            }
            ImPlot::PopStyleColor();
        }
        ImPlot::EndPlot();
    }

    // Phase plot
    ImGui::Spacing();
    ImGui::Text("Phase Response:");

    if (ImPlot::BeginPlot("##PhaseResponse", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("Frequency (Hz)", "Phase (degrees)");

        if (!filter_.freq_axis.empty() && !filter_.freq_response_phase.empty()) {
            std::vector<double> phase_deg(filter_.freq_response_phase.size());
            for (size_t i = 0; i < filter_.freq_response_phase.size(); i++) {
                phase_deg[i] = filter_.freq_response_phase[i] * 180.0 / 3.14159265358979323846;
            }
            ImPlot::PlotLine("Phase", filter_.freq_axis.data(), phase_deg.data(),
                             static_cast<int>(filter_.freq_axis.size()));
        }
        ImPlot::EndPlot();
    }
}

void FilterDesignerPanel::RenderCoefficients() {
    ImGui::Text("Filter Coefficients (b):");
    ImGui::Text("Order: %d taps", filter_.order);
    ImGui::Spacing();

    // Copy button
    if (ImGui::Button(ICON_FA_COPY " Copy to Clipboard")) {
        std::ostringstream oss;
        oss << "b = [";
        for (size_t i = 0; i < filter_.b.size(); i++) {
            if (i > 0) oss << ", ";
            oss << filter_.b[i];
        }
        oss << "]";
        ImGui::SetClipboardText(oss.str().c_str());
    }

    ImGui::Spacing();

    // Table of coefficients
    if (ImGui::BeginTable("CoeffTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_ScrollY, ImVec2(0, 300))) {
        ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_WidthFixed, 60);
        ImGui::TableSetupColumn("Coefficient", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < filter_.b.size(); i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%zu", i);
            ImGui::TableNextColumn();
            ImGui::Text("%.10f", filter_.b[i]);
        }
        ImGui::EndTable();
    }

    // Impulse response plot
    ImGui::Spacing();
    ImGui::Text("Impulse Response (Filter Taps):");

    if (ImPlot::BeginPlot("##ImpulseResponse", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("Sample", "Amplitude");

        std::vector<double> indices(filter_.b.size());
        for (size_t i = 0; i < indices.size(); i++) {
            indices[i] = static_cast<double>(i);
        }

        ImPlot::PlotStems("h[n]", indices.data(), filter_.b.data(),
                          static_cast<int>(filter_.b.size()));
        ImPlot::EndPlot();
    }
}

void FilterDesignerPanel::RenderSignalFiltering() {
    ImGui::Text("Filter Test Signal:");
    ImGui::Text("Input: %.1f Hz + %.1f Hz sinusoids", test_freq1_, test_freq2_);

    // Before/After comparison
    if (ImPlot::BeginPlot("##FilterComparison", ImVec2(-1, 250))) {
        ImPlot::SetupAxes("Time (s)", "Amplitude");
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);

        if (!test_signal_.empty() && !time_axis_.empty()) {
            ImPlot::PlotLine("Original", time_axis_.data(), test_signal_.data(),
                             static_cast<int>(test_signal_.size()));
        }

        if (has_filtered_ && !filtered_signal_.empty()) {
            ImPlot::PlotLine("Filtered", time_axis_.data(), filtered_signal_.data(),
                             static_cast<int>(filtered_signal_.size()));
        }

        ImPlot::EndPlot();
    }

    if (!has_filtered_) {
        ImGui::TextDisabled("Click 'Apply to Signal' to see filtering result");
    } else {
        // Show RMS comparison
        double orig_rms = 0, filt_rms = 0;
        for (double v : test_signal_) orig_rms += v * v;
        for (double v : filtered_signal_) filt_rms += v * v;
        orig_rms = std::sqrt(orig_rms / test_signal_.size());
        filt_rms = std::sqrt(filt_rms / filtered_signal_.size());

        ImGui::Text("Original RMS: %.4f | Filtered RMS: %.4f", orig_rms, filt_rms);
        ImGui::Text("Attenuation: %.2f dB", 20.0 * std::log10(filt_rms / orig_rms));
    }

    // Frequency content comparison
    ImGui::Spacing();
    ImGui::Text("Expected filtering behavior:");

    const char* type_name = "";
    switch (filter_type_) {
        case FilterType::Lowpass: type_name = "Lowpass"; break;
        case FilterType::Highpass: type_name = "Highpass"; break;
        case FilterType::Bandpass: type_name = "Bandpass"; break;
        case FilterType::Bandstop: type_name = "Bandstop"; break;
    }

    if (filter_type_ == FilterType::Lowpass) {
        ImGui::BulletText("%.1f Hz: %s", test_freq1_,
                          test_freq1_ < cutoff1_ ? "PASS" : "ATTENUATE");
        ImGui::BulletText("%.1f Hz: %s", test_freq2_,
                          test_freq2_ < cutoff1_ ? "PASS" : "ATTENUATE");
    } else if (filter_type_ == FilterType::Highpass) {
        ImGui::BulletText("%.1f Hz: %s", test_freq1_,
                          test_freq1_ > cutoff1_ ? "PASS" : "ATTENUATE");
        ImGui::BulletText("%.1f Hz: %s", test_freq2_,
                          test_freq2_ > cutoff1_ ? "PASS" : "ATTENUATE");
    } else if (filter_type_ == FilterType::Bandpass) {
        ImGui::BulletText("%.1f Hz: %s", test_freq1_,
                          (test_freq1_ > cutoff1_ && test_freq1_ < cutoff2_) ? "PASS" : "ATTENUATE");
        ImGui::BulletText("%.1f Hz: %s", test_freq2_,
                          (test_freq2_ > cutoff1_ && test_freq2_ < cutoff2_) ? "PASS" : "ATTENUATE");
    } else {
        ImGui::BulletText("%.1f Hz: %s", test_freq1_,
                          (test_freq1_ < cutoff1_ || test_freq1_ > cutoff2_) ? "PASS" : "ATTENUATE");
        ImGui::BulletText("%.1f Hz: %s", test_freq2_,
                          (test_freq2_ < cutoff1_ || test_freq2_ > cutoff2_) ? "PASS" : "ATTENUATE");
    }
}

void FilterDesignerPanel::DesignFilterAsync() {
    if (is_computing_.load()) return;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_filter_ = false;
    has_filtered_ = false;
    error_message_.clear();

    // Capture parameters
    FilterType type = filter_type_;
    double c1 = cutoff1_;
    double c2 = cutoff2_;
    int order = filter_order_;
    double sr = sample_rate_;

    compute_thread_ = std::make_unique<std::thread>([this, type, c1, c2, order, sr]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            switch (type) {
                case FilterType::Lowpass:
                    filter_ = SignalProcessing::DesignLowpass(c1, sr, order);
                    break;
                case FilterType::Highpass:
                    filter_ = SignalProcessing::DesignHighpass(c1, sr, order);
                    break;
                case FilterType::Bandpass:
                    filter_ = SignalProcessing::DesignBandpass(c1, c2, sr, order);
                    break;
                case FilterType::Bandstop:
                    filter_ = SignalProcessing::DesignBandstop(c1, c2, sr, order);
                    break;
            }

            if (filter_.success) {
                has_filter_ = true;
                spdlog::info("Filter designed: {} taps", filter_.order);
            } else {
                error_message_ = filter_.error_message;
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void FilterDesignerPanel::GenerateTestSignal() {
    test_signal_.clear();
    time_axis_.clear();

    const double PI = 3.14159265358979323846;
    double dt = 1.0 / sample_rate_;

    for (int i = 0; i < test_samples_; i++) {
        double t = i * dt;
        time_axis_.push_back(t);

        double value = 0.5 * std::sin(2.0 * PI * test_freq1_ * t) +
                       0.5 * std::sin(2.0 * PI * test_freq2_ * t);
        test_signal_.push_back(value);
    }

    has_filtered_ = false;
}

void FilterDesignerPanel::ApplyFilter() {
    if (!has_filter_ || test_signal_.empty()) return;

    filtered_signal_ = SignalProcessing::ApplyFilter(test_signal_, filter_);
    has_filtered_ = true;
    spdlog::info("Filter applied to signal");
}

} // namespace cyxwiz
