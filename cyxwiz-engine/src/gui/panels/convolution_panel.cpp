#include "convolution_panel.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace cyxwiz {

ConvolutionPanel::ConvolutionPanel() {
    GenerateSignal();
    ApplyKernelPreset();
}

ConvolutionPanel::~ConvolutionPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        compute_thread_->join();
    }
}

void ConvolutionPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 650), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_ASTERISK " Convolution Calculator###ConvolutionPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            float panel_width = ImGui::GetContentRegionAvail().x;

            ImGui::BeginChild("InputPanel", ImVec2(panel_width * 0.35f, 0), true);
            RenderInputs();
            ImGui::EndChild();

            ImGui::SameLine();

            ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
            RenderResults();
            ImGui::EndChild();
        }
    }
    ImGui::End();
}

void ConvolutionPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Convolve")) {
        ComputeAsync();
    }

    ImGui::SameLine();

    ImGui::Text("Mode:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    const char* modes[] = { "Full", "Same", "Valid" };
    int current_mode = static_cast<int>(conv_mode_);
    if (ImGui::Combo("##Mode", &current_mode, modes, IM_ARRAYSIZE(modes))) {
        conv_mode_ = static_cast<ConvMode>(current_mode);
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

    ImGui::TextDisabled("Signal[%d] * Kernel[%d]", static_cast<int>(signal_.size()),
                        static_cast<int>(kernel_.size()));
}

void ConvolutionPanel::RenderInputs() {
    if (ImGui::CollapsingHeader(ICON_FA_SIGNAL " Signal", ImGuiTreeNodeFlags_DefaultOpen)) {
        RenderSignalInput();
    }

    ImGui::Spacing();

    if (ImGui::CollapsingHeader(ICON_FA_BORDER_ALL " Kernel", ImGuiTreeNodeFlags_DefaultOpen)) {
        RenderKernelInput();
    }
}

void ConvolutionPanel::RenderSignalInput() {
    // Signal type selector
    ImGui::Text("Type:");
    const char* signal_types[] = { "Step", "Impulse", "Ramp", "Sine", "Custom" };
    int current_type = static_cast<int>(signal_type_);
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##SignalType", &current_type, signal_types, IM_ARRAYSIZE(signal_types))) {
        signal_type_ = static_cast<SignalType>(current_type);
        if (signal_type_ != SignalType::Custom) {
            GenerateSignal();
        }
    }

    if (signal_type_ != SignalType::Custom) {
        ImGui::Text("Size:");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::SliderInt("##SignalSize", &signal_size_, 8, 64)) {
            GenerateSignal();
        }

        if (ImGui::Button("Regenerate", ImVec2(-1, 0))) {
            GenerateSignal();
            has_result_ = false;
        }
    }

    ImGui::Spacing();
    ImGui::Text("Values (%d samples):", static_cast<int>(signal_.size()));

    // Editable table of values
    if (ImGui::BeginTable("SignalTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY,
                          ImVec2(0, 120))) {
        for (size_t i = 0; i < signal_.size(); i++) {
            if (i % 4 == 0) ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::PushID(static_cast<int>(i));
            ImGui::SetNextItemWidth(-1);
            if (ImGui::InputDouble("##s", &signal_[i], 0, 0, "%.2f")) {
                signal_type_ = SignalType::Custom;
                has_result_ = false;
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
}

void ConvolutionPanel::RenderKernelInput() {
    // Kernel preset selector
    ImGui::Text("Preset:");
    const char* presets[] = { "Custom", "Moving Avg", "Gaussian", "Derivative", "Laplacian", "Edge" };
    int current_preset = static_cast<int>(kernel_preset_);
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##KernelPreset", &current_preset, presets, IM_ARRAYSIZE(presets))) {
        kernel_preset_ = static_cast<KernelPreset>(current_preset);
        if (kernel_preset_ != KernelPreset::Custom) {
            ApplyKernelPreset();
        }
    }

    if (kernel_preset_ == KernelPreset::MovingAverage || kernel_preset_ == KernelPreset::Gaussian) {
        ImGui::Text("Size:");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::SliderInt("##KernelSize", &kernel_size_, 3, 15)) {
            if (kernel_size_ % 2 == 0) kernel_size_++;  // Keep odd
            ApplyKernelPreset();
        }
    }

    ImGui::Spacing();
    ImGui::Text("Values (%d taps):", static_cast<int>(kernel_.size()));

    // Editable table of kernel values
    if (ImGui::BeginTable("KernelTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY,
                          ImVec2(0, 100))) {
        for (size_t i = 0; i < kernel_.size(); i++) {
            if (i % 4 == 0) ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::PushID(1000 + static_cast<int>(i));
            ImGui::SetNextItemWidth(-1);
            if (ImGui::InputDouble("##k", &kernel_[i], 0, 0, "%.3f")) {
                kernel_preset_ = KernelPreset::Custom;
                has_result_ = false;
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    // Kernel sum info
    double kernel_sum = std::accumulate(kernel_.begin(), kernel_.end(), 0.0);
    ImGui::Text("Sum: %.4f", kernel_sum);
    if (std::abs(kernel_sum - 1.0) > 0.01 && std::abs(kernel_sum) > 0.01) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1, 0.5, 0, 1), "(not normalized)");
    }
}

void ConvolutionPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Computing convolution...", ICON_FA_SPINNER);
}

void ConvolutionPanel::RenderResults() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (signal_.empty() || kernel_.empty()) {
        ImGui::TextDisabled("Configure signal and kernel first");
        return;
    }

    if (ImGui::BeginTabBar("ConvTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Plots")) {
            RenderResultPlot();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CALCULATOR " Math")) {
            RenderMathView();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void ConvolutionPanel::RenderResultPlot() {
    // Signal plot
    ImGui::Text("Signal x[n]:");
    if (ImPlot::BeginPlot("##Signal", ImVec2(-1, 150))) {
        ImPlot::SetupAxes("n", "x[n]");

        std::vector<double> indices(signal_.size());
        std::iota(indices.begin(), indices.end(), 0);

        ImPlot::PlotStems("x[n]", indices.data(), signal_.data(),
                          static_cast<int>(signal_.size()), 0);
        ImPlot::EndPlot();
    }

    // Kernel plot
    ImGui::Text("Kernel h[n]:");
    if (ImPlot::BeginPlot("##Kernel", ImVec2(-1, 100))) {
        ImPlot::SetupAxes("n", "h[n]");

        std::vector<double> indices(kernel_.size());
        std::iota(indices.begin(), indices.end(), 0);

        ImPlot::PlotStems("h[n]", indices.data(), kernel_.data(),
                          static_cast<int>(kernel_.size()), 0);
        ImPlot::EndPlot();
    }

    // Result plot
    if (!has_result_) {
        ImGui::TextDisabled("Click 'Convolve' to compute x[n] * h[n]");
        return;
    }

    ImGui::Text("Output y[n] = x[n] * h[n]:");
    if (ImPlot::BeginPlot("##Output", ImVec2(-1, 150))) {
        ImPlot::SetupAxes("n", "y[n]");

        std::vector<double> indices(result_.output.size());
        std::iota(indices.begin(), indices.end(), 0);

        ImPlot::PlotStems("y[n]", indices.data(), result_.output.data(),
                          static_cast<int>(result_.output.size()), 0);
        ImPlot::EndPlot();
    }

    // Size info
    ImGui::Text("Output size: %d", result_.output_size);
    const char* mode_names[] = { "full", "same", "valid" };
    ImGui::Text("Mode: %s", mode_names[static_cast<int>(conv_mode_)]);

    // Size formula
    int N = static_cast<int>(signal_.size());
    int M = static_cast<int>(kernel_.size());
    ImGui::TextDisabled("N=%d, M=%d", N, M);
    switch (conv_mode_) {
        case ConvMode::Full:
            ImGui::TextDisabled("Full: N + M - 1 = %d", N + M - 1);
            break;
        case ConvMode::Same:
            ImGui::TextDisabled("Same: max(N, M) = %d", std::max(N, M));
            break;
        case ConvMode::Valid:
            ImGui::TextDisabled("Valid: max(N, M) - min(N, M) + 1 = %d",
                               std::max(N, M) - std::min(N, M) + 1);
            break;
    }
}

void ConvolutionPanel::RenderMathView() {
    ImGui::Text("Convolution Definition:");
    ImGui::TextWrapped("y[n] = (x * h)[n] = sum_{k=-inf}^{inf} x[k] * h[n-k]");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Discrete Convolution Formula:");
    ImGui::TextWrapped("For finite sequences of length N and M:");
    ImGui::TextWrapped("y[n] = sum_{k=0}^{M-1} h[k] * x[n-k]");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Output Lengths by Mode:");
    ImGui::BulletText("Full: N + M - 1 (complete overlap)");
    ImGui::BulletText("Same: max(N, M) (centered output)");
    ImGui::BulletText("Valid: |N - M| + 1 (no zero-padding)");

    if (has_result_) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::Text("Result Values:");

        if (ImGui::BeginTable("ResultTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_ScrollY, ImVec2(0, 200))) {
            ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_WidthFixed, 60);
            ImGui::TableSetupColumn("y[n]", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableHeadersRow();

            for (size_t i = 0; i < result_.output.size(); i++) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%zu", i);
                ImGui::TableNextColumn();
                ImGui::Text("%.6f", result_.output[i]);
            }
            ImGui::EndTable();
        }
    }
}

void ConvolutionPanel::GenerateSignal() {
    signal_.resize(signal_size_);

    switch (signal_type_) {
        case SignalType::Step:
            // Unit step: 0 for first half, 1 for second half
            for (int i = 0; i < signal_size_; i++) {
                signal_[i] = (i >= signal_size_ / 2) ? 1.0 : 0.0;
            }
            break;

        case SignalType::Impulse:
            // Delta function at center
            std::fill(signal_.begin(), signal_.end(), 0.0);
            signal_[signal_size_ / 2] = 1.0;
            break;

        case SignalType::Ramp:
            // Linear ramp
            for (int i = 0; i < signal_size_; i++) {
                signal_[i] = static_cast<double>(i) / (signal_size_ - 1);
            }
            break;

        case SignalType::Sine: {
            // One period of sine
            const double PI = 3.14159265358979323846;
            for (int i = 0; i < signal_size_; i++) {
                signal_[i] = std::sin(2.0 * PI * i / signal_size_);
            }
            break;
        }

        case SignalType::Custom:
            // Keep existing values
            break;
    }

    has_result_ = false;
}

void ConvolutionPanel::ApplyKernelPreset() {
    switch (kernel_preset_) {
        case KernelPreset::Custom:
            // Keep existing
            break;

        case KernelPreset::MovingAverage:
            // Simple box filter
            kernel_.resize(kernel_size_);
            for (int i = 0; i < kernel_size_; i++) {
                kernel_[i] = 1.0 / kernel_size_;
            }
            break;

        case KernelPreset::Gaussian: {
            // Gaussian approximation
            kernel_.resize(kernel_size_);
            double sigma = kernel_size_ / 6.0;
            int center = kernel_size_ / 2;
            double sum = 0;
            for (int i = 0; i < kernel_size_; i++) {
                double x = i - center;
                kernel_[i] = std::exp(-0.5 * x * x / (sigma * sigma));
                sum += kernel_[i];
            }
            // Normalize
            for (auto& k : kernel_) k /= sum;
            break;
        }

        case KernelPreset::Derivative:
            // First derivative (central difference)
            kernel_ = { -1.0, 0.0, 1.0 };
            break;

        case KernelPreset::Laplacian:
            // Second derivative
            kernel_ = { 1.0, -2.0, 1.0 };
            break;

        case KernelPreset::EdgeDetect:
            // Sharpening kernel (1D approximation)
            kernel_ = { -1.0, 2.0, -1.0 };
            break;
    }

    kernel_size_ = static_cast<int>(kernel_.size());
    has_result_ = false;
}

void ConvolutionPanel::ComputeAsync() {
    if (is_computing_.load()) return;
    if (signal_.empty() || kernel_.empty()) {
        error_message_ = "Signal and kernel must not be empty";
        return;
    }

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_result_ = false;
    error_message_.clear();

    // Get mode string
    std::string mode_str = "same";
    if (conv_mode_ == ConvMode::Full) mode_str = "full";
    else if (conv_mode_ == ConvMode::Valid) mode_str = "valid";

    compute_thread_ = std::make_unique<std::thread>([this, mode_str]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            result_ = SignalProcessing::Convolve1D(signal_, kernel_, mode_str);

            if (result_.success) {
                has_result_ = true;
                spdlog::info("Convolution computed: {} output samples", result_.output_size);
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
