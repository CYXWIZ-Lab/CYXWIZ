#pragma once

#include <cyxwiz/signal_processing.h>
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>

namespace cyxwiz {

/**
 * ConvolutionPanel - Convolution Calculator Tool
 *
 * Features:
 * - 1D and 2D convolution
 * - Multiple convolution modes (full, same, valid)
 * - Preset kernels (edge detection, blur, sharpen)
 * - Visualization of signal, kernel, and result
 */
class ConvolutionPanel {
public:
    ConvolutionPanel();
    ~ConvolutionPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

private:
    void RenderToolbar();
    void RenderInputs();
    void RenderSignalInput();
    void RenderKernelInput();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderResultPlot();
    void RenderMathView();

    void GenerateSignal();
    void ApplyKernelPreset();
    void ComputeAsync();

    bool visible_ = false;

    // Input mode
    enum class ConvMode { Full, Same, Valid };
    ConvMode conv_mode_ = ConvMode::Same;

    // Signal data
    std::vector<double> signal_;
    int signal_size_ = 16;

    // Signal generation
    enum class SignalType { Step, Impulse, Ramp, Sine, Custom };
    SignalType signal_type_ = SignalType::Step;

    // Kernel data
    std::vector<double> kernel_;
    int kernel_size_ = 5;

    // Kernel presets
    enum class KernelPreset { Custom, MovingAverage, Gaussian, Derivative, Laplacian, EdgeDetect };
    KernelPreset kernel_preset_ = KernelPreset::MovingAverage;

    // Results
    ConvolutionResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;
};

} // namespace cyxwiz
