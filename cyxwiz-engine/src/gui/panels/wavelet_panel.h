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
 * WaveletPanel - Discrete Wavelet Transform Tool
 *
 * Features:
 * - Multi-level DWT decomposition
 * - Multiple wavelet families (Haar, Daubechies)
 * - Visualization of approximation and detail coefficients
 * - Signal reconstruction and error analysis
 * - Coefficient thresholding for denoising
 */
class WaveletPanel {
public:
    WaveletPanel();
    ~WaveletPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

private:
    void RenderToolbar();
    void RenderSignalInput();
    void RenderWaveletSettings();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderDecomposition();
    void RenderReconstruction();
    void RenderDenoising();

    void GenerateSignal();
    void DecomposeAsync();
    void ReconstructSignal();
    void ApplyDenoising();

    bool visible_ = false;

    // Signal generation
    enum class SignalType { Sine, Step, Noisy, Chirp, Custom };
    SignalType signal_type_ = SignalType::Noisy;
    double signal_freq_ = 10.0;
    int signal_length_ = 256;
    double noise_level_ = 0.3;

    // Signal data
    std::vector<double> signal_;
    std::vector<double> time_axis_;

    // Wavelet settings
    enum class WaveletType { Haar, DB1, DB2, DB3, DB4 };
    WaveletType wavelet_type_ = WaveletType::Haar;
    int decomp_levels_ = 3;

    // Results
    WaveletResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Reconstruction
    std::vector<double> reconstructed_;
    bool has_reconstruction_ = false;
    double reconstruction_error_ = 0.0;

    // Denoising
    double threshold_ = 0.1;
    enum class ThresholdType { Hard, Soft };
    ThresholdType threshold_type_ = ThresholdType::Soft;
    std::vector<double> denoised_;
    bool has_denoised_ = false;

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;
};

} // namespace cyxwiz
