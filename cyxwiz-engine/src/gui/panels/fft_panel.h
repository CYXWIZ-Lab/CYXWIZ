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
 * FFTPanel - Fast Fourier Transform Tool
 *
 * Features:
 * - Signal generation (sine, square, sawtooth, noise)
 * - Manual signal input
 * - FFT computation with magnitude and phase
 * - Time domain and frequency domain visualization
 * - Peak frequency detection
 */
class FFTPanel {
public:
    FFTPanel();
    ~FFTPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

private:
    void RenderToolbar();
    void RenderSignalInput();
    void RenderGenerateSignal();
    void RenderManualInput();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderTimeDomain();
    void RenderFrequencyDomain();
    void RenderPhasePlot();
    void RenderPeaks();

    void GenerateSignal();
    void ComputeAsync();
    void FindPeaks();

    bool visible_ = false;

    // Signal input mode
    enum class InputMode { Generate, Manual };
    InputMode input_mode_ = InputMode::Generate;

    // Signal generation parameters
    enum class SignalType { Sine, Square, Sawtooth, WhiteNoise, Composite };
    SignalType signal_type_ = SignalType::Sine;
    double frequency1_ = 100.0;  // Primary frequency (Hz)
    double frequency2_ = 250.0;  // Secondary frequency for composite
    double amplitude1_ = 1.0;
    double amplitude2_ = 0.5;
    double sample_rate_ = 1000.0;  // Hz
    int num_samples_ = 1024;
    double noise_snr_ = 20.0;  // SNR in dB
    bool add_noise_ = false;

    // Signal data
    std::vector<double> signal_;
    std::vector<double> time_axis_;

    // Results
    FFTResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Peak detection
    struct Peak {
        double frequency;
        double magnitude;
        int index;
    };
    std::vector<Peak> peaks_;
    int num_peaks_to_find_ = 5;
    double peak_threshold_ = 0.1;  // As fraction of max magnitude

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;
};

} // namespace cyxwiz
