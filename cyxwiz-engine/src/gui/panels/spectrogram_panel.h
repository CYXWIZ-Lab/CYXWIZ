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
 * SpectrogramPanel - Time-Frequency Analysis Tool
 *
 * Features:
 * - STFT-based spectrogram computation
 * - Multiple window functions (Hamming, Hann, Blackman)
 * - Configurable window size and hop size
 * - Heatmap visualization with color scale options
 * - Signal generation or manual input
 */
class SpectrogramPanel {
public:
    SpectrogramPanel();
    ~SpectrogramPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

private:
    void RenderToolbar();
    void RenderSignalInput();
    void RenderParameters();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderSpectrogram();
    void RenderSignalPreview();

    void GenerateSignal();
    void ComputeAsync();

    bool visible_ = false;

    // Signal generation parameters
    enum class SignalType { ChirpLinear, ChirpExponential, MultiTone, AM, FM };
    SignalType signal_type_ = SignalType::ChirpLinear;
    double start_freq_ = 100.0;
    double end_freq_ = 500.0;
    double carrier_freq_ = 440.0;
    double modulation_freq_ = 10.0;
    double sample_rate_ = 4000.0;
    int num_samples_ = 4096;

    // Signal data
    std::vector<double> signal_;
    std::vector<double> time_axis_;

    // Spectrogram parameters
    enum class WindowType { Hamming, Hann, Blackman, Rectangular };
    WindowType window_type_ = WindowType::Hamming;
    int window_size_ = 256;
    int hop_size_ = 64;

    // Results
    SpectrogramResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Visualization
    enum class ColorScale { Viridis, Plasma, Inferno, Grayscale };
    ColorScale color_scale_ = ColorScale::Viridis;
    bool log_scale_ = true;
    double dynamic_range_ = 60.0;  // dB

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;
};

} // namespace cyxwiz
