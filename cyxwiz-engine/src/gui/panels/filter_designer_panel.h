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
 * FilterDesignerPanel - FIR Filter Design Tool
 *
 * Features:
 * - Design lowpass, highpass, bandpass, bandstop filters
 * - Configurable cutoff frequencies and filter order
 * - Frequency response visualization (magnitude & phase)
 * - Apply filter to test signal
 * - Export filter coefficients
 */
class FilterDesignerPanel {
public:
    FilterDesignerPanel();
    ~FilterDesignerPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

private:
    void RenderToolbar();
    void RenderFilterSettings();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderFrequencyResponse();
    void RenderCoefficients();
    void RenderSignalFiltering();

    void DesignFilterAsync();
    void GenerateTestSignal();
    void ApplyFilter();

    bool visible_ = false;

    // Filter type
    enum class FilterType { Lowpass, Highpass, Bandpass, Bandstop };
    FilterType filter_type_ = FilterType::Lowpass;

    // Filter parameters
    double cutoff1_ = 200.0;  // Primary cutoff (Hz)
    double cutoff2_ = 500.0;  // Secondary cutoff for bandpass/bandstop
    int filter_order_ = 51;   // Number of taps (odd for symmetric FIR)
    double sample_rate_ = 2000.0;

    // Results
    FilterCoefficients filter_;
    bool has_filter_ = false;
    std::string error_message_;

    // Test signal
    std::vector<double> test_signal_;
    std::vector<double> filtered_signal_;
    std::vector<double> time_axis_;
    bool has_filtered_ = false;

    // Test signal parameters
    double test_freq1_ = 100.0;
    double test_freq2_ = 400.0;
    int test_samples_ = 1024;

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;
};

} // namespace cyxwiz
