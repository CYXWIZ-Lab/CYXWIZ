#pragma once

// Frame-time overlay and per-panel timers (TOFIX129 step 0.2). Developer
// tool behind Tools > Diagnostics > Frame Time Overlay: frame time, frames
// per second, the panels that cost the most this frame, draw data and the
// font atlas size, so every later cost claim is measured, not guessed.
//
// Cost when off: one clock read per timed panel. The history and the
// overlay are only kept while the overlay is shown.

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

namespace gui {

class FrameMetrics {
public:
    static FrameMetrics& Instance();

    bool enabled() const { return enabled_; }
    void set_enabled(bool on) { enabled_ = on; }
    bool* enabled_ptr() { return &enabled_; }

    // Call once per frame before the panels render, and once after.
    void BeginFrame();
    void EndFrame();

    // Record one panel's time this frame.
    void AddPanelTime(const char* name, double milliseconds);

    // Draw the overlay (no-op when disabled).
    void Render();

    struct PanelCost {
        std::string name;
        double milliseconds = 0.0;  // smoothed over the last frames
    };

private:
    FrameMetrics() = default;

    bool enabled_ = false;
    std::chrono::steady_clock::time_point frame_start_{};
    std::vector<float> frame_history_;  // last 180 frame times, ms
    size_t history_pos_ = 0;
    std::vector<PanelCost> panels_;     // this frame, then smoothed
    double panels_total_ms_ = 0.0;
};

// Times a panel's Render() while the overlay is enabled.
class PanelTimer {
public:
    explicit PanelTimer(const char* name);
    ~PanelTimer();

private:
    const char* name_;
    bool active_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace gui
