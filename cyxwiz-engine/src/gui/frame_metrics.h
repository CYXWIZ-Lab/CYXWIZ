#pragma once

// Frame times: the overlay behind Tools > Diagnostics > Frame Time Overlay
// and the figures of Monitor > Frames. The bookkeeping is core/frame_times;
// this file times the frame and draws the overlay.
//
// Two times, never under one label: the whole frame (what the screen shows,
// with the waits for the display and the Engine's idle pauses) and building
// the interface (the Engine's own work between BeginFrame and EndFrame).
//
// Cost when nothing watches: one flag read per timed window. Frames are
// recorded only while the overlay is shown or Monitor > Frames is open, and
// the history starts again each time recording starts.

#include "../core/frame_times.h"

#include <chrono>
#include <functional>

namespace gui {

class FrameMetrics {
public:
    static FrameMetrics& Instance();

    // The overlay.
    bool enabled() const { return enabled_; }
    void set_enabled(bool on) { enabled_ = on; }
    bool* enabled_ptr() { return &enabled_; }

    // Monitor > Frames calls this every frame it is drawn: frames are
    // recorded while it does, with or without the overlay.
    void Watch() { watched_frames_ = 2; }
    // True while frames are being recorded.
    bool recording() const { return recording_; }

    // Call once per frame before the windows render, and once after.
    void BeginFrame();
    void EndFrame();

    // Record one window's time this frame.
    void AddPanelTime(const char* name, double milliseconds);

    // Draw the overlay (no-op when it is off).
    void Render();

    const cyxwiz::FrameTimes& times() const { return times_; }

    // The overlay's "More in Monitor > Frames" link.
    void SetOpenMonitor(std::function<void()> open) { open_monitor_ = std::move(open); }

private:
    FrameMetrics() = default;

    bool enabled_ = false;
    int watched_frames_ = 0;
    bool recording_ = false;
    std::chrono::steady_clock::time_point frame_start_{};
    cyxwiz::FrameTimes times_;
    std::function<void()> open_monitor_;
};

// Times a window's Render() while frames are recorded.
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
