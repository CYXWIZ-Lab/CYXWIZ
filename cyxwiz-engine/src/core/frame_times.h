#pragma once

// The time the Engine spends building its interface each frame, and which
// windows it goes to. Plain bookkeeping (no ImGui): the frame overlay and
// Monitor > Frames draw it, and its test feeds it by hand.
//
// Two times are kept apart everywhere this is shown:
// - the whole frame: what the screen shows, including waiting for the display
//   and the pauses the Engine takes when idle (the caller reads that from the
//   UI library);
// - building the interface: BeginFrame to EndFrame, the Engine's own work.

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

class FrameTimes {
public:
    static constexpr size_t kHistory = 180;      // frames kept for average and worst
    static constexpr double kSmoothing = 0.1;    // weight of a new frame in a window's average

    // Forgets everything (the overlay was switched on again).
    void Reset();

    void BeginFrame();
    // One window's time this frame. Several calls with one name add up.
    void AddWindow(const std::string& name, double milliseconds);
    // `build_milliseconds`: BeginFrame to EndFrame.
    void EndFrame(double build_milliseconds);

    struct WindowTime {
        std::string name;
        double now = 0.0;      // this frame, as measured
        double average = 0.0;  // smoothed over the last frames
    };

    struct Summary {
        size_t frames = 0;         // in the history (at most kHistory)
        double build_now = 0.0;    // the last finished frame
        double build_average = 0.0;
        double build_worst = 0.0;
        std::vector<WindowTime> windows;  // drawn in the last finished frame, costliest first
        double windows_total = 0.0;       // their sum, as measured
    };
    Summary Summarize() const;

    // The build times oldest first, for a plot.
    std::vector<float> History() const;

private:
    struct Entry {
        std::string name;
        double current = 0.0;   // the frame being built
        bool seen = false;      // in the frame being built
        double last = 0.0;      // the last finished frame
        bool last_seen = false;
        double average = 0.0;
        bool has_average = false;
    };

    std::vector<Entry> entries_;
    std::vector<float> history_;
    size_t next_ = 0;
    double last_build_ = 0.0;
};

}  // namespace cyxwiz
