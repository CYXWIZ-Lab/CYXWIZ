#pragma once

// Where the Studio Debugger's Runtime Timeline draws its events. An event
// that carries its time (milliseconds from the run's start to its end) is
// drawn on a time axis: its bar starts when the event started and is as wide
// as it lasted, so gaps and overlaps between threads are what happened. A
// run recorded before events carried their time has none; its events are
// packed one after another, wider for longer, and the view says so.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace cyxwiz {

struct TimelineEvent {
    double end_ms = -1.0;      // from the run's start; below 0: not recorded
    double duration_ms = 0.0;
};

struct TimelineBar {
    float x0 = 0.0f;  // from the left of the drawing area
    float x1 = 0.0f;
};

struct TimelineLayout {
    bool timed = false;       // true: a time axis; false: packed in order
    double start_ms = 0.0;    // the axis (timed only)
    double end_ms = 0.0;
    float width = 0.0f;       // of everything drawn
    std::vector<std::vector<TimelineBar>> lanes;  // one bar per event, per lane
};

// `lanes`: each thread's events, oldest first. `width`: the area to fill when
// timed. A bar is never narrower than `min_bar`, so a very short event can
// still be pointed at.
inline TimelineLayout LayoutTimeline(const std::vector<std::vector<TimelineEvent>>& lanes, float width, float min_bar = 3.0f) {
    TimelineLayout layout;
    layout.lanes.resize(lanes.size());
    bool any = false, all_timed = true;
    double start = 0.0, end = 0.0, longest = 1.0;
    for (const auto& lane : lanes) {
        for (const auto& event : lane) {
            const double duration = std::max(0.0, event.duration_ms);
            longest = std::max(longest, duration);
            if (event.end_ms < 0.0) {
                all_timed = false;
                continue;
            }
            const double begin = event.end_ms - duration;
            start = any ? std::min(start, begin) : begin;
            end = any ? std::max(end, event.end_ms) : event.end_ms;
            any = true;
        }
    }
    if (any && all_timed && end > start && width > min_bar) {
        layout.timed = true;
        layout.start_ms = start;
        layout.end_ms = end;
        layout.width = width;
        const double scale = static_cast<double>(width - min_bar) / (end - start);
        for (size_t l = 0; l < lanes.size(); ++l) {
            for (const auto& event : lanes[l]) {
                const double duration = std::max(0.0, event.duration_ms);
                TimelineBar bar;
                bar.x0 = static_cast<float>((event.end_ms - duration - start) * scale);
                bar.x1 = std::max(bar.x0 + min_bar, static_cast<float>((event.end_ms - start) * scale));
                layout.lanes[l].push_back(bar);
            }
        }
        return layout;
    }
    // Packed: in order, a gap between bars, wider for longer.
    const float gap = 4.0f, narrow = 7.0f, wide = 165.0f;
    for (size_t l = 0; l < lanes.size(); ++l) {
        float x = 0.0f;
        for (const auto& event : lanes[l]) {
            const double duration = std::max(0.0, event.duration_ms);
            const float bar_width = duration > 0.0 ? narrow + static_cast<float>(duration / longest) * wide : narrow;
            layout.lanes[l].push_back({x, x + bar_width});
            x += bar_width + gap;
        }
        layout.width = std::max(layout.width, x);
    }
    return layout;
}

// "0 ms", "250 ms", "1.5 s", "2 min 5 s": a time on the axis.
inline std::string TimelineTimeText(double ms) {
    char text[48];
    if (ms < 0.0) ms = 0.0;
    if (ms < 1000.0) std::snprintf(text, sizeof(text), "%.0f ms", ms);
    else if (ms < 60000.0) std::snprintf(text, sizeof(text), "%.1f s", ms / 1000.0);
    else std::snprintf(text, sizeof(text), "%d min %d s", static_cast<int>(ms / 60000.0), static_cast<int>(std::fmod(ms, 60000.0) / 1000.0));
    return text;
}

}  // namespace cyxwiz
