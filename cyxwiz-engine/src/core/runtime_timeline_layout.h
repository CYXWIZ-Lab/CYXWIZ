#pragma once

// Where the Studio Debugger's Runtime Timeline draws its events. An event
// that carries its time (milliseconds from the run's start to its end) is
// drawn on a time axis: its bar starts when the event started and is as wide
// as it lasted, so gaps and overlaps between threads are what happened. A
// run recorded before events carried their time has none; its events are
// packed one after another, wider for longer, and the view says so.
//
// The axis can be narrowed to a window of the run (zoom), and the bars of a
// thread that would overlap at the drawn scale merge into one cluster with
// its count, so a busy thread's short events stay readable.

#include <algorithm>
#include <cmath>
#include <cstddef>
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
    bool visible = true;  // false: outside the zoom window (not drawn)
};

// A time window of the run, the axis when zoomed in.
struct TimelineWindow {
    double start_ms = 0.0;
    double end_ms = 0.0;
};

struct TimelineLayout {
    bool timed = false;       // true: a time axis; false: packed in order
    double start_ms = 0.0;    // the axis (timed only): the window when zoomed
    double end_ms = 0.0;
    double run_start_ms = 0.0;  // the whole run's span (timed only)
    double run_end_ms = 0.0;
    bool zoomed = false;      // the axis is a window narrower than the run
    float width = 0.0f;       // of everything drawn
    std::vector<std::vector<TimelineBar>> lanes;  // one bar per event, per lane
};

// `lanes`: each thread's events, oldest first. `width`: the area to fill when
// timed. A bar is never narrower than `min_bar`, so a very short event can
// still be pointed at. `window`: the part of the run to draw (clamped to the
// run; ignored when it is empty or the run has no event times).
inline TimelineLayout LayoutTimeline(const std::vector<std::vector<TimelineEvent>>& lanes, float width, float min_bar = 3.0f,
                                     const TimelineWindow* window = nullptr) {
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
        layout.run_start_ms = start;
        layout.run_end_ms = end;
        double axis_start = start, axis_end = end;
        if (window && window->end_ms > window->start_ms) {
            axis_start = std::max(start, window->start_ms);
            axis_end = std::min(end, window->end_ms);
            if (axis_end <= axis_start) {
                axis_start = start;
                axis_end = end;
            }
        }
        layout.start_ms = axis_start;
        layout.end_ms = axis_end;
        layout.zoomed = axis_start > start || axis_end < end;
        layout.width = width;
        const double scale = static_cast<double>(width - min_bar) / (axis_end - axis_start);
        for (size_t l = 0; l < lanes.size(); ++l) {
            for (const auto& event : lanes[l]) {
                const double duration = std::max(0.0, event.duration_ms);
                const double begin = event.end_ms - duration;
                TimelineBar bar;
                bar.visible = event.end_ms >= axis_start && begin <= axis_end;
                // Clipped to the window: an event that started before it starts at its edge.
                bar.x0 = static_cast<float>(std::max(0.0, (begin - axis_start) * scale));
                bar.x0 = std::min(bar.x0, width - min_bar);
                bar.x1 = std::max(bar.x0 + min_bar,
                                  static_cast<float>(std::min(static_cast<double>(width), (event.end_ms - axis_start) * scale)));
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
            layout.lanes[l].push_back({x, x + bar_width, true});
            x += bar_width + gap;
        }
        layout.width = std::max(layout.width, x);
    }
    return layout;
}

// Short bars of one lane that overlap at the drawn scale (more than `overlap`
// pixels; a busy thread's back-to-back events only touch and stay apart): one
// cluster. That overlap comes from bars too short to draw at this scale (each
// is at least the minimum width), so zooming in splits a cluster. A long bar
// (wider than `long_bar` pixels) never merges: when other events of its
// thread happened during it (a validation pass and the steps inside it) it is
// a span drawn under them. A cluster of one is a plain bar.
struct TimelineCluster {
    float x0 = 0.0f;
    float x1 = 0.0f;
    std::vector<size_t> events;  // indices into the lane, left to right
    bool span = false;           // a long event other bars overlap: drawn under them
};

inline std::vector<std::vector<TimelineCluster>> ClusterTimeline(const TimelineLayout& layout, float overlap = 0.5f,
                                                                 float long_bar = 12.0f) {
    std::vector<std::vector<TimelineCluster>> out(layout.lanes.size());
    for (size_t l = 0; l < layout.lanes.size(); ++l) {
        const auto& bars = layout.lanes[l];
        std::vector<size_t> order;
        for (size_t i = 0; i < bars.size(); ++i)
            if (bars[i].visible) order.push_back(i);
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return bars[a].x0 < bars[b].x0; });
        const auto is_long = [&](size_t i) { return bars[i].x1 - bars[i].x0 > long_bar; };
        std::vector<TimelineCluster> spans, shorts;
        for (size_t i : order) {
            if (is_long(i)) {
                TimelineCluster span{bars[i].x0, bars[i].x1, {i}, false};
                // A span: shorter events of the thread happened during it (the
                // steps inside a validation pass; not a step beside it).
                const float width_i = bars[i].x1 - bars[i].x0;
                for (size_t j : order)
                    if (j != i && bars[j].x1 - bars[j].x0 < width_i && bars[j].x0 < bars[i].x1 - overlap &&
                        bars[j].x1 > bars[i].x0 + overlap)
                        span.span = true;
                spans.push_back(std::move(span));
            } else if (!shorts.empty() && bars[i].x0 < shorts.back().x1 - overlap) {
                shorts.back().x1 = std::max(shorts.back().x1, bars[i].x1);
                shorts.back().events.push_back(i);
            } else {
                shorts.push_back({bars[i].x0, bars[i].x1, {i}, false});
            }
        }
        // Spans first, so they are drawn under the bars of their thread.
        out[l] = std::move(spans);
        out[l].insert(out[l].end(), shorts.begin(), shorts.end());
    }
    return out;
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

// A mark on an axis that spans `span_ms`: precise enough that five marks
// across it read differently ("1.43 s" .. "2.01 s" over 0.58 s, "1652.4 ms"
// when zoomed into a few milliseconds).
inline std::string TimelineAxisText(double ms, double span_ms) {
    if (ms < 0.0) ms = 0.0;
    char text[48];
    if (span_ms >= 20000.0 || ms >= 60000.0) return TimelineTimeText(ms);
    if (span_ms >= 2000.0) std::snprintf(text, sizeof(text), "%.1f s", ms / 1000.0);
    else if (span_ms >= 200.0) std::snprintf(text, sizeof(text), "%.2f s", ms / 1000.0);
    else if (span_ms >= 20.0) std::snprintf(text, sizeof(text), "%.3f s", ms / 1000.0);
    else if (span_ms >= 2.0) std::snprintf(text, sizeof(text), "%.1f ms", ms);
    else if (span_ms >= 0.2) std::snprintf(text, sizeof(text), "%.2f ms", ms);
    else std::snprintf(text, sizeof(text), "%.3f ms", ms);
    return text;
}

}  // namespace cyxwiz
