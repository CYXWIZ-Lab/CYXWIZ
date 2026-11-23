#include "frame_times.h"

#include <algorithm>

namespace cyxwiz {

void FrameTimes::Reset() {
    entries_.clear();
    history_.clear();
    next_ = 0;
    last_build_ = 0.0;
}

void FrameTimes::BeginFrame() {
    for (auto& entry : entries_) {
        entry.current = 0.0;
        entry.seen = false;
    }
}

void FrameTimes::AddWindow(const std::string& name, double milliseconds) {
    for (auto& entry : entries_) {
        if (entry.name == name) {
            entry.current += milliseconds;
            entry.seen = true;
            return;
        }
    }
    Entry entry;
    entry.name = name;
    entry.current = milliseconds;
    entry.seen = true;
    entries_.push_back(std::move(entry));
}

void FrameTimes::EndFrame(double build_milliseconds) {
    last_build_ = build_milliseconds;
    if (history_.size() < kHistory) {
        history_.push_back(static_cast<float>(build_milliseconds));
        next_ = history_.size() % kHistory;
    } else {
        history_[next_] = static_cast<float>(build_milliseconds);
        next_ = (next_ + 1) % kHistory;
    }
    for (auto& entry : entries_) {
        entry.last = entry.current;
        entry.last_seen = entry.seen;
        if (!entry.seen) continue;  // a window that was not drawn keeps its average
        entry.average = entry.has_average ? entry.average + kSmoothing * (entry.current - entry.average) : entry.current;
        entry.has_average = true;
    }
}

FrameTimes::Summary FrameTimes::Summarize() const {
    Summary summary;
    summary.frames = history_.size();
    summary.build_now = last_build_;
    double sum = 0.0;
    for (const float ms : history_) {
        sum += ms;
        summary.build_worst = std::max(summary.build_worst, static_cast<double>(ms));
    }
    if (!history_.empty()) summary.build_average = sum / static_cast<double>(history_.size());
    for (const auto& entry : entries_) {
        if (!entry.last_seen) continue;
        summary.windows.push_back({entry.name, entry.last, entry.average});
        summary.windows_total += entry.last;
    }
    std::stable_sort(summary.windows.begin(), summary.windows.end(),
                     [](const WindowTime& a, const WindowTime& b) { return a.now > b.now; });
    return summary;
}

std::vector<float> FrameTimes::History() const {
    if (history_.size() < kHistory) return history_;
    std::vector<float> ordered(history_.size());
    for (size_t i = 0; i < history_.size(); ++i) ordered[i] = history_[(next_ + i) % history_.size()];
    return ordered;
}

}  // namespace cyxwiz
