#include "runtime_log_presentation.h"

#include <ctime>
#include <iomanip>
#include <sstream>

namespace cyxwiz::logs {

namespace {

std::tm ToTm(std::time_t time, bool utc) {
    std::tm value{};
#ifdef _WIN32
    if (utc)
        gmtime_s(&value, &time);
    else
        localtime_s(&value, &time);
#else
    if (utc)
        gmtime_r(&time, &value);
    else
        localtime_r(&time, &value);
#endif
    return value;
}

std::string Format(std::chrono::system_clock::time_point timestamp, bool utc,
                   const char *pattern, const char *suffix) {
    const auto time = std::chrono::system_clock::to_time_t(timestamp);
    const std::tm parts = ToTm(time, utc);
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                      timestamp.time_since_epoch())
                      .count() %
                  1000;
    if (millis < 0)
        millis += 1000;
    std::ostringstream out;
    out << std::put_time(&parts, pattern) << '.' << std::setfill('0') << std::setw(3)
        << millis << suffix;
    return out.str();
}

}  // namespace

size_t LevelIndex(RuntimeLogLevel level) {
    const auto index = static_cast<size_t>(level);
    return index < kLevelLabels.size() ? index : 2;
}

const char *LevelLabel(RuntimeLogLevel level) { return kLevelLabels[LevelIndex(level)]; }

bool IsProblemLevel(RuntimeLogLevel level) {
    return level == RuntimeLogLevel::Error || level == RuntimeLogLevel::Critical;
}

std::string FormatCount(uint64_t value) {
    std::string out = std::to_string(value);
    for (size_t pos = out.size(); pos > 3; pos -= 3)
        out.insert(pos - 3, ",");
    return out;
}

std::string FormatLocalTime(std::chrono::system_clock::time_point timestamp) {
    return Format(timestamp, false, "%H:%M:%S", "");
}

std::string FormatUtcTimestamp(std::chrono::system_clock::time_point timestamp) {
    return Format(timestamp, true, "%Y-%m-%dT%H:%M:%S", "Z");
}

std::string FormatLocalTimestamp(std::chrono::system_clock::time_point timestamp) {
    return Format(timestamp, false, "%Y-%m-%d %H:%M:%S", "");
}

std::string DeviceLabel(const RuntimeLogEvent &event) {
    if (!event.backend.empty() && event.device_id >= 0)
        return event.backend + " \xC2\xB7 " + std::to_string(event.device_id);
    if (!event.backend.empty())
        return event.backend;
    if (event.device_id >= 0)
        return "device " + std::to_string(event.device_id);
    return {};
}

MessageLine FirstLine(const std::string &message) {
    MessageLine line;
    const size_t eol = message.find_first_of("\r\n");
    if (eol == std::string::npos) {
        line.text = message;
        return line;
    }
    line.text = message.substr(0, eol);
    line.more_lines = message.find_first_not_of("\r\n", eol) != std::string::npos;
    return line;
}

std::string DetailsTitle(const RuntimeLogEvent &event) {
    std::string title = FirstLine(event.message).text;
    constexpr size_t kMax = 90;
    if (title.size() > kMax) {
        size_t cut = kMax;
        while (cut > 0 && (static_cast<unsigned char>(title[cut]) & 0xC0) == 0x80)
            --cut;  // keep UTF-8 sequences whole
        title = title.substr(0, cut) + "...";
    }
    return title;
}

StatusView BuildStatusView(const StatusInput &in) {
    StatusView view;
    view.live = !in.paused;
    view.state = in.paused ? "Paused at #" + FormatCount(in.high_water) : "Live";
    view.showing = "Showing " + FormatCount(in.shown) + " of " + FormatCount(in.matched) +
                   " matched";
    view.retained = "Retained " + FormatCount(in.stats.size) + " / " +
                    FormatCount(in.stats.capacity);
    view.evicted = "Evicted " + FormatCount(in.stats.evicted_count);
    view.losses = "Dropped " + FormatCount(in.stats.dropped_count) + " \xC2\xB7 Rejected " +
                  FormatCount(in.stats.rejected_count) + " \xC2\xB7 Suppressed " +
                  FormatCount(in.stats.suppressed_count);
    view.has_losses = in.stats.dropped_count != 0 || in.stats.rejected_count != 0 ||
                      in.stats.suppressed_count != 0;
    view.high_water = "High-water #" + FormatCount(in.high_water);
    if (in.hidden_through != 0)
        view.hidden = "Cleared view hides events through #" + FormatCount(in.hidden_through);
    if (in.truncated && in.matched > in.shown) {
        view.truncation = "Display limited to the newest " + FormatCount(in.display_limit) +
                          " rows. Narrow the filter or export to see all.";
    }
    return view;
}

std::vector<std::pair<std::string, std::string>> DetailFields(const RuntimeLogEvent &e) {
    std::vector<std::pair<std::string, std::string>> fields;
    auto add = [&fields](const char *key, const std::string &value) {
        if (!value.empty())
            fields.emplace_back(key, value);
    };
    add("Time (UTC)", FormatUtcTimestamp(e.timestamp_utc));
    add("Local time", FormatLocalTimestamp(e.timestamp_utc));
    add("Sequence", "#" + FormatCount(e.sequence));
    add("Level", LevelLabel(e.level));
    add("Category", e.category);
    add("Source", e.source);
    add("Event", e.event_name);
    add("Thread", e.thread_id);
    add("Code", e.primary_error_code);
    std::string issues;
    for (const auto &code : e.issue_codes)
        issues += (issues.empty() ? "" : ", ") + code;
    add("Issue codes", issues);
    add("Run", e.run_id);
    add("Task", e.task_id != 0 ? std::to_string(e.task_id) : std::string());
    add("Backend", e.backend);
    add("Device", e.device_id >= 0 ? std::to_string(e.device_id) : std::string());
    add("Device name", e.device_name);
    add("Node", e.node_id >= 0 ? std::to_string(e.node_id) : std::string());
    add("Dataset", e.dataset_name);
    add("Phase", e.diagnostic_phase);
    add("Component", e.component);
    for (const auto &[key, value] : e.details)
        fields.emplace_back(key, value.empty() ? std::string("(empty)") : value);
    return fields;
}

std::string FormatRow(const RuntimeLogEvent &e) {
    std::ostringstream out;
    out << '#' << e.sequence << ' ' << FormatUtcTimestamp(e.timestamp_utc)
        << " level=" << LevelLabel(e.level) << " category=" << e.category;
    if (!e.source.empty())
        out << " source=" << e.source;
    if (!e.primary_error_code.empty())
        out << " code=" << e.primary_error_code;
    if (!e.run_id.empty())
        out << " run=" << e.run_id;
    if (e.task_id != 0)
        out << " task=" << e.task_id;
    if (!e.backend.empty())
        out << " backend=" << e.backend;
    if (e.device_id >= 0)
        out << " device_id=" << e.device_id;
    if (!e.thread_id.empty())
        out << " thread=" << e.thread_id;
    out << " | " << e.message;
    return out.str();
}

size_t CountFieldFilters(bool category, bool source, bool code, bool run, bool backend,
                         bool task, bool device) {
    return static_cast<size_t>(category) + static_cast<size_t>(source) +
           static_cast<size_t>(code) + static_cast<size_t>(run) +
           static_cast<size_t>(backend) + static_cast<size_t>(task) +
           static_cast<size_t>(device);
}

}  // namespace cyxwiz::logs
