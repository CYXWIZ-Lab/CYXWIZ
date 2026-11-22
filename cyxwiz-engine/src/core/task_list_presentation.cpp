#include "task_list_presentation.h"

#include "rl_device_presentation.h"

#include <cmath>
#include <cstdio>

namespace cyxwiz::task_list {
namespace {

const char* const kDot = " \xC2\xB7 ";

std::string Plural(size_t n, const char* one, const char* many) {
    return std::to_string(n) + " " + (n == 1 ? one : many);
}

std::tm LocalTime(std::time_t t) {
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    return tm;
}

Row MakeRow(const TaskFacts& t, std::time_t now) {
    Row row;
    row.id = t.id;
    row.name = t.name;
    row.message = t.message;
    switch (t.state) {
        case State::Pending:
            row.mark = Mark::Waiting;
            row.took = "waiting to start";
            break;
        case State::Running: {
            row.mark = Mark::Running;
            row.show_progress = true;
            row.progress = std::fmax(0.0f, std::fmin(1.0f, t.progress));
            char percent[16];
            std::snprintf(percent, sizeof percent, "%.0f%%", row.progress * 100.0f);
            row.progress_text = std::string(percent) + kDot + Duration(t.elapsed_ms);
            break;
        }
        case State::Completed:
        case State::Failed:
        case State::Cancelled:
            row.mark = t.state == State::Completed ? Mark::Done
                       : t.state == State::Failed  ? Mark::Failed
                                                   : Mark::Cancelled;
            row.took = (t.state == State::Cancelled ? "cancelled after " : "took ") + Duration(t.elapsed_ms);
            if (t.ended_at != 0) row.clock = Clock(t.ended_at, now);
            if (t.state == State::Failed && !t.error.empty()) row.error = "Failed: " + t.error;
            break;
    }
    row.can_cancel = t.cancellable && (t.state == State::Pending || t.state == State::Running);
    return row;
}

RunLine MakeRunLine(const RunFacts& run, std::time_t now) {
    RunLine line;
    if (run.status.empty() && run.backend.empty()) return line;
    line.shown = true;
    const std::string& s = run.status;
    if (run.training_now) {
        line.mark = Mark::Running;
        line.text = "Training now";
    } else if (s == "running") {
        // Saved as running, while nothing trains here: it ended without a record (the Engine closed during it) or runs in another Engine
        line.mark = Mark::Cancelled;
        line.text = "Last training run: no end recorded";
    } else {
        const char* word = s == "completed"       ? "finished"
                           : s == "early_stopped" ? "stopped early"
                           : s == "cancelled"     ? "cancelled"
                           : s == "stopped"       ? "stopped"
                           : s == "failed"        ? "failed"
                                                  : nullptr;
        line.mark = (s == "completed" || s == "early_stopped") ? Mark::Done
                    : s == "failed"                            ? Mark::Failed
                                                               : Mark::Cancelled;
        line.text = std::string("Last training run: ") + (word ? word : s.c_str());
    }
    if (!run.backend.empty()) {
        line.detail = "on " + rl_device::BackendWord(run.backend);
        if (!run.device.empty()) line.detail += ", " + run.device;
    }
    if (!run.training_now) {
        if (const std::time_t at = ParseUtc(run.latest_timestamp); at != 0)
            line.detail += (line.detail.empty() ? "" : kDot) + Clock(at, now);
    }
    return line;
}

}  // namespace

std::string Duration(int64_t ms) {
    if (ms < 0) ms = 0;
    char buf[32];
    if (ms < 1000) return std::to_string(ms) + " ms";
    if (ms < 60000) {
        std::snprintf(buf, sizeof buf, "%.1f s", static_cast<double>(ms) / 1000.0);
        return buf;
    }
    const int64_t seconds = ms / 1000;
    if (seconds < 3600) return std::to_string(seconds / 60) + " min " + std::to_string(seconds % 60) + " s";
    return std::to_string(seconds / 3600) + " h " + std::to_string((seconds % 3600) / 60) + " min";
}

std::string Clock(std::time_t at, std::time_t now) {
    const std::tm t = LocalTime(at);
    const std::tm n = LocalTime(now);
    char buf[32];
    if (t.tm_year == n.tm_year && t.tm_yday == n.tm_yday) {
        std::strftime(buf, sizeof buf, "%H:%M", &t);
    } else {
        std::strftime(buf, sizeof buf, "%d %b %H:%M", &t);
    }
    return buf;
}

std::time_t ParseUtc(const std::string& iso) {
    int y = 0, mo = 0, d = 0, h = 0, mi = 0, s = 0;
    char z = 0;
    if (std::sscanf(iso.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d%c", &y, &mo, &d, &h, &mi, &s, &z) != 7 || z != 'Z')
        return 0;
    if (mo < 1 || mo > 12 || d < 1 || d > 31 || h > 23 || mi > 59 || s > 60) return 0;
    std::tm tm{};
    tm.tm_year = y - 1900;
    tm.tm_mon = mo - 1;
    tm.tm_mday = d;
    tm.tm_hour = h;
    tm.tm_min = mi;
    tm.tm_sec = s;
#ifdef _WIN32
    const std::time_t t = _mkgmtime(&tm);
#else
    const std::time_t t = timegm(&tm);
#endif
    return t < 0 ? 0 : t;
}

View Build(const Input& in) {
    View v;
    const size_t running = in.running.size();
    const size_t finished = in.finished_total;
    if (running == 0) {
        v.summary = finished == 0 ? "Nothing running" : "Nothing running, " + std::to_string(finished) + " finished";
    } else {
        v.summary = std::to_string(running) + " running" + (finished ? ", " + std::to_string(finished) + " finished" : "");
    }

    size_t cancellable = 0;
    for (const TaskFacts& t : in.running) {
        v.running.push_back(MakeRow(t, in.now));
        if (v.running.back().can_cancel) ++cancellable;
    }
    v.cancel_all_enabled = cancellable > 0;
    v.cancel_all_tooltip = cancellable > 0 ? "Cancel every task that can stop (" + std::to_string(cancellable) + ")"
                           : running > 0   ? "The running tasks cannot be cancelled"
                                           : "Nothing running can be cancelled";
    if (running == 0) v.running_empty = "Nothing is running. Loads, training and cache work show here while they run.";

    if (in.last_run) v.last_run = MakeRunLine(*in.last_run, in.now);

    if (in.show_finished) {
        v.finished_heading = "Finished";
        if (finished > in.finished.size())
            v.finished_heading += std::string(kDot) + "last " + std::to_string(in.finished.size()) + " of " +
                                  std::to_string(finished);
        for (const TaskFacts& t : in.finished) v.finished.push_back(MakeRow(t, in.now));
        if (in.finished.empty()) v.finished_empty = "Nothing has finished yet in this session.";
    } else if (finished > 0) {
        v.hidden_note = Plural(finished, "finished task", "finished tasks") + " hidden. Tick Show finished to see " +
                        (finished == 1 ? "it." : "them.");
    }
    return v;
}

}  // namespace cyxwiz::task_list
