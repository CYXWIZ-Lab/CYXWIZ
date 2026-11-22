#pragma once

// What the Background Tasks window shows: the running tasks, the finished
// ones, the last training run in one line and the words of each. Pure data
// in and out (no ImGui, no task manager), so the window and its test agree.

#include <cstdint>
#include <ctime>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::task_list {

enum class State { Pending, Running, Completed, Failed, Cancelled };

// One task as the task manager reports it.
struct TaskFacts {
    uint64_t id = 0;
    std::string name;
    std::string message;  // the task's latest status line
    std::string error;    // why it failed
    State state = State::Pending;
    float progress = 0.0f;  // 0 .. 1
    bool cancellable = false;
    int64_t elapsed_ms = 0;  // running: so far; finished: start to end
    std::time_t ended_at = 0;  // finished tasks: when (wall clock)
};

// The latest training run, from its execution trace.
struct RunFacts {
    std::string status;            // "running", "completed", "early_stopped", "cancelled", "stopped", "failed"
    std::string backend;           // "arrayfire_cuda"
    std::string device;            // "NVIDIA GeForce GTX 1050 Ti"
    std::string latest_timestamp;  // "2026-11-22T04:05:10Z" (UTC)
    bool training_now = false;     // a run trains in this Engine now
};

struct Input {
    std::vector<TaskFacts> running;   // running and pending, oldest first
    std::vector<TaskFacts> finished;  // newest first, at most the shown count
    size_t finished_total = 0;        // finished tasks the history holds
    bool show_finished = true;
    std::optional<RunFacts> last_run;
    std::time_t now = 0;
};

enum class Mark { Waiting, Running, Done, Failed, Cancelled };

struct Row {
    uint64_t id = 0;
    Mark mark = Mark::Waiting;
    std::string name;
    std::string message;
    std::string error;        // failed tasks: "Failed: <why>"
    bool show_progress = false;
    float progress = 0.0f;
    std::string progress_text;  // "42%, 3 s" (a middle dot between)
    std::string took;           // finished: "took 12.6 s"; waiting: "waiting to start"
    std::string clock;          // finished: "08:11" (another day: "21 Nov 08:11")
    bool can_cancel = false;
};

struct RunLine {
    bool shown = false;
    Mark mark = Mark::Done;
    std::string text;    // "Last training run: cancelled"
    std::string detail;  // "on CUDA, NVIDIA GeForce GTX 1050 Ti", a middle dot, "08:05"
};

struct View {
    std::string summary;  // "1 running, 2 finished"
    bool cancel_all_enabled = false;
    std::string cancel_all_tooltip;
    RunLine last_run;
    std::vector<Row> running;
    std::string running_empty;  // set when nothing runs
    std::string finished_heading;
    std::vector<Row> finished;  // empty when hidden
    std::string finished_empty;  // shown, nothing finished yet
    std::string hidden_note;     // hidden, some finished
};

// The finished tasks the window lists.
constexpr size_t kFinishedShown = 20;

View Build(const Input& input);

// "11 ms", "12.6 s", "2 min 5 s", "1 h 3 min"
std::string Duration(int64_t ms);
// "08:11" on the day of `now`, else "21 Nov 08:11" (local time)
std::string Clock(std::time_t at, std::time_t now);
// An ISO UTC stamp ("2026-11-22T04:05:10Z") as a time; 0 when it is not one.
std::time_t ParseUtc(const std::string& iso);

}  // namespace cyxwiz::task_list
