// What the Background Tasks window says (TOFIX129 C6): the summary, the
// running and finished rows, Cancel all and its reason, the last training
// run in one line, the empty and hidden texts. Real data from the dev PC.
#include "../src/core/task_list_presentation.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

using namespace cyxwiz::task_list;

const std::string kDot = " \xC2\xB7 ";

TaskFacts Finished(uint64_t id, const char* name, const char* message, int64_t ms, std::time_t at) {
    TaskFacts t;
    t.id = id;
    t.name = name;
    t.message = message;
    t.state = State::Completed;
    t.elapsed_ms = ms;
    t.ended_at = at;
    return t;
}

Input DevPc(std::time_t now) {
    Input in;
    in.now = now;
    in.finished = {Finished(3, "Prune the project's prepared-data caches", "nothing to remove", 11, now - 60),
                   Finished(2, "Scanning project files", "Completed", 12600, now - 62)};
    in.finished_total = 2;
    return in;
}

void CheckDurations() {
    Check(Duration(11) == "11 ms", "milliseconds below a second");
    Check(Duration(12600) == "12.6 s", "seconds with one decimal");
    Check(Duration(125000) == "2 min 5 s", "minutes and seconds");
    Check(Duration(3780000) == "1 h 3 min", "hours and minutes");
    Check(Duration(-5) == "0 ms", "a negative time reads as none");
}

void CheckParseAndClock() {
    const std::time_t t = ParseUtc("2026-11-22T04:05:10Z");
    Check(t != 0, "an ISO UTC stamp is read");
    Check(ParseUtc("2026-11-22 04:05:10") == 0, "a stamp without T and Z is not read");
    Check(ParseUtc("2026-13-22T04:05:10Z") == 0, "month 13 is not read");
    Check(ParseUtc("") == 0, "no stamp is not read");
    Check(ParseUtc("2026-11-22T04:05:10Z") - ParseUtc("2026-11-22T04:04:10Z") == 60, "stamps a minute apart");
    const std::string today = Clock(t, t + 60);
    Check(today.size() == 5 && today[2] == ':', "the same day shows the time only: " + today);
    const std::string other = Clock(t, t + 3 * 86400);
    Check(other.size() > 5 && other.find(':') != std::string::npos, "another day shows the date too: " + other);
}

void CheckIdleWithHistory() {
    const std::time_t now = ParseUtc("2026-11-22T04:12:00Z");
    const View v = Build(DevPc(now));
    Check(v.summary == "Nothing running, 2 finished", "summary with nothing running: " + v.summary);
    Check(!v.cancel_all_enabled, "Cancel all is off when nothing runs");
    Check(v.cancel_all_tooltip == "Nothing running can be cancelled", "and says why");
    Check(!v.running_empty.empty(), "the running list has its empty text");
    Check(v.finished_heading == "Finished", "the finished heading without a count when all are shown");
    Check(v.finished.size() == 2 && v.finished[0].name == "Prune the project's prepared-data caches",
          "finished tasks newest first");
    Check(v.finished[0].took == "took 11 ms" && v.finished[1].took == "took 12.6 s", "each says how long it took");
    Check(v.finished[0].mark == Mark::Done && !v.finished[0].can_cancel, "a finished task is done, no cancel");
    Check(v.finished[0].clock.size() == 5, "a task finished today shows its time: " + v.finished[0].clock);
    Check(v.hidden_note.empty() && v.finished_empty.empty(), "no hidden note, no empty finished text");
    Check(!v.last_run.shown, "no training run: no run line");
}

void CheckHidden() {
    Input in = DevPc(ParseUtc("2026-11-22T04:12:00Z"));
    in.show_finished = false;
    const View v = Build(in);
    Check(v.finished.empty(), "hidden: no finished rows");
    Check(v.hidden_note == "2 finished tasks hidden. Tick Show finished to see them.",
          "hidden: the window says what is hidden (never blank): " + v.hidden_note);
    in.finished_total = 1;
    in.finished.resize(1);
    Check(Build(in).hidden_note == "1 finished task hidden. Tick Show finished to see it.", "one hidden task");
    in.finished_total = 0;
    in.finished.clear();
    Check(Build(in).hidden_note.empty(), "nothing finished: no hidden note");
}

void CheckRunning() {
    Input in = DevPc(ParseUtc("2026-11-22T04:12:00Z"));
    TaskFacts load;
    load.id = 5;
    load.name = "Loading reviews_20k.csv";
    load.message = "Reading rows into memory";
    load.state = State::Running;
    load.progress = 0.42f;
    load.cancellable = true;
    load.elapsed_ms = 3000;
    TaskFacts wait;
    wait.id = 6;
    wait.name = "Profile reviews_20k";
    wait.state = State::Pending;
    wait.cancellable = false;
    in.running = {load, wait};
    const View v = Build(in);
    Check(v.summary == "2 running, 2 finished", "summary while tasks run: " + v.summary);
    Check(v.cancel_all_enabled, "Cancel all is on when a task can stop");
    Check(v.cancel_all_tooltip == "Cancel every task that can stop (1)", "and counts them: " + v.cancel_all_tooltip);
    Check(v.running_empty.empty(), "no empty text while tasks run");
    Check(v.running.size() == 2 && v.running[0].id == 5, "running rows keep the order they came in");
    Check(v.running[0].show_progress && v.running[0].progress_text == "42%" + kDot + "3.0 s",
          "a running task shows its progress and time: " + v.running[0].progress_text);
    Check(v.running[0].can_cancel, "a cancellable running task can be cancelled");
    Check(v.running[1].mark == Mark::Waiting && v.running[1].took == "waiting to start" && !v.running[1].can_cancel,
          "a pending task waits, no cancel when it cannot stop");

    in.running = {wait};
    Check(Build(in).cancel_all_tooltip == "The running tasks cannot be cancelled",
          "tasks run but none can stop: Cancel all says so");
}

void CheckFailedAndCount() {
    Input in;
    in.now = ParseUtc("2026-11-22T04:12:00Z");
    TaskFacts bad = Finished(9, "Profile reviews_20k", "", 800, in.now - 10);
    bad.state = State::Failed;
    bad.error = "the file is locked";
    TaskFacts stop = Finished(8, "Sorting by rating", "", 2500, in.now - 20);
    stop.state = State::Cancelled;
    in.finished = {bad, stop};
    in.finished_total = 57;
    const View v = Build(in);
    Check(v.finished[0].mark == Mark::Failed && v.finished[0].error == "Failed: the file is locked",
          "a failed task says why");
    Check(v.finished[1].mark == Mark::Cancelled && v.finished[1].took == "cancelled after 2.5 s",
          "a cancelled task says when it stopped");
    Check(v.finished_heading == "Finished" + kDot + "last 2 of 57", "the heading counts the shown of the held: " +
                                                                        v.finished_heading);
    in.finished.clear();
    in.finished_total = 0;
    Check(Build(in).finished_empty == "Nothing has finished yet in this session.", "nothing finished yet");
}

void CheckRunLine() {
    Input in;
    in.now = ParseUtc("2026-11-22T04:12:00Z");
    in.last_run = RunFacts{"cancelled", "arrayfire_cuda", "NVIDIA GeForce GTX 1050 Ti", "2026-11-22T04:05:10Z"};
    const RunLine line = Build(in).last_run;
    Check(line.shown && line.mark == Mark::Cancelled, "a cancelled run shows, marked cancelled");
    Check(line.text == "Last training run: cancelled", "its outcome in words: " + line.text);
    Check(line.detail.rfind("on CUDA, NVIDIA GeForce GTX 1050 Ti" + kDot, 0) == 0,
          "where it ran, then when: " + line.detail);
    Check(line.detail.find("placement") == std::string::npos && line.detail.find("arrayfire_cuda") == std::string::npos,
          "no developer codes in the line");

    in.last_run = RunFacts{"running", "arrayfire_cpu", "", "2026-11-22T04:11:00Z", true};
    const RunLine now = Build(in).last_run;
    Check(now.text == "Training now" && now.detail == "on CPU" && now.mark == Mark::Running,
          "a run in progress: no end time: " + now.detail);
    in.last_run = RunFacts{"running", "", "", "2026-11-22T04:11:00Z", false};
    const RunLine stale = Build(in).last_run;
    Check(stale.text == "Last training run: no end recorded" && stale.mark == Mark::Cancelled,
          "a trace saved as running while nothing trains here: no end recorded (seen on screen: it read Training now)");
    Check(!stale.detail.empty(), "it still says when: " + stale.detail);

    in.last_run = RunFacts{"early_stopped", "arrayfire_cuda", "", ""};
    Check(Build(in).last_run.text == "Last training run: stopped early" && Build(in).last_run.mark == Mark::Done,
          "early stop reads as stopped early, done");
    in.last_run = RunFacts{"failed", "arrayfire_cuda", "", "bad"};
    Check(Build(in).last_run.mark == Mark::Failed && Build(in).last_run.detail == "on CUDA",
          "a failed run; a bad stamp adds no time");
    in.last_run = RunFacts{};
    Check(!Build(in).last_run.shown, "an empty trace shows no line");
}

}  // namespace

int main() {
    CheckDurations();
    CheckParseAndClock();
    CheckIdleWithHistory();
    CheckHidden();
    CheckRunning();
    CheckFailedAndCount();
    CheckRunLine();
    if (failures == 0) std::printf("test_task_list_presentation: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
