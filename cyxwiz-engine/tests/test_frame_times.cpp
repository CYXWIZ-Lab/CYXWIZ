// The frame-time bookkeeping behind the overlay and Monitor > Frames.

#include "core/frame_pace.h"
#include "core/frame_times.h"

#include <cmath>
#include <iostream>
#include <string>

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::cerr << "FAILED: " << what << "\n";
    }
}

bool Near(double a, double b) { return std::fabs(a - b) < 1e-9; }

using cyxwiz::FrameTimes;

void Frame(FrameTimes& times, double build, std::initializer_list<std::pair<const char*, double>> windows) {
    times.BeginFrame();
    for (const auto& [name, ms] : windows) times.AddWindow(name, ms);
    times.EndFrame(build);
}

}  // namespace

int main() {
    FrameTimes times;
    Check(times.Summarize().frames == 0 && times.Summarize().windows.empty() && times.History().empty(), "new: nothing");

    Frame(times, 2.0, {{"Studio", 1.0}, {"Console", 0.5}});
    auto s = times.Summarize();
    Check(s.frames == 1 && Near(s.build_now, 2.0) && Near(s.build_average, 2.0) && Near(s.build_worst, 2.0), "one frame");
    Check(s.windows.size() == 2 && s.windows[0].name == "Studio" && Near(s.windows[0].now, 1.0), "costliest first");
    Check(Near(s.windows[0].average, 1.0), "a window's first frame is its average, not a tenth of it");
    Check(Near(s.windows_total, 1.5), "the total is this frame's sum");

    // The second frame: Console is not drawn, Studio costs more, a window is timed twice.
    Frame(times, 4.0, {{"Studio", 2.0}, {"Data", 0.25}, {"Data", 0.25}});
    s = times.Summarize();
    Check(s.frames == 2 && Near(s.build_now, 4.0) && Near(s.build_average, 3.0) && Near(s.build_worst, 4.0),
          "now, average and worst of two frames");
    Check(s.windows.size() == 2, "only the windows drawn this frame are listed (Console is not)");
    Check(s.windows[0].name == "Studio" && Near(s.windows[0].now, 2.0), "this frame's time is as measured, not smoothed");
    Check(Near(s.windows[0].average, 1.0 + FrameTimes::kSmoothing * (2.0 - 1.0)), "the average moves a tenth of the way");
    Check(s.windows[1].name == "Data" && Near(s.windows[1].now, 0.5), "two timings of one window add up");
    Check(Near(s.windows_total, 2.5), "total of the second frame");

    // Console comes back: its average was kept, not decayed while hidden.
    Frame(times, 1.0, {{"Console", 0.5}});
    s = times.Summarize();
    Check(s.windows.size() == 1 && s.windows[0].name == "Console" && Near(s.windows[0].average, 0.5),
          "a hidden window's average is kept");

    // The history holds the last kHistory frames, oldest first.
    for (int i = 0; i < 400; ++i) Frame(times, static_cast<double>(i), {});
    s = times.Summarize();
    const auto history = times.History();
    Check(s.frames == FrameTimes::kHistory && history.size() == FrameTimes::kHistory, "the history is capped");
    Check(history.front() == 220.0f && history.back() == 399.0f, "oldest first: 220 .. 399");
    Check(Near(s.build_worst, 399.0) && Near(s.build_now, 399.0) && Near(s.build_average, (220.0 + 399.0) / 2.0),
          "figures cover the kept frames only");

    times.Reset();
    s = times.Summarize();
    Check(s.frames == 0 && s.windows.empty() && Near(s.build_worst, 0.0) && times.History().empty(),
          "Reset forgets the old frames (the overlay switched on again)");

    // The frame pace: one decision from four facts.
    using cyxwiz::FramePace;
    using cyxwiz::PaceOf;
    Check(PaceOf(false, false, false, true) == FramePace::Full, "in front and active: full speed");
    Check(PaceOf(false, true, false, true) == FramePace::Idle, "no input for a second: idle");
    Check(PaceOf(false, true, true, true) == FramePace::Full, "training keeps a focused Engine at full speed");
    Check(PaceOf(false, false, false, false) == FramePace::NotInFront, "another window in front: 30 a second");
    Check(PaceOf(false, true, true, false) == FramePace::NotInFront, "training behind another window: 30 a second");
    Check(PaceOf(false, true, false, false) == FramePace::Idle, "idle wins over not in front (the longer wait)");
    Check(PaceOf(true, false, true, true) == FramePace::Minimised, "minimised draws nothing, whatever else is true");
    Check(std::string(cyxwiz::PaceText(FramePace::Idle)) == "10 frames a second (idle: no input for a second)" &&
              std::string(cyxwiz::PaceText(FramePace::Minimised)) == "Not drawing (the window is minimised)",
          "the pace in words");

    if (g_failures != 0) {
        std::cerr << g_failures << " of " << g_checks << " checks failed\n";
        return 1;
    }
    std::cout << "frame times: " << g_checks << " checks passed\n";
    return 0;
}
