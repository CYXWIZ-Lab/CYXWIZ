// Where the Runtime Timeline draws its events: on a time axis when they carry
// their time, packed in order when they do not.

#include "core/runtime_timeline_layout.h"

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

bool Near(double a, double b, double tolerance = 1e-3) { return std::fabs(a - b) <= tolerance; }

}  // namespace

int main() {
    using namespace cyxwiz;

    // Two threads. The trainer: a 100 ms batch ending at 200 ms, then a 50 ms
    // one ending at 400 ms (a 150 ms gap). The loader: 300 ms ending at 400 ms,
    // overlapping both.
    const std::vector<std::vector<TimelineEvent>> timed = {{{200.0, 100.0}, {400.0, 50.0}}, {{400.0, 300.0}}};
    const TimelineLayout a = LayoutTimeline(timed, 303.0f, 3.0f);
    Check(a.timed, "events with their time are drawn on a time axis");
    Check(Near(a.start_ms, 100.0) && Near(a.end_ms, 400.0), "the axis runs from the first start to the last end");
    Check(a.lanes.size() == 2 && a.lanes[0].size() == 2 && a.lanes[1].size() == 1, "one bar per event");
    // 300 px for 300 ms: one pixel a millisecond.
    Check(Near(a.lanes[0][0].x0, 0.0) && Near(a.lanes[0][0].x1, 100.0), "a 100 ms event is 100 px wide at 1 px/ms");
    Check(Near(a.lanes[0][1].x0, 250.0) && Near(a.lanes[0][1].x1, 300.0), "the gap before the second event is kept");
    Check(Near(a.lanes[1][0].x0, 0.0) && Near(a.lanes[1][0].x1, 300.0), "the other thread's event overlaps both, as it did");
    Check(Near(a.width, 303.0), "the drawing fills the width");

    // A very short event can still be pointed at.
    const TimelineLayout tiny = LayoutTimeline({{{1000.0, 0.01}, {2000.0, 500.0}}}, 400.0f, 3.0f);
    Check(tiny.timed && tiny.lanes[0][0].x1 - tiny.lanes[0][0].x0 >= 3.0f, "a bar is never narrower than the minimum");

    // One event without its time (an old run): everything is packed in order.
    const TimelineLayout old = LayoutTimeline({{{-1.0, 10.0}, {-1.0, 20.0}, {-1.0, 0.0}}}, 400.0f);
    Check(!old.timed, "a run without event times is not drawn on a time axis");
    Check(old.lanes[0].size() == 3 && old.lanes[0][0].x0 == 0.0f, "packed from the left");
    Check(old.lanes[0][1].x0 > old.lanes[0][0].x1 && old.lanes[0][2].x0 > old.lanes[0][1].x1, "in order, with a gap");
    Check(old.lanes[0][1].x1 - old.lanes[0][1].x0 > old.lanes[0][0].x1 - old.lanes[0][0].x0, "wider for longer");
    Check(Near(old.lanes[0][2].x1 - old.lanes[0][2].x0, 7.0), "an event with no duration has the narrow width");
    const TimelineLayout mixed = LayoutTimeline({{{200.0, 100.0}}, {{-1.0, 5.0}}}, 400.0f);
    Check(!mixed.timed, "one event without its time: the whole view is packed (no half axis)");

    // Nothing, and a single instant.
    Check(!LayoutTimeline({}, 400.0f).timed && LayoutTimeline({}, 400.0f).lanes.empty(), "no lanes");
    Check(!LayoutTimeline({{{500.0, 0.0}}}, 400.0f).timed, "one instant has no span: packed");

    Check(TimelineTimeText(0.0) == "0 ms" && TimelineTimeText(250.4) == "250 ms", "milliseconds");
    Check(TimelineTimeText(1500.0) == "1.5 s" && TimelineTimeText(125000.0) == "2 min 5 s", "seconds and minutes");

    if (g_failures != 0) {
        std::cerr << g_failures << " of " << g_checks << " checks failed\n";
        return 1;
    }
    std::cout << "runtime timeline layout: " << g_checks << " checks passed\n";
    return 0;
}
