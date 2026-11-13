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

    // Zoom: the same two threads, the axis narrowed to 250..400 ms (2 px a ms).
    const TimelineWindow window{250.0, 400.0};
    const TimelineLayout z = LayoutTimeline(timed, 303.0f, 3.0f, &window);
    Check(z.timed && z.zoomed && Near(z.start_ms, 250.0) && Near(z.end_ms, 400.0), "a window narrows the axis");
    Check(Near(z.run_start_ms, 100.0) && Near(z.run_end_ms, 400.0), "the run's whole span is kept beside the window");
    Check(!z.lanes[0][0].visible, "an event that ended before the window is not drawn");
    Check(z.lanes[0][1].visible && Near(z.lanes[0][1].x0, 200.0) && Near(z.lanes[0][1].x1, 300.0),
          "an event inside the window is placed on its scale");
    Check(z.lanes[1][0].visible && Near(z.lanes[1][0].x0, 0.0) && Near(z.lanes[1][0].x1, 300.0),
          "an event that began before the window starts at its edge");
    const TimelineWindow wider{0.0, 9000.0};
    const TimelineLayout w = LayoutTimeline(timed, 303.0f, 3.0f, &wider);
    Check(!w.zoomed && Near(w.start_ms, 100.0) && Near(w.end_ms, 400.0), "a window wider than the run is the whole run");
    const TimelineWindow empty{300.0, 300.0};
    Check(!LayoutTimeline(timed, 303.0f, 3.0f, &empty).zoomed, "an empty window is ignored");

    // Clusters, shaped like a real CPU run: a busy thread with 40 one-millisecond
    // steps every 2 ms, and a slow thread whose one event spans 4 s.
    std::vector<TimelineEvent> busy;
    for (int i = 0; i < 40; ++i) busy.push_back({2.0 * i + 1.0, 1.0});
    const std::vector<std::vector<TimelineEvent>> run = {busy, {{4000.0, 4000.0}}};
    const auto whole = ClusterTimeline(LayoutTimeline(run, 600.0f));
    Check(whole[0].size() == 1 && whole[0][0].events.size() == 40, "at the whole run's scale the busy thread is one cluster of 40");
    Check(whole[1].size() == 1 && whole[1][0].events.size() == 1, "the slow thread's event stays one bar");
    const TimelineWindow first80{0.0, 80.0};
    const auto close = ClusterTimeline(LayoutTimeline(run, 600.0f, 3.0f, &first80));
    Check(close[0].size() == 40, "zoomed into 80 ms the 40 steps come apart (7.5 px each)");
    // A long event with steps inside it (a 60 ms validation pass on the same
    // thread as 30 one-millisecond steps): it never swallows them.
    std::vector<TimelineEvent> nested = {{60.0, 60.0}};
    for (int i = 0; i < 30; ++i) nested.push_back({2.0 * i + 1.0, 1.0});
    const auto nested_whole = ClusterTimeline(LayoutTimeline({nested}, 303.0f));
    Check(!nested_whole[0].empty() && nested_whole[0][0].span && nested_whole[0][0].events.size() == 1 &&
              nested_whole[0][0].events[0] == 0,
          "the long event is a span of its own, listed first (drawn under the steps)");
    size_t nested_steps = 0;
    for (size_t c = 1; c < nested_whole[0].size(); ++c) {
        Check(!nested_whole[0][c].span, "only the long event is a span");
        nested_steps += nested_whole[0][c].events.size();
    }
    Check(nested_steps == 30 && nested_whole[0].size() == 31, "its 30 steps (5 px each, 10 apart) stay 30 bars");
    const TimelineWindow nested_zoom{0.0, 10.0};
    const auto nested_close = ClusterTimeline(LayoutTimeline({nested}, 303.0f, 3.0f, &nested_zoom));
    Check(nested_close[0][0].span && nested_close[0].size() == 1 + 6, "zoomed in, the span stays under the 6 steps in view");

    // Back to back (Forward, then Backward, then the optimizer): bars that
    // only touch stay apart when each is wide enough to draw.
    std::vector<TimelineEvent> steps;
    for (int i = 0; i < 30; ++i) steps.push_back({1.0 * (i + 1), 1.0});
    const auto touching = ClusterTimeline(LayoutTimeline({steps}, 303.0f));
    Check(touching[0].size() == 30, "back-to-back events 10 px wide stay 30 bars");
    const auto squeezed = ClusterTimeline(LayoutTimeline({steps}, 63.0f));
    Check(squeezed[0].size() < 30, "the same events at 2 px each overlap the minimum width and merge");
    const TimelineWindow some{20.0, 40.0};
    const auto part = ClusterTimeline(LayoutTimeline(run, 600.0f, 3.0f, &some));
    size_t shown = 0;
    for (const auto& c : part[0]) shown += c.events.size();
    Check(shown == 11, "only the steps inside the window are clustered or drawn");
    bool ordered = true;
    for (const auto& c : part[0])
        for (size_t k = 1; k < c.events.size(); ++k) ordered = ordered && c.events[k] > c.events[k - 1];
    Check(ordered && part[0].front().x0 <= part[0].back().x0, "clusters run left to right with their events in order");

    // Axis marks precise enough to read differently across the span.
    Check(TimelineAxisText(1428.1, 580.8) == "1.43 s" && TimelineAxisText(2008.9, 580.8) == "2.01 s", "a 0.58 s span in hundredths");
    Check(TimelineAxisText(1652.43, 10.0) == "1652.4 ms", "a few milliseconds in tenths of a millisecond");
    Check(TimelineAxisText(1652.437, 1.0) == "1652.44 ms" && TimelineAxisText(1652.4371, 0.1) == "1652.437 ms",
          "under two milliseconds, finer still");
    Check(TimelineAxisText(1652.43, 60.0) == "1.652 s" && TimelineAxisText(3000.0, 30000.0) == "3.0 s", "wider spans");

    Check(TimelineTimeText(0.0) == "0 ms" && TimelineTimeText(250.4) == "250 ms", "milliseconds");
    Check(TimelineTimeText(1500.0) == "1.5 s" && TimelineTimeText(125000.0) == "2 min 5 s", "seconds and minutes");

    if (g_failures != 0) {
        std::cerr << g_failures << " of " << g_checks << " checks failed\n";
        return 1;
    }
    std::cout << "runtime timeline layout: " << g_checks << " checks passed\n";
    return 0;
}
