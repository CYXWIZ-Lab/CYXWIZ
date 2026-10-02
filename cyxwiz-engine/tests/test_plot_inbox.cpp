// Worker -> UI inboxes (TOFIX134 P0 items 6 and 7): every figure a finished
// script publishes, and every RL metric a script reports, is taken exactly
// once, in order, also when the UI takes late or while a worker publishes.
#include "../src/scripting/plot_inbox.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

using namespace scripting;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

CapturedPlot Figure(const std::string& label) {
    CapturedPlot plot;
    plot.label = label;
    plot.png_data = {1, 2, 3};
    return plot;
}
}  // namespace

int main() {
    PlotInbox inbox;
    Check(inbox.Take().empty(), "nothing published, nothing taken");

    // Two runs finish before the UI looks (the window was hidden): both kept.
    inbox.Publish({Figure("run1 a"), Figure("run1 b")});
    inbox.Publish({Figure("run2")});
    inbox.Publish({});
    auto taken = inbox.Take();
    Check(taken.size() == 3 && taken[0].label == "run1 a" && taken[2].label == "run2",
          "figures of runs finished while hidden all arrive, in order");
    Check(taken[0].png_data.size() == 3, "image bytes kept");
    Check(inbox.Take().empty(), "each figure is taken once");

    // A worker publishes while the UI takes every "frame".
    constexpr int kRuns = 2000;
    std::thread worker([&] {
        for (int i = 0; i < kRuns; ++i) inbox.Publish({Figure(std::to_string(i))});
    });
    int received = 0, next = 0;
    bool ordered = true;
    while (received < kRuns) {
        for (const auto& plot : inbox.Take()) {
            ordered = ordered && plot.label == std::to_string(next++);
            ++received;
        }
    }
    worker.join();
    Check(ordered && inbox.Take().empty(), "no figure lost or doubled across threads");
    std::cout << "plot inbox: hidden-window runs kept, taken once, thread safe. OK\n";
    return 0;
}
