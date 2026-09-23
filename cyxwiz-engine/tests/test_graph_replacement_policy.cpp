#include "gui/graph_replacement_policy.h"
#include <iostream>
#include <stdexcept>

void Check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main() {
    try {
        gui::GraphReplacementActivity state;
        auto allowed = [&] { return gui::GraphReplacementBlockReason(state).empty(); };
        Check(allowed(), "idle editor accepts replacement");
        // Pause and stop-request do not clear the owner's active state.
        state.training = true;
        for (const char* phase : {"running", "paused", "stop requested", "worker cleanup"})
            Check(!allowed(), phase);
        state.testing = true;
        state.training = false;
        Check(!allowed(), "training ending must not unlock an active test");
        state.testing = false;
        state.preparing_test = true;
        Check(!allowed(), "queued preparation owns the graph before evaluation starts");
        state.testing = true;
        state.preparing_test = false;
        Check(!allowed(), "preparation handoff must retain evaluation protection");
        state.testing = false;
        Check(allowed(), "completed test releases replacement");
        state.simulation = state.pipeline = state.reinforcement_learning = true;
        state.simulation = false;
        Check(!allowed(), "pipeline remains active after simulation stops");
        state.pipeline = false;
        Check(!allowed(), "RL still owns the canvas");
        state.reinforcement_learning = false;
        Check(allowed(), "all owners idle releases replacement");
        std::cout << "PASS: graph replacement admission across overlapping activity and lifecycle transitions\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
