#pragma once
#include <string_view>

namespace gui {
// UI-thread admission policy. A stop request is not idle: owners must finish
// their cleanup before clearing the corresponding activity flag.
struct GraphReplacementActivity {
    bool training = false;
    bool testing = false;
    bool preparing_test = false;
    bool simulation = false;
    bool pipeline = false;
    bool reinforcement_learning = false;
};
inline std::string_view GraphReplacementBlockReason(const GraphReplacementActivity& state) {
    if (state.training) return "training is active (including pause or cleanup)";
    if (state.testing) return "testing is active";
    if (state.preparing_test) return "Run Test preparation is active";
    if (state.simulation) return "simulation is active";
    if (state.pipeline) return "a data pipeline is active";
    if (state.reinforcement_learning) return "reinforcement learning is active";
    return {};
}
} // namespace gui
