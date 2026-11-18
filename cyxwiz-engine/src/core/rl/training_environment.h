#pragma once

// What a trainer (PPO, SAC) needs from N copies of an environment. The host
// adapts its batch (the plugin registry's EnvironmentBatch) into this, so the
// trainers do not depend on the plugin host. GUI-free.

#include "../../plugin/interfaces/i_environment_provider.h"

#include <cstdint>
#include <functional>
#include <string>

namespace cyxwiz::rl {

struct TrainingEnvironment {
    plugin::PluginEnvironmentInfo info;
    int count = 0;
    std::function<bool(uint64_t seed, float* observations, std::string& error)> reset;
    std::function<bool(const float* actions, const plugin::PluginEnvironmentStepBuffers& out, std::string& error)> step;
};

}  // namespace cyxwiz::rl
