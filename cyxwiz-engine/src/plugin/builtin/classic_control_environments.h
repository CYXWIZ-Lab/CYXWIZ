#pragma once

// Environments the Engine offers itself, without any plugin (TOFIX195):
// Gymnasium's classic-control CartPole-v1, ported exactly (dynamics, reward,
// endings, start noise). It tests the learner and the discrete policy head
// with nothing installed. PluginEnvironmentRegistry registers it under the
// provider id below, like a plugin's environments.

#include "../interfaces/i_environment_provider.h"

namespace cyxwiz::plugin {

inline constexpr const char* kBuiltinEnvironmentProviderId = "cyxwiz.builtin";

IEnvironmentProvider& ClassicControlEnvironments();

}  // namespace cyxwiz::plugin
