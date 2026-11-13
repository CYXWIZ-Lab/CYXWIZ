#pragma once

// A compiled reinforcement learning graph (TOFIX195 S4.3): Environment ->
// ordinary layers -> Policy Head / Value Head -> PPO. GraphCompiler fills it
// (rl_graph_compile.cpp); the PPO run builds the two networks from it.

#include "graph_compiler.h"
#include <cyxwiz/device.h>
#include "rl/ppo_trainer.h"
#include "../plugin/interfaces/i_environment_provider.h"

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

struct RLTrainingConfiguration {
    // Environment
    std::string environment_key;  // "<provider>:<id>"
    std::map<std::string, std::string> environment_settings;
    int copies = 1;
    plugin::PluginEnvironmentInfo environment;

    // The hidden layers of each network, in order from the observation
    // (the heads' output layers are added when the networks are built)
    std::vector<CompiledLayer> policy_layers;
    std::vector<CompiledLayer> value_layers;
    size_t observation_size = 0;
    size_t policy_outputs = 0;  // logits (discrete) or means (continuous)

    rl::PpoConfig ppo;

    int environment_node_id = -1;
    int policy_head_node_id = -1;
    int value_head_node_id = -1;
    int ppo_node_id = -1;
};

// How the compiler finds an environment by key, as the Environment node's
// settings make it (its observation size may follow them). The Engine sets the
// plugin environment registry's lookup; unset, no environment is known. False
// with an empty error: no such environment; with an error: the settings are
// not valid (the plugin's reason).
using RLEnvironmentLookup =
    std::function<bool(const std::string& key, const std::map<std::string, std::string>& settings,
                       plugin::PluginEnvironmentInfo& info, std::string& error)>;
void SetRLEnvironmentLookup(RLEnvironmentLookup lookup);

// True for the node types of an RL graph
bool IsRLGraphNode(gui::NodeType type);

// Whether the device's neural provider would run both networks as fused
// small-network kernels (Dense layers of at most 256 units with the fused
// activations, one LeakyReLU slope and one ELU alpha per network, the
// minibatch within the backward's rows): asked of the provider registry's
// capability checks; no kernel runs. The first use of the registry loads the
// provider modules (as supervised Compile's placement does). A prediction
// for Compile; the run's compute path says what it did.
bool FusedNetworksExpected(const RLTrainingConfiguration& config, DeviceType platform, int device_id);

// The units of a network's layers that have them (its hidden widths)
std::vector<int> HiddenUnits(const std::vector<CompiledLayer>& layers);

}  // namespace cyxwiz
