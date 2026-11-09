#pragma once

// A compiled reinforcement learning graph (TOFIX195 S4.3): Environment ->
// ordinary layers -> Policy Head / Value Head -> PPO. GraphCompiler fills it
// (rl_graph_compile.cpp); the PPO run builds the two networks from it.

#include "graph_compiler.h"
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

// How the compiler finds an environment by key. The Engine sets the plugin
// environment registry's lookup; unset, no environment is known.
using RLEnvironmentLookup = std::function<bool(const std::string& key, plugin::PluginEnvironmentInfo& info)>;
void SetRLEnvironmentLookup(RLEnvironmentLookup lookup);

// True for the node types of an RL graph
bool IsRLGraphNode(gui::NodeType type);

}  // namespace cyxwiz
