#pragma once

// A compiled reinforcement learning graph (TOFIX195 S4.3, SAC, TD3 and DQN
// TOFIX216): Environment -> ordinary layers -> Policy Head / Value Head -> PPO,
// Policy Head / Q Head -> SAC or TD3, or Q Head -> DQN. GraphCompiler fills it
// (rl_graph_compile.cpp); the run builds the networks from it.

#include "graph_compiler.h"
#include "rl_device_presentation.h"
#include <cyxwiz/device.h>
#include "rl/ppo_trainer.h"
#include "rl/sac_config.h"
#include "rl/td3_config.h"
#include "rl/dqn_config.h"
#include "../plugin/interfaces/i_environment_provider.h"

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

// The learner node of an RL graph
enum class RLAlgorithm { PPO, SAC, TD3, DQN };

// The learner's name, as its node is called ("PPO", "SAC", "TD3", "DQN")
const char* RLAlgorithmName(RLAlgorithm algorithm);
// Whether the learner trains a policy beside twin Q networks on
// [observation | action] from a Q Head (SAC, TD3)
bool UsesTwinQ(RLAlgorithm algorithm);
// Whether the learner learns from a replay buffer and reports by environment
// steps (SAC, TD3, DQN) rather than by rollouts (PPO)
bool IsOffPolicy(RLAlgorithm algorithm);

struct RLTrainingConfiguration {
    RLAlgorithm algorithm = RLAlgorithm::PPO;

    // Environment
    std::string environment_key;  // "<provider>:<id>"
    std::map<std::string, std::string> environment_settings;
    int copies = 1;
    plugin::PluginEnvironmentInfo environment;

    // The hidden layers of each network, in order from the observation
    // (the heads' output layers are added when the networks are built)
    std::vector<CompiledLayer> policy_layers;
    std::vector<CompiledLayer> value_layers;  // PPO
    // SAC, TD3: the Q branch's layers, from the observation (the action joins
    // the first layer's input); the trainer builds the twin Q networks and
    // their targets from it. DQN: its Q network's layers, from the observation
    // alone (one Q value per choice of action)
    std::vector<CompiledLayer> q_layers;
    size_t observation_size = 0;
    size_t policy_outputs = 0;  // logits (discrete) or means (continuous); DQN: its choices

    rl::PpoConfig ppo;
    rl::SacConfig sac;
    rl::Td3Config td3;
    rl::DqnConfig dqn;

    int environment_node_id = -1;
    int policy_head_node_id = -1;
    int value_head_node_id = -1;
    int q_head_node_id = -1;
    int learner_node_id = -1;  // the PPO, SAC, TD3 or DQN node
};

// The learner's seed, whichever learner it is
uint64_t RLSeed(const RLTrainingConfiguration& config);

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
// True for the learner node types (PPO, SAC, TD3, DQN)
bool IsRLLearnerNode(gui::NodeType type);
// The learner a learner node trains with (PPO for any other node type)
RLAlgorithm RLAlgorithmOf(gui::NodeType type);
// That learner's name ("PPO", "SAC", "TD3", "DQN")
const char* RLLearnerName(gui::NodeType type);

// Whether the device's neural provider would run the networks (PPO's policy
// and value; SAC's and TD3's actor and Q, the twin and targets alike; DQN's Q
// network and its target) as fused
// small-network kernels (Dense layers of at most 512 units with the fused
// activations, one LeakyReLU slope and one ELU alpha per network, the
// minibatch within the backward's rows): asked of the provider registry's
// capability checks; no kernel runs. The first use of the registry loads the
// provider modules (as supervised Compile's placement does). When they would
// not, the forecast says which cause it is: a provider that is missing (not
// built, not installed, another ABI, turned off) with what to do, or another
// (the provider declines these networks, the networks are not a fusable
// chain, no provider is offered for them on that device family). A
// prediction for Compile; the run's compute path says what it did.
rl_device::NetworkForecast ForecastRLNetworks(const RLTrainingConfiguration& config, DeviceType platform,
                                              int device_id);

// The units of a network's layers that have them (its hidden widths)
std::vector<int> HiddenUnits(const std::vector<CompiledLayer>& layers);

// The hidden widths of the learner's two networks, as the device notes name
// them: PPO's policy and value, SAC's and TD3's actor and Q, DQN's Q network
// and its target (the same chain)
std::pair<std::vector<int>, std::vector<int>> RLNetworkUnits(const RLTrainingConfiguration& config);

}  // namespace cyxwiz
