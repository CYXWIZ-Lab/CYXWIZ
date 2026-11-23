#pragma once

// What the RL nodes show in Properties (TOFIX195 S4.3): the environment list,
// an environment's facts, what a head adds, the tuned PPO, SAC, TD3 and DQN settings
// for an environment, and the Environment node's plugin settings. Pure data in and
// out (no ImGui), so the Properties panel and its tests agree.

#include "graph_model.h"
#include "../plugin/interfaces/i_environment_provider.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::rl_graph {

inline constexpr const char* kSettingPrefix = "setting.";  // Environment node: plugin settings

struct Fact {
    std::string label;
    std::string value;
};

struct EnvironmentChoice {
    std::string key;    // "<provider>:<id>"
    std::string label;  // "CartPole-v1 · CyxWiz (built in)"
};

struct ListedEnvironment {
    std::string key;
    std::string provider_name;  // the plugin's name, or "CyxWiz (built in)"
    plugin::PluginEnvironmentInfo info;
};

// Sorted by id; an unknown key (plugin not loaded) is added as "<key> · not loaded"
std::vector<EnvironmentChoice> EnvironmentChoices(const std::vector<ListedEnvironment>& environments,
                                                  const std::string& current_key);

// Observation, actions, episode limit, one step
std::vector<Fact> EnvironmentFacts(const plugin::PluginEnvironmentInfo& info);

// The learner node (PPO, SAC, TD3, DQN) a node's output feeds; Unknown when none
gui::NodeType LearnerFed(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links, int node_id);

// What the head adds. features: the width reaching it, 0 when not known yet.
// learner: the node it feeds; under SAC a tanh-squashed Gaussian whose log
// standard deviation comes from the state, under TD3 a deterministic tanh
// action (PPO's otherwise)
std::vector<Fact> PolicyHeadFacts(const plugin::PluginEnvironmentInfo& info, size_t features,
                                  gui::NodeType learner = gui::NodeType::PPO);
std::vector<Fact> ValueHeadFacts(size_t features);
// actions: the environment's action size (DQN: its choices), 0 when not known
// yet; learner: the node it feeds (TD3 adds that the policy learns from Q
// network 1; DQN's head gives one value per choice, from the observation alone)
std::vector<Fact> QHeadFacts(size_t features, size_t actions, gui::NodeType learner = gui::NodeType::SAC);
// TD3's exploration, as "Gaussian noise, std 0.1 of the action range, after
// the random steps"
std::string ExplorationNoiseText(bool gaussian, const std::string& std_text);
// DQN's epsilon schedule, as "random 100 % of the time at first, down to 4 %
// after 8,000 steps (16 %)"; total_timesteps 0 when not known
std::string ExplorationScheduleText(float initial_eps, float final_eps, float fraction, int64_t total_timesteps);

// The replay buffer's size on the training device, as "about 168 MB"
std::string ReplayMemoryText(uint64_t bytes);
// ... and against the training device's memory (device_bytes 0: not known):
// "about 168 MB, 4 % of the NVIDIA GeForce GTX 1050 Ti's 4.3 GB"; the advice
// when it leaves the networks little room or does not fit at all
struct ReplayFit {
    std::string text;
    std::string advice;  // empty: it fits with room to spare
};
ReplayFit ReplayFitText(uint64_t bytes, const std::string& device_name, uint64_t device_bytes);
// SAC's automatic target entropy, as "-6 (minus the 6 actions)"
std::string AutoTargetEntropyText(size_t actions);

// The environment's plugin settings as the node stores them ("setting.<key>"),
// empty values left out
std::map<std::string, std::string> StoredSettings(const std::map<std::string, std::string>& parameters);

// The Environment node a node is reached from, following its inputs back; null if none
const gui::MLNode* UpstreamEnvironment(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links,
                                       int node_id);
// The units of the nearest Dense feeding the node (through activations), 0 if none
size_t UpstreamUnits(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links, int node_id);

// Tuned settings for an environment the Engine knows, else nullopt.
struct TunedPreset {
    std::string source;                          // where the values come from
    std::map<std::string, std::string> learner;  // the PPO, SAC, TD3 or DQN node's parameters
    std::map<std::string, std::string> environment;  // Environment node parameters
    std::map<std::string, std::string> policy_head;  // Policy Head parameters
    std::string layers_note;                     // the layers the source used, when they differ from 64-64 tanh
};
std::optional<TunedPreset> TunedPpoSettings(const std::string& environment_id);
std::optional<TunedPreset> TunedSacSettings(const std::string& environment_id);
std::optional<TunedPreset> TunedTd3Settings(const std::string& environment_id);
std::optional<TunedPreset> TunedDqnSettings(const std::string& environment_id);

}  // namespace cyxwiz::rl_graph
