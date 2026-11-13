#pragma once

// What the RL nodes show in Properties (TOFIX195 S4.3): the environment list,
// an environment's facts, what a head adds, the tuned PPO settings for an
// environment, and the Environment node's plugin settings. Pure data in and
// out (no ImGui), so the Properties panel and its tests agree.

#include "graph_model.h"
#include "../plugin/interfaces/i_environment_provider.h"

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

// What the head adds. features: the width reaching it, 0 when not known yet.
std::vector<Fact> PolicyHeadFacts(const plugin::PluginEnvironmentInfo& info, size_t features);
std::vector<Fact> ValueHeadFacts(size_t features);

// The environment's plugin settings as the node stores them ("setting.<key>"),
// empty values left out
std::map<std::string, std::string> StoredSettings(const std::map<std::string, std::string>& parameters);

// The Environment node a node is reached from, following its inputs back; null if none
const gui::MLNode* UpstreamEnvironment(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links,
                                       int node_id);
// The units of the nearest Dense feeding the node (through activations), 0 if none
size_t UpstreamUnits(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links, int node_id);

// Tuned settings for an environment the Engine knows, else nullopt.
struct PpoPreset {
    std::string source;                          // where the values come from
    std::map<std::string, std::string> ppo;      // PPO node parameters
    std::map<std::string, std::string> environment;  // Environment node parameters
    std::map<std::string, std::string> policy_head;  // Policy Head parameters
    std::string layers_note;                     // the layers the source used, when they differ from 64-64 tanh
};
std::optional<PpoPreset> TunedPpoSettings(const std::string& environment_id);

}  // namespace cyxwiz::rl_graph
