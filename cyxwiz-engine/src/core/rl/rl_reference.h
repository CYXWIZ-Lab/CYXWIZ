#pragma once

// Stable-Baselines3 reference curves for an RL run (TOFIX195 S4.3, R9; SAC
// TOFIX216 S5, TD3 U5, DQN V5): the recorded SB3 runs (generate_ppo_sb3_curves.py ->
// ppo_sb3_curves.json, generate_sac_sb3_curves.py -> sac_sb3_curves.json,
// generate_td3_sb3_curves.py -> td3_sb3_curves.json, generate_dqn_sb3_curves.py
// -> dqn_sb3_curves.json) on
// the same environment with the same copies and settings, as the mean episode
// return by environment step. A run with other settings has no reference: a
// curve of other settings would mislead. The recorded runs use each
// environment's own defaults (no grounds, pushes or throws).

#include "ppo_trainer.h"
#include "sac_config.h"
#include "td3_config.h"
#include "dqn_config.h"

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {
struct RLTrainingConfiguration;
}

namespace cyxwiz::rl {

struct RLReference {
    std::string label;  // "SB3 2.7.1, 3 seeds"
    std::vector<std::vector<std::pair<double, double>>> seeds;  // (steps, mean return of the last 100)
    // The recorded networks' hidden widths, each network's (SB3's net_arch: the
    // task's own, else the learner's default as SB3 or RL Zoo ran it)
    std::vector<int> layers;

    // The seeds' mean at this step (each seed interpolated); nullopt past the shortest
    std::optional<double> At(double steps) const;
};

// The PPO reference for this environment, copy count and settings in the
// recorded file's text (ppo_sb3_curves.json), or nullopt
std::optional<RLReference> FindPpoReference(const std::string& json_text, const std::string& environment_id,
                                            int copies, const PpoConfig& config);

// The SAC reference likewise (sac_sb3_curves.json)
std::optional<RLReference> FindSacReference(const std::string& json_text, const std::string& environment_id,
                                            int copies, const SacConfig& config);
// The TD3 reference likewise (td3_sb3_curves.json)
std::optional<RLReference> FindTd3Reference(const std::string& json_text, const std::string& environment_id,
                                            int copies, const Td3Config& config);
// The DQN reference likewise (dqn_sb3_curves.json)
std::optional<RLReference> FindDqnReference(const std::string& json_text, const std::string& environment_id,
                                            int copies, const DqnConfig& config);

// The reference the dashboard draws under a compiled run: its learner's
// recorded file, the same settings and copies, the environment at its
// defaults, and both networks of the recorded hidden widths; nullopt when any
// differs (a curve of other settings or layers would mislead)
std::optional<RLReference> FindRLReference(const RLTrainingConfiguration& config);
// Whether both of the run's networks have the reference's hidden widths
bool SameLayers(const RLReference& reference, const RLTrainingConfiguration& config);

// Whether `settings` (the Environment node's) leave every setting the
// environment declares at its default, as the recorded runs do: a value equal
// to the default (as a number when both are numbers) or not given. A key the
// environment does not declare counts as changed.
bool DefaultEnvironmentSettings(const plugin::PluginEnvironmentInfo& environment,
                                const std::map<std::string, std::string>& settings);

// A recorded file ("ppo_sb3_curves.json", "sac_sb3_curves.json", "td3_sb3_curves.json",
// "dqn_sb3_curves.json") in the
// Engine's resources/rl_reference, empty if none
std::string ReadReferenceFile(const std::string& name);
// The folders ReadReferenceFile looks in, in order: rl_reference under the
// resources beside the executable (an installation, a macOS app bundle) and
// under the source tree's resources next to a build tree
std::vector<std::filesystem::path> ReferenceFileFolders();

}  // namespace cyxwiz::rl
