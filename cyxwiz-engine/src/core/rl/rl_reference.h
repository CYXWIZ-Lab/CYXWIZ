#pragma once

// Stable-Baselines3 reference curves for an RL run (TOFIX195 S4.3, R9; SAC
// TOFIX216 S5): the recorded SB3 runs (generate_ppo_sb3_curves.py ->
// ppo_sb3_curves.json, generate_sac_sb3_curves.py -> sac_sb3_curves.json) on
// the same environment with the same copies and settings, as the mean episode
// return by environment step. A run with other settings has no reference: a
// curve of other settings would mislead. The recorded runs use each
// environment's own defaults (no grounds, pushes or throws).

#include "ppo_trainer.h"
#include "sac_config.h"

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::rl {

struct RLReference {
    std::string label;  // "SB3 2.7.1, 3 seeds"
    std::vector<std::vector<std::pair<double, double>>> seeds;  // (steps, mean return of the last 100)

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

// Whether `settings` (the Environment node's) leave every setting the
// environment declares at its default, as the recorded runs do: a value equal
// to the default (as a number when both are numbers) or not given. A key the
// environment does not declare counts as changed.
bool DefaultEnvironmentSettings(const plugin::PluginEnvironmentInfo& environment,
                                const std::map<std::string, std::string>& settings);

// A recorded file ("ppo_sb3_curves.json", "sac_sb3_curves.json") in the
// Engine's resources/rl_reference, empty if none
std::string ReadReferenceFile(const std::string& name);

}  // namespace cyxwiz::rl
