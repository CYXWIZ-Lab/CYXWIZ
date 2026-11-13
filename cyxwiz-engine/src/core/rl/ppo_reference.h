#pragma once

// Stable-Baselines3 reference curves for a PPO run (TOFIX195 S4.3, R9): the
// recorded SB3 runs (generate_ppo_sb3_curves.py, ppo_sb3_curves.json) on the
// same environment with the same copies and settings, as the mean episode
// return by environment step. A run with other settings has no reference:
// a curve of other settings would mislead.

#include "ppo_trainer.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::rl {

struct PpoReference {
    std::string label;  // "SB3 2.7.1, 3 seeds"
    std::vector<std::vector<std::pair<double, double>>> seeds;  // (steps, mean return of the last 100)

    // The seeds' mean at this step (each seed interpolated); nullopt past the shortest
    std::optional<double> At(double steps) const;
};

// The reference for this environment, copy count and settings in the
// recorded file's text, or nullopt
std::optional<PpoReference> FindPpoReference(const std::string& json_text, const std::string& environment_id,
                                             int copies, const PpoConfig& config);

// The recorded file in the Engine's resources/rl_reference, empty if none
std::string ReadPpoReferenceFile();

}  // namespace cyxwiz::rl
