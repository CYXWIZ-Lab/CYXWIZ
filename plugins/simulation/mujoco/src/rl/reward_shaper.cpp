#include "reward_shaper.h"
#include <numeric>

namespace cyxwiz::plugin::rl {

RewardConfig RewardConfig::FromParameters(const std::map<std::string, std::string>& params) {
    RewardConfig cfg;
    auto get = [&](const std::string& key, const std::string& def) -> std::string {
        auto it = params.find(key);
        return it != params.end() ? it->second : def;
    };

    try { cfg.alive_bonus = std::stof(get("alive_bonus", "1.0")); } catch (...) {}
    try { cfg.ctrl_cost_weight = std::stof(get("ctrl_cost_weight", "0.1")); } catch (...) {}
    cfg.velocity_reward = (get("velocity_reward", "true") == "true");
    try { cfg.height_penalty_threshold = std::stof(get("height_penalty_threshold", "0.0")); } catch (...) {}
    try { cfg.height_penalty_value = std::stof(get("height_penalty_value", "10.0")); } catch (...) {}

    return cfg;
}

RewardShaper::RewardShaper(const RewardConfig& config)
    : config_(config) {}

float RewardShaper::ComputeReward(const EnvState& state) const {
    float reward = 0.0f;

    // Alive bonus
    reward += config_.alive_bonus;

    // Control cost penalty: -w * sum(u^2)
    if (!state.ctrl.empty() && config_.ctrl_cost_weight > 0.0f) {
        float ctrl_cost = 0.0f;
        for (float u : state.ctrl) {
            ctrl_cost += u * u;
        }
        reward -= config_.ctrl_cost_weight * ctrl_cost;
    }

    // Forward velocity reward (qvel[0] is typically forward velocity)
    if (config_.velocity_reward && !state.qvel.empty()) {
        reward += state.qvel[0];
    }

    // Height penalty (qpos[2] is typically torso height for locomotion)
    if (config_.height_penalty_threshold > 0.0f && state.qpos.size() > 2) {
        if (state.qpos[2] < config_.height_penalty_threshold) {
            reward -= config_.height_penalty_value;
        }
    }

    return reward;
}

void RewardShaper::Reset() {
    // Reserved for future episode-level tracking
}

} // namespace cyxwiz::plugin::rl
