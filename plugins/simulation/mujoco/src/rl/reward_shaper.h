#pragma once

#include <vector>
#include <string>
#include <map>
#include <cmath>

namespace cyxwiz::plugin::rl {

struct RewardConfig {
    float alive_bonus = 1.0f;
    float ctrl_cost_weight = 0.1f;
    bool velocity_reward = true;
    float height_penalty_threshold = 0.0f;  // 0 = disabled
    float height_penalty_value = 10.0f;

    static RewardConfig FromParameters(const std::map<std::string, std::string>& params);
};

struct EnvState {
    std::vector<float> qpos;
    std::vector<float> qvel;
    std::vector<float> ctrl;       // last control input
    std::vector<float> sensor;
    float sim_time = 0.0f;
};

/**
 * RewardShaper - Computes shaped reward from environment state.
 *
 * Supports configurable reward components:
 * - Alive bonus (constant per timestep)
 * - Control cost penalty (L2 norm of control)
 * - Forward velocity reward (qvel[0])
 * - Height penalty (if qpos[2] drops below threshold)
 */
class RewardShaper {
public:
    explicit RewardShaper(const RewardConfig& config = {});

    void SetConfig(const RewardConfig& config) { config_ = config; }
    const RewardConfig& GetConfig() const { return config_; }

    // Compute reward from environment state
    float ComputeReward(const EnvState& state) const;

    // Reset episode tracking (for potential future extensions like cumulative tracking)
    void Reset();

private:
    RewardConfig config_;
};

} // namespace cyxwiz::plugin::rl
