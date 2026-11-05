#pragma once

#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

// What Train RL reads from the RL Training and plugin nodes.
struct RLTrainingConfig {
    std::string env_mjcf_path;
    int total_timesteps = 1000000;
    int max_episode_steps = 1000;

    // PPO hyperparameters
    float learning_rate = 3e-4f;
    float gamma = 0.99f;
    float gae_lambda = 0.95f;
    float clip_range = 0.2f;
    int n_steps = 2048;
    int batch_size = 64;
    int n_epochs = 10;
    std::vector<int> hidden_sizes = {64, 64};
    float entropy_coeff = 0.01f;
    float value_coeff = 0.5f;

    // Reward shaping
    float alive_bonus = 1.0f;
    float ctrl_cost_weight = 0.1f;
    bool velocity_reward = true;

    // Observation filter
    bool include_qpos = true;
    bool include_qvel = true;
    bool include_sensors = false;
    bool normalize_obs = true;
};

class RLScriptGenerator {
public:
    static std::string Generate(
        const RLTrainingConfig& config,
        const std::map<std::string, std::string>& reward_params,
        const std::map<std::string, std::string>& obs_filter_params,
        const std::string& save_path);
};

} // namespace cyxwiz
