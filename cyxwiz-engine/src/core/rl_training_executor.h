#pragma once

#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <string>

namespace cyxwiz {

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

    // Plugin qualified name for MuJoCo Plant node
    std::string plugin_qualified_name;
};

struct RLTrainingMetrics {
    int episode_count = 0;
    int total_timesteps = 0;
    float mean_episode_reward = 0.0f;
    float mean_episode_length = 0.0f;
    float policy_loss = 0.0f;
    float value_loss = 0.0f;
    float entropy = 0.0f;
    float explained_variance = 0.0f;

    bool is_training = false;
    bool is_paused = false;
    std::string status_message;

    std::vector<float> reward_history;
    std::vector<float> length_history;
};

using RLEpisodeCallback = std::function<void(int episode, float reward, float length)>;
using RLUpdateCallback = std::function<void(int timesteps, const RLTrainingMetrics& metrics)>;
using RLCompleteCallback = std::function<void(const RLTrainingMetrics& final_metrics)>;

/**
 * RLTrainingExecutor - Runs RL training loop with viewport integration.
 *
 * Manages the PPO training loop in a background thread, using the MuJoCo plugin
 * for environment simulation. Reports metrics via callbacks for dashboard display.
 */
class RLTrainingExecutor {
public:
    explicit RLTrainingExecutor(const RLTrainingConfig& config);
    ~RLTrainingExecutor();

    void Start(
        RLEpisodeCallback episode_cb = nullptr,
        RLUpdateCallback update_cb = nullptr,
        RLCompleteCallback complete_cb = nullptr
    );
    void Stop();
    void Pause();
    void Resume();

    bool IsTraining() const { return is_training_.load(); }
    bool IsPaused() const { return is_paused_.load(); }
    RLTrainingMetrics GetMetrics() const;

    void SetViewportUpdateInterval(int episodes) { viewport_update_interval_ = episodes; }

private:
    void TrainingLoop();

    RLTrainingConfig config_;
    std::atomic<bool> is_training_{false};
    std::atomic<bool> is_paused_{false};
    std::atomic<bool> stop_requested_{false};

    std::thread training_thread_;
    mutable std::mutex metrics_mutex_;
    RLTrainingMetrics metrics_;

    RLEpisodeCallback episode_callback_;
    RLUpdateCallback update_callback_;
    RLCompleteCallback complete_callback_;

    int viewport_update_interval_ = 10;
};

} // namespace cyxwiz
