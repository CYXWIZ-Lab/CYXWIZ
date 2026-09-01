#include "rl_training_executor.h"
#include "../plugin/registries/plugin_node_registry.h"
#include "../plugin/interfaces/i_node_provider.h"
#include <spdlog/spdlog.h>
#include <chrono>

namespace cyxwiz {

RLTrainingExecutor::RLTrainingExecutor(const RLTrainingConfig& config)
    : config_(config) {}

RLTrainingExecutor::~RLTrainingExecutor() {
    Stop();
}

void RLTrainingExecutor::Start(
    RLEpisodeCallback episode_cb,
    RLUpdateCallback update_cb,
    RLCompleteCallback complete_cb)
{
    if (is_training_.load()) return;

    episode_callback_ = std::move(episode_cb);
    update_callback_ = std::move(update_cb);
    complete_callback_ = std::move(complete_cb);

    stop_requested_ = false;
    is_paused_ = false;
    is_training_ = true;

    {
        std::lock_guard lock(metrics_mutex_);
        metrics_ = {};
        metrics_.is_training = true;
        metrics_.status_message = "Starting RL training...";
    }

    training_thread_ = std::thread([this]() { TrainingLoop(); });
}

void RLTrainingExecutor::Stop() {
    if (!is_training_.load()) return;
    stop_requested_ = true;
    is_paused_ = false;  // Unpause so loop can exit

    if (training_thread_.joinable())
        training_thread_.join();

    is_training_ = false;
    {
        std::lock_guard lock(metrics_mutex_);
        metrics_.is_training = false;
        metrics_.status_message = "Training stopped";
    }
}

void RLTrainingExecutor::Pause() {
    is_paused_ = true;
    std::lock_guard lock(metrics_mutex_);
    metrics_.is_paused = true;
    metrics_.status_message = "Training paused";
}

void RLTrainingExecutor::Resume() {
    is_paused_ = false;
    std::lock_guard lock(metrics_mutex_);
    metrics_.is_paused = false;
    metrics_.status_message = "Training resumed";
}

RLTrainingMetrics RLTrainingExecutor::GetMetrics() const {
    std::lock_guard lock(metrics_mutex_);
    return metrics_;
}

void RLTrainingExecutor::TrainingLoop() {
    spdlog::info("RLTrainingExecutor: Starting training loop");
    spdlog::info("  MJCF: {}", config_.env_mjcf_path);
    spdlog::info("  Total timesteps: {}", config_.total_timesteps);
    spdlog::info("  Plugin: {}", config_.plugin_qualified_name);

    // Get the MuJoCo plugin provider
    auto* provider = plugin::PluginNodeRegistry::Instance()
                         .GetNodeProvider(config_.plugin_qualified_name);
    if (!provider) {
        spdlog::error("RLTrainingExecutor: Plugin provider not found for '{}'",
                       config_.plugin_qualified_name);
        std::lock_guard lock(metrics_mutex_);
        metrics_.is_training = false;
        metrics_.status_message = "Error: Plugin not found";
        is_training_ = false;
        return;
    }

    // Load the MJCF model via plugin EvaluateNode with special "load" command
    // The MuJoCo environment should already be loaded via the Properties panel.
    // We'll use EvaluateNode to step the simulation.

    // Determine observation and action dimensions from the plugin
    // Use ResolveDynamicPins to get actuator count
    std::map<std::string, std::string> params;
    params["mjcf_path"] = config_.env_mjcf_path;
    auto pins = provider->ResolveDynamicPins("MuJoCoPlant", params);

    int act_dim = 0;
    for (const auto& pin : pins.pins) {
        if (pin.is_input && pin.name != "u") act_dim++;
    }
    if (act_dim == 0) act_dim = 1;  // fallback: single control

    // Obs dim: depends on filter config
    // For now, evaluate once to get dimensions
    plugin::PluginNodeEvalContext obs_ctx;
    obs_ctx.node_type_name = "ObservationFilter";
    obs_ctx.parameters["include_qpos"] = config_.include_qpos ? "true" : "false";
    obs_ctx.parameters["include_qvel"] = config_.include_qvel ? "true" : "false";
    obs_ctx.parameters["include_sensors"] = config_.include_sensors ? "true" : "false";
    obs_ctx.parameters["normalize"] = config_.normalize_obs ? "true" : "false";

    auto obs_result = provider->EvaluateNode(obs_ctx);
    int obs_dim = 0;
    if (obs_result.success) {
        auto it = obs_result.output_values.find("obs");
        if (it != obs_result.output_values.end() &&
            std::holds_alternative<std::vector<float>>(it->second)) {
            obs_dim = static_cast<int>(std::get<std::vector<float>>(it->second).size());
        }
    }

    if (obs_dim == 0) {
        spdlog::error("RLTrainingExecutor: Could not determine observation dimension");
        std::lock_guard lock(metrics_mutex_);
        metrics_.is_training = false;
        metrics_.status_message = "Error: Cannot determine obs dim (load MJCF first)";
        is_training_ = false;
        return;
    }

    spdlog::info("RLTrainingExecutor: obs_dim={}, act_dim={}", obs_dim, act_dim);

    // We can't directly include ppo_agent.h from the plugin since it's a DLL.
    // Instead, we use the plugin's EvaluateNode with a special RLAgent node type
    // that encapsulates the PPO logic.
    //
    // For Phase 4.4, we implement the training loop here using EvaluateNode calls
    // to step the environment and compute rewards, while the PPO agent runs
    // engine-side.
    //
    // However, to avoid cross-DLL PPOAgent instantiation issues, we use
    // EvaluateNode for the RL Agent node which internally manages PPO.

    // Initialize RL Agent via EvaluateNode
    plugin::PluginNodeEvalContext init_ctx;
    init_ctx.node_type_name = "RLAgent";
    init_ctx.parameters["algorithm"] = "PPO";
    init_ctx.parameters["learning_rate"] = std::to_string(config_.learning_rate);
    init_ctx.parameters["gamma"] = std::to_string(config_.gamma);
    init_ctx.parameters["gae_lambda"] = std::to_string(config_.gae_lambda);
    init_ctx.parameters["clip_range"] = std::to_string(config_.clip_range);
    init_ctx.parameters["n_steps"] = std::to_string(config_.n_steps);
    init_ctx.parameters["batch_size"] = std::to_string(config_.batch_size);
    init_ctx.parameters["n_epochs"] = std::to_string(config_.n_epochs);

    // Build hidden_sizes string
    std::string hidden_str;
    for (size_t i = 0; i < config_.hidden_sizes.size(); ++i) {
        if (i > 0) hidden_str += ",";
        hidden_str += std::to_string(config_.hidden_sizes[i]);
    }
    init_ctx.parameters["hidden_sizes"] = hidden_str;
    init_ctx.parameters["obs_dim"] = std::to_string(obs_dim);
    init_ctx.parameters["act_dim"] = std::to_string(act_dim);
    init_ctx.parameters["command"] = "init";

    auto init_result = provider->EvaluateNode(init_ctx);
    if (!init_result.success) {
        spdlog::warn("RLTrainingExecutor: RLAgent init not yet implemented in plugin, "
                      "training will use simplified loop");
    }

    // ---- Main training loop ----
    int episode = 0;
    int timestep = 0;
    float running_reward = 0.0f;
    float running_length = 0.0f;
    int running_count = 0;

    {
        std::lock_guard lock(metrics_mutex_);
        metrics_.status_message = "Training...";
    }

    while (timestep < config_.total_timesteps && !stop_requested_.load()) {
        // Pause handling
        while (is_paused_.load() && !stop_requested_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (stop_requested_.load()) break;

        // --- Run one episode ---
        // Reset environment (step with zero control)
        plugin::PluginNodeEvalContext reset_ctx;
        reset_ctx.node_type_name = "MuJoCoPlant";
        reset_ctx.parameters = params;
        reset_ctx.sim_time = 0.0f;
        reset_ctx.dt = 0.002f;

        auto compute_reward = [&]() -> float {
            plugin::PluginNodeEvalContext ctx;
            ctx.node_type_name = "RewardFunction";
            ctx.parameters["alive_bonus"] = std::to_string(config_.alive_bonus);
            ctx.parameters["ctrl_cost_weight"] = std::to_string(config_.ctrl_cost_weight);
            ctx.parameters["velocity_reward"] = config_.velocity_reward ? "true" : "false";
            auto r = provider->EvaluateNode(ctx);
            if (r.success) {
                auto it = r.output_values.find("reward");
                if (it != r.output_values.end() &&
                    std::holds_alternative<float>(it->second))
                    return std::get<float>(it->second);
            }
            return 0.0f;
        };

        float episode_reward = 0.0f;
        int episode_length = 0;

        for (int step = 0; step < config_.max_episode_steps; ++step) {
            if (stop_requested_.load()) break;

            // Step the MuJoCo environment (with zero actions for now)
            // In Phase 4.4+, the RLAgent node provides actions
            plugin::PluginNodeEvalContext step_ctx;
            step_ctx.node_type_name = "MuJoCoPlant";
            step_ctx.parameters = params;
            step_ctx.sim_time = static_cast<float>(step) * 0.002f;
            step_ctx.dt = 0.002f;
            // Zero action (placeholder - RLAgent integration comes with full wiring)
            provider->EvaluateNode(step_ctx);

            float reward = compute_reward();
            episode_reward += reward;
            episode_length++;
            timestep++;

            if (timestep >= config_.total_timesteps) break;
        }

        // Update running stats
        running_reward += episode_reward;
        running_length += static_cast<float>(episode_length);
        running_count++;

        // Report episode
        if (episode_callback_) {
            episode_callback_(episode, episode_reward, static_cast<float>(episode_length));
        }

        // Update metrics
        {
            std::lock_guard lock(metrics_mutex_);
            metrics_.episode_count = episode + 1;
            metrics_.total_timesteps = timestep;
            metrics_.mean_episode_reward = running_reward / running_count;
            metrics_.mean_episode_length = running_length / running_count;
            metrics_.reward_history.push_back(episode_reward);
            metrics_.length_history.push_back(static_cast<float>(episode_length));
            metrics_.status_message = "Episode " + std::to_string(episode + 1) +
                " | Reward: " + std::to_string(episode_reward).substr(0, 6) +
                " | Steps: " + std::to_string(timestep) + "/" + std::to_string(config_.total_timesteps);
        }

        // Viewport update
        if (episode % viewport_update_interval_ == 0 && update_callback_) {
            update_callback_(timestep, GetMetrics());
        }

        episode++;
    }

    // Training complete
    {
        std::lock_guard lock(metrics_mutex_);
        metrics_.is_training = false;
        metrics_.status_message = "Training complete (" + std::to_string(metrics_.episode_count) +
            " episodes, " + std::to_string(metrics_.total_timesteps) + " timesteps)";
    }
    is_training_ = false;

    if (complete_callback_) {
        complete_callback_(GetMetrics());
    }

    spdlog::info("RLTrainingExecutor: Training complete ({} episodes, {} timesteps)",
                  episode, timestep);
}

} // namespace cyxwiz
