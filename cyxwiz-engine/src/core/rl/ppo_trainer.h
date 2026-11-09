#pragma once

// Proximal Policy Optimization in the Engine (TOFIX195 S4.2): collects a
// rollout from N copies of an environment, computes GAE on the device and
// updates a policy network and a value network for K epochs of minibatches,
// as Stable-Baselines3's PPO does (its defaults, its loss: cyxwiz/rl/ppo.h).
// The networks are ordinary built models (IExecutableModel); updates go
// through the supervised trainer's ApplyOptimizerStep. GUI-free.

#include "../executable_model.h"
#include "../../plugin/interfaces/i_environment_provider.h"
#include "vec_normalize.h"

#include <cyxwiz/optimizers/optimizer_base.h>
#include <cyxwiz/rl/ppo.h>
#include <cyxwiz/tensor.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>

namespace cyxwiz {
namespace rl {

// What the trainer needs from N copies of an environment. The host adapts
// its batch (the plugin registry's EnvironmentBatch) into this, so this
// library does not depend on the plugin host.
struct PpoEnvironment {
    plugin::PluginEnvironmentInfo info;
    int count = 0;
    std::function<bool(uint64_t seed, float* observations, std::string& error)> reset;
    std::function<bool(const float* actions, const plugin::PluginEnvironmentStepBuffers& out, std::string& error)> step;
};

// Stable-Baselines3 PPO's defaults.
struct PpoConfig {
    int64_t total_timesteps = 100000;
    int n_steps = 2048;        // per copy, per rollout
    int batch_size = 64;
    int n_epochs = 10;
    float learning_rate = 3e-4f;
    float adam_epsilon = 1e-5f;
    float gamma = 0.99f;
    float gae_lambda = 0.95f;
    float max_grad_norm = 0.5f;  // over the policy, the value and log_std together; 0: off
    float log_std_init = 0.0f;
    PpoLossSettings loss;
    uint64_t seed = 0;
    // VecNormalize; its gamma is the trainer's
    bool normalize_observations = false;
    bool normalize_rewards = false;
    float clip_observations = 10.0f;
    float clip_rewards = 10.0f;
    // Stop once the last 100 episodes average at least this; NaN: run all steps
    float target_return = std::numeric_limits<float>::quiet_NaN();
};

// After each update.
struct PpoUpdateMetrics {
    int update = 0;
    int64_t timesteps = 0;
    int episodes = 0;                 // finished so far
    float mean_episode_return = 0.0f; // over the last 100 episodes (NaN before the first)
    float mean_episode_length = 0.0f;
    float loss = 0.0f;
    float policy_loss = 0.0f;
    float value_loss = 0.0f;
    float entropy_loss = 0.0f;
    float approx_kl = 0.0f;
    float clip_fraction = 0.0f;
    float explained_variance = 0.0f;
    float std = 0.0f;                 // mean action standard deviation (Gaussian)
    float steps_per_second = 0.0f;
    bool reached_target = false;      // the run stops after this update
};

// Stable-Baselines3's MlpPolicy initialisation (ortho_init): every weight
// matrix orthogonal with gain sqrt 2, the last one (the output layer) with
// output_gain (0.01 for a policy, 1 for a value network), biases zero. Draws
// from the current ArrayFire random engine.
void InitializeLikeStableBaselines3(IExecutableModel& model, float output_gain);

// What a run of PpoTrainer leaves to play or save: the networks, the log
// standard deviation, the observation statistics and the optimizers' states
// ("policy", "value", and "log_std" for a continuous policy).
struct PpoTrainedState {
    std::unique_ptr<IExecutableModel> policy;
    std::unique_ptr<IExecutableModel> value;
    Tensor log_std;
    std::unique_ptr<VecNormalize> normalizer;
    std::map<std::string, OptimizerState> optimizers;
};

class PpoTrainer {
public:
    // policy: observations [B, obs] -> means [B, A] (continuous) or logits [B, K] (discrete);
    // value: observations [B, obs] -> [B, 1].
    PpoTrainer(PpoEnvironment environment, std::unique_ptr<IExecutableModel> policy,
               std::unique_ptr<IExecutableModel> value, PpoConfig config);
    ~PpoTrainer();

    // Runs to total_timesteps, or until Stop. False with the reason on an
    // environment or device failure.
    bool Train(const std::function<void(const PpoUpdateMetrics&)>& on_update, std::string& error);

    void Stop() { stop_ = true; }
    void Pause() { paused_ = true; }
    void Resume() { paused_ = false; }

    // Moves out what Train left (once, after Train). False with the reason
    // when Train never started or an optimizer's state cannot be exported.
    bool TakeTrained(PpoTrainedState& out, std::string& error);

private:
    bool CheckSetup(std::string& error) const;

    PpoEnvironment environment_;
    std::unique_ptr<IExecutableModel> policy_;
    std::unique_ptr<IExecutableModel> value_;
    PpoConfig config_;
    Tensor log_std_;
    std::unique_ptr<Optimizer> policy_optimizer_;
    std::unique_ptr<Optimizer> value_optimizer_;
    std::unique_ptr<Optimizer> log_std_optimizer_;
    std::unique_ptr<VecNormalize> normalizer_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> paused_{false};
};

}  // namespace rl
}  // namespace cyxwiz
