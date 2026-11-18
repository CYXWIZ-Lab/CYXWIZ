#pragma once

// Soft Actor-Critic in the Engine (TOFIX216 S3), as Stable-Baselines3 2.7's
// SAC does it: random actions until learning starts, then the actor's own
// draws; every train_freq steps, gradient_steps updates of the entropy weight,
// the twin critics and the actor from minibatches of a replay buffer on the
// training device, with Polyak-averaged target critics (cyxwiz/rl/sac.h has
// the math). The networks are ordinary built models (IExecutableModel);
// updates go through ApplyOptimizerStep. GUI-free.

#include "../executable_model.h"
#include "replay_buffer.h"
#include "sac_config.h"
#include "training_environment.h"
#include "vec_normalize.h"

#include <cyxwiz/optimizers/optimizer_base.h>
#include <cyxwiz/tensor.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz::rl {

// PyTorch's default nn.Linear initialisation, which SB3's SAC keeps: every
// weight and bias uniform in +-1/sqrt(fan_in). Draws from the current
// ArrayFire random engine.
void InitializeLikeTorchLinear(IExecutableModel& model);

// The actor gives [mean | log std] per row ([B, 2A], log std before its
// clamp); each critic takes [observation | action] and gives [B, 1]; the
// targets start as copies of the critics.
struct SacNetworks {
    std::unique_ptr<IExecutableModel> actor;
    std::unique_ptr<IExecutableModel> q1;
    std::unique_ptr<IExecutableModel> q2;
    std::unique_ptr<IExecutableModel> q1_target;
    std::unique_ptr<IExecutableModel> q2_target;
};

// The actor's actions in [-1, 1] [N, A] for observations [N, obs] (already
// normalised when the policy was trained so): tanh(mean) when noise is null,
// else drawn with noise [N, A] as in training. Training, preview and playback
// all act through it.
Tensor SacActorActions(IExecutableModel& actor, const Tensor& observations, size_t action_size, const Tensor* noise);

// One gradient step's numbers, on the device ([1] each)
struct SacStepLosses {
    Tensor critic_loss;
    Tensor actor_loss;
    Tensor ent_coef;       // alpha this step used
    Tensor ent_coef_loss;  // empty with a fixed entropy weight
    Tensor mean_q;         // mean min(Q1, Q2) of the actor's actions
};

// The networks, their optimizers and the entropy weight, and SB3's
// SAC.train for one gradient step. Works on the current ArrayFire device.
class SacLearner {
public:
    // target_entropy: the config's, already resolved (auto = -actions)
    SacLearner(SacNetworks networks, size_t observation_size, size_t action_size, const SacConfig& config);
    ~SacLearner();

    // One gradient step on a replay batch (actions in [-1, 1]); noise_pi and
    // noise_next [B, A] are the actor's draws on the batch's observations and
    // on its next observations. update_targets: Polyak-average the targets.
    SacStepLosses Step(const ReplayBuffer::Batch& batch, const Tensor& noise_pi, const Tensor& noise_next,
                       bool update_targets);

    // Actions in [-1, 1] [N, A] for observations [N, obs]: tanh(mean) when
    // noise is null, else drawn as in training.
    Tensor Act(const Tensor& observations, const Tensor* noise);

    SacNetworks& Networks() { return networks_; }
    const Tensor& LogEntCoef() const { return log_ent_coef_; }
    // "actor", "q1", "q2" and, when the weight is learned, "ent_coef"
    bool ExportOptimizers(std::map<std::string, OptimizerState>& out, std::string& error) const;

private:
    SacNetworks networks_;
    size_t observation_size_;
    size_t action_size_;
    SacConfig config_;
    Tensor log_ent_coef_;  // [1]
    Tensor fixed_ent_coef_;
    std::unique_ptr<Optimizer> actor_optimizer_;
    std::unique_ptr<Optimizer> q1_optimizer_;
    std::unique_ptr<Optimizer> q2_optimizer_;
    std::unique_ptr<Optimizer> ent_coef_optimizer_;
};

// Reported every report_every environment steps (and at the end, or at a
// Stop), averaged over the gradient steps since the last report.
struct SacUpdateMetrics {
    int report = 0;
    int64_t timesteps = 0;
    int64_t updates = 0;               // gradient steps so far
    bool learning = false;             // past learning_starts
    int episodes = 0;
    float mean_episode_return = 0.0f;  // over the last 100 episodes (NaN before the first)
    float mean_episode_length = 0.0f;
    float critic_loss = 0.0f;
    float actor_loss = 0.0f;
    float ent_coef = 0.0f;
    float ent_coef_loss = 0.0f;
    float mean_q = 0.0f;
    int64_t buffer_size = 0;
    int64_t buffer_capacity = 0;
    uint64_t buffer_bytes = 0;         // the whole table on the device
    float steps_per_second = 0.0f;
    bool reached_target = false;       // the run stops after this report
    std::string compute_path;
    std::vector<plugin::PluginEnvironmentStat> environment_stats;
};

// What a run leaves to play or save.
struct SacTrainedState {
    SacNetworks networks;
    Tensor log_ent_coef;
    std::unique_ptr<VecNormalize> normalizer;
    std::map<std::string, OptimizerState> optimizers;
};

// The actor as it is between two reports (a training preview).
struct SacPolicyCopy {
    std::map<std::string, Tensor> parameters;
    std::unique_ptr<VecNormalize> normalizer;
};

class SacTrainer {
public:
    // The environment's actions must be continuous and bounded.
    SacTrainer(TrainingEnvironment environment, SacNetworks networks, SacConfig config);
    ~SacTrainer();

    // Runs to total_timesteps, or until Stop. False with the reason on an
    // environment or device failure (the replay buffer not fitting included).
    bool Train(const std::function<void(const SacUpdateMetrics&)>& on_report, std::string& error);

    void Stop() { stop_ = true; }
    void Pause() { paused_ = true; }
    void Resume() { paused_ = false; }

    // Moves out what Train left (once, after Train). False with the reason.
    bool TakeTrained(SacTrainedState& out, std::string& error);
    // A copy of the actor now. Call from on_report, on the thread that trains.
    bool CopyPolicy(SacPolicyCopy& out, std::string& error);

private:
    bool CheckSetup(std::string& error) const;

    TrainingEnvironment environment_;
    SacNetworks networks_;  // until Train hands them to the learner
    SacConfig config_;
    std::unique_ptr<SacLearner> learner_;
    std::unique_ptr<VecNormalize> normalizer_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> paused_{false};
};

}  // namespace cyxwiz::rl
