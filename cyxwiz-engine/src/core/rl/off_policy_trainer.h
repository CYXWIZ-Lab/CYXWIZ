#pragma once

// The off-policy training loop (TOFIX216), as Stable-Baselines3 2.7's
// OffPolicyAlgorithm runs SAC and TD3: uniform random actions until
// learning_starts, then the learner's exploring actions; each step's
// transitions into a replay buffer on the training device (an ended copy's
// next observation is its final one); every train_freq steps, gradient_steps
// updates of the learner from minibatches drawn from it; a report every
// report_every environment steps. The learner (SacLearner, Td3Learner) holds
// the networks and its algorithm's update. GUI-free.

#include "../executable_model.h"
#include "replay_buffer.h"
#include "training_environment.h"
#include "vec_normalize.h"

#include <cyxwiz/optimizers/optimizer_base.h>
#include <cyxwiz/tensor.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz::rl {

// The loop's own settings, SB3's names; each algorithm's config gives them.
struct OffPolicySettings {
    int64_t total_timesteps = 1000000;
    int64_t buffer_size = 1000000;
    int64_t learning_starts = 100;
    int batch_size = 256;
    int train_freq = 1;
    int gradient_steps = 1;
    float gamma = 0.99f;  // VecNormalize's reward discount (observations only here)
    uint64_t seed = 1;
    bool normalize_observations = false;
    float target_return = std::numeric_limits<float>::quiet_NaN();  // NaN: no early stop
    int64_t report_every = 1000;
};

// Reported every report_every environment steps (and at the end, or at a
// Stop), the losses averaged over the gradient steps since the last report.
struct OffPolicyUpdateMetrics {
    int report = 0;
    int64_t timesteps = 0;
    int64_t updates = 0;               // gradient steps so far
    bool learning = false;             // past learning_starts
    int episodes = 0;
    float mean_episode_return = 0.0f;  // over the last 100 episodes (NaN before the first)
    float mean_episode_length = 0.0f;
    float critic_loss = 0.0f;
    float actor_loss = 0.0f;
    float mean_q = 0.0f;               // the critics' value of the actor's actions
    float ent_coef = 0.0f;             // SAC: the entropy weight
    float ent_coef_loss = 0.0f;        // SAC, when the weight is learned
    int64_t buffer_size = 0;
    int64_t buffer_capacity = 0;
    uint64_t buffer_bytes = 0;         // the whole table on the device
    float steps_per_second = 0.0f;
    bool reached_target = false;       // the run stops after this report
    std::string compute_path;
    std::vector<plugin::PluginEnvironmentStat> environment_stats;
};

// An algorithm's networks and update, driven by OffPolicyLoop. Every random
// draw comes from next_seed, in the loop's order, so a seed gives one run.
class OffPolicyLearner {
public:
    using NextSeed = std::function<uint64_t()>;
    virtual ~OffPolicyLearner() = default;

    // Actions in [-1, 1] [N, A] to explore with, for observations [N, obs]
    // (past the random steps)
    virtual Tensor Explore(const Tensor& observations, const NextSeed& next_seed) = 0;
    // The random steps' uniform actions [N, A] as the environment gets them:
    // as they are, unless the learner adds its exploration noise to them too
    // (TD3, as SB3 does)
    virtual Tensor Warmup(const Tensor& actions, const NextSeed& /*next_seed*/) { return actions; }
    // One gradient step on a replay batch; gradient_step counts within this
    // training call (SB3's train(), whose counters some settings follow)
    virtual void Learn(const ReplayBuffer::Batch& batch, int gradient_step, const NextSeed& next_seed) = 0;
    // The losses averaged over the gradient steps since the last report into
    // metrics (left as they are when there were none), then starts again
    virtual void Report(OffPolicyUpdateMetrics& metrics) = 0;
    // How the networks run: "fused small-network kernels (<provider>)" or
    // "ArrayFire operations" (with the reason a fused call failed)
    virtual const std::string& NetworkPath() const = 0;
};

// The environment and the loop's settings every off-policy learner needs:
// continuous, bounded actions; counts of at least 1 (the random steps first at
// least 0). algorithm names it in the reasons ("SAC").
bool CheckOffPolicySetup(const TrainingEnvironment& environment, const OffPolicySettings& settings,
                         const std::string& algorithm, std::string& error);

class OffPolicyLoop {
public:
    OffPolicyLoop(TrainingEnvironment environment, OffPolicySettings settings, std::string algorithm);

    const TrainingEnvironment& Environment() const { return environment_; }
    const OffPolicySettings& Settings() const { return settings_; }

    // Runs to total_timesteps, the target return, or a Stop. False with the
    // reason on an environment or device failure (the replay buffer not
    // fitting included). Call CheckOffPolicySetup first.
    bool Run(OffPolicyLearner& learner, const std::function<void(const OffPolicyUpdateMetrics&)>& on_report,
             std::string& error);

    void Stop() { stop_ = true; }
    void Pause() { paused_ = true; }
    void Resume() { paused_ = false; }

    // The observation statistics a run used (null before Run)
    const VecNormalize* Normalizer() const { return normalizer_.get(); }
    std::unique_ptr<VecNormalize> TakeNormalizer() { return std::move(normalizer_); }

private:
    TrainingEnvironment environment_;
    OffPolicySettings settings_;
    std::string algorithm_;
    std::unique_ptr<VecNormalize> normalizer_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> paused_{false};
};

// An off-policy learner's actor as it is between two reports (a training
// preview), with the observation statistics
struct ActorCopy {
    std::map<std::string, Tensor> parameters;
    std::unique_ptr<VecNormalize> normalizer;
};

// ---- Pieces the learners share

// PyTorch's default nn.Linear initialisation, which SB3's SAC and TD3 keep:
// every weight and bias uniform in +-1/sqrt(fan_in). Draws from the current
// ArrayFire random engine.
void InitializeLikeTorchLinear(IExecutableModel& model);
// SB3's optimizer for the off-policy learners: th.optim.Adam with its defaults
std::unique_ptr<Optimizer> OffPolicyAdam(float learning_rate);
// A network's parameters copied into its target
void CopyParameters(IExecutableModel& from, IExecutableModel& to);
// target = (1 - tau) * target + tau * network, parameter by parameter
void PolyakUpdate(IExecutableModel& network, IExecutableModel& target, float tau);
// The first value of a device tensor
float FirstValue(const Tensor& t);

// One Adam step for each of several networks (or loose tensors, as SAC's log
// entropy weight): in one launch when the device's provider steps Adam
// (AdamOptimizer::StepTogether), else each through ApplyOptimizerStep.
// together: whether to try one launch; cleared, for good, when it cannot.
struct OffPolicyUpdate {
    Optimizer* optimizer;
    IExecutableModel* model;                    // its parameters, or (null)
    std::map<std::string, Tensor>* parameters;  // these, updated in place
    std::map<std::string, Tensor>* gradients;
};
void StepOffPolicyOptimizers(const std::vector<OffPolicyUpdate>& updates, bool& together, const char* learner);

}  // namespace cyxwiz::rl
