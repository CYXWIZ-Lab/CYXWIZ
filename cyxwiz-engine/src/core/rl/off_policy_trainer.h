#pragma once

// The off-policy training loop (TOFIX216), as Stable-Baselines3 2.7's
// OffPolicyAlgorithm runs SAC, TD3 and DQN: uniform random actions (DQN:
// random choices) until learning_starts, then the learner's exploring
// actions; each step's transitions into a replay buffer on the training
// device (an ended copy's next observation is its final one; a choice kept as
// one value), then the learner's AfterStep (SB3's _on_step); every train_freq
// steps, gradient_steps updates of the learner from minibatches drawn from
// it; a report every report_every environment steps. The learner (SacLearner,
// Td3Learner, DqnLearner) holds the networks and its algorithm's update.
// GUI-free.

#include "../executable_model.h"
#include "rl_common.h"
#include "replay_buffer.h"
#include "training_environment.h"
#include "vec_normalize.h"

#include <cyxwiz/optimizers/optimizer_base.h>
#include <cyxwiz/tensor.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
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

// The loop's settings of a SAC, TD3 or DQN config (each names them as the loop does)
template <class Config>
OffPolicySettings LoopSettings(const Config& c) {
    OffPolicySettings s;
    s.total_timesteps = c.total_timesteps;
    s.buffer_size = c.buffer_size;
    s.learning_starts = c.learning_starts;
    s.batch_size = c.batch_size;
    s.train_freq = c.train_freq;
    s.gradient_steps = c.gradient_steps;
    s.gamma = c.gamma;
    s.seed = c.seed;
    s.normalize_observations = c.normalize_observations;
    s.target_return = c.target_return;
    s.report_every = c.report_every;
    return s;
}


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
    float critic_loss = 0.0f;          // the Q networks' loss (DQN: its Huber loss)
    float actor_loss = 0.0f;
    float mean_q = 0.0f;               // the critics' value of the actor's actions (DQN: of the chosen ones)
    float exploration_rate = 0.0f;     // DQN: the share of random choices now (epsilon)
    float ent_coef = 0.0f;             // SAC: the entropy weight
    float ent_coef_loss = 0.0f;        // SAC, when the weight is learned
    int64_t buffer_size = 0;
    int64_t buffer_capacity = 0;
    uint64_t buffer_bytes = 0;         // the whole table on the device
    float steps_per_second = 0.0f;
    bool reached_target = false;       // the run stops after this report
    // How the run computes: the networks, then the Adam step, each with the
    // provider that runs it or ArrayFire with the reason (FormatComputePath)
    std::string compute_path;
    std::vector<ComputePathPart> compute_parts;  // the same as data
    // "<name>. <fix>" when the networks are on ArrayFire because a provider
    // is missing and can be supplied; empty otherwise (NetworkRoute)
    std::string missing_provider;
    std::vector<plugin::PluginEnvironmentStat> environment_stats;
};

// An algorithm's networks and update, driven by OffPolicyLoop. Every random
// draw comes from next_seed, in the loop's order, so a seed gives one run.
class OffPolicyLearner {
public:
    using NextSeed = std::function<uint64_t()>;
    virtual ~OffPolicyLearner() = default;

    // Actions to explore with, for observations [N, obs] (past the random
    // steps): in [-1, 1] [N, A] for continuous actions; for a choice of
    // actions, N choice indices (as floats)
    virtual Tensor Explore(const Tensor& observations, const NextSeed& next_seed) = 0;
    // The random steps' uniform actions [N, A] as the environment gets them:
    // as they are, unless the learner adds its exploration noise to them too
    // (TD3, as SB3 does)
    virtual Tensor Warmup(const Tensor& actions, const NextSeed& /*next_seed*/) { return actions; }
    // After each step of the copies, its transitions stored (SB3's _on_step):
    // timesteps, the environment steps so far; vector_steps, the steps of the
    // copies so far. DQN moves its target network and its epsilon here.
    virtual void AfterStep(int64_t /*timesteps*/, int64_t /*vector_steps*/) {}
    // One gradient step on a replay batch; gradient_step counts within this
    // training call (SB3's train(), whose counters some settings follow)
    virtual void Learn(const ReplayBuffer::Batch& batch, int gradient_step, const NextSeed& next_seed) = 0;
    // The losses averaged over the gradient steps since the last report into
    // metrics (left as they are when there were none), then starts again
    virtual void Report(OffPolicyUpdateMetrics& metrics) = 0;
    // How the networks run, and why when that is not a provider
    virtual const NetworkRoute& Route() const = 0;
    // Whether its optimizers still step together on the provider, and why not
    virtual const AdamRoute& Adam() const = 0;
    // Its words: "fused small-network kernels (<provider>)" or "ArrayFire
    // operations (<why>)"
    const std::string& NetworkPath() const { return Route().text; }
};

// The environment and the loop's settings every off-policy learner needs:
// continuous, bounded actions (choices: a choice of actions, DQN's); counts of
// at least 1 (the random steps first at least 0). algorithm names it in the
// reasons ("SAC").
bool CheckOffPolicySetup(const TrainingEnvironment& environment, const OffPolicySettings& settings,
                         const std::string& algorithm, std::string& error, bool choices = false);

class OffPolicyLoop {
public:
    OffPolicyLoop(TrainingEnvironment environment, OffPolicySettings settings, std::string algorithm);

    const TrainingEnvironment& Environment() const { return environment_; }
    const OffPolicySettings& Settings() const { return settings_; }
    const std::string& Algorithm() const { return algorithm_; }

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
// adam: whether to try one launch; cleared, for good, with why, when it
// cannot (refused before any provider call, or the call failed).
// max_grad_norm > 0: all the gradients' global norm clipped to it first
// (clip_grad_norm_, DQN's).
struct OffPolicyUpdate {
    Optimizer* optimizer;
    IExecutableModel* model;                    // its parameters, or (null)
    std::map<std::string, Tensor>* parameters;  // these, updated in place
    std::map<std::string, Tensor>* gradients;
};
void StepOffPolicyOptimizers(const std::vector<OffPolicyUpdate>& updates, AdamRoute& adam, const char* learner,
                              float max_grad_norm = 0.0f);

// ---- The trainer around a learner

// What SacTrainer, Td3Trainer and DqnTrainer share: the loop, the networks
// until Train hands them to the learner, Stop / Pause / Resume and what a run
// leaves. Each trainer checks its own setup and builds its own learner;
// TrainedState holds networks, normalizer and optimizers.
template <class Learner, class Networks, class Config, class TrainedState>
class OffPolicyTrainer {
public:
    virtual ~OffPolicyTrainer() = default;

    // Runs to total_timesteps, the target return, or a Stop. False with the
    // reason on an environment or device failure (the replay buffer not
    // fitting included).
    bool Train(const std::function<void(const OffPolicyUpdateMetrics&)>& on_report, std::string& error) {
        error.clear();
        if (!CheckSetup(error)) return false;
        try {
            learner_ = MakeLearner(std::move(networks_));
        } catch (const std::exception& e) {
            error = loop_.Algorithm() + " training failed: " + e.what();
            return false;
        }
        return loop_.Run(*learner_, on_report, error);
    }

    void Stop() { loop_.Stop(); }
    void Pause() { loop_.Pause(); }
    void Resume() { loop_.Resume(); }
    // Its loop (a run stops, pauses and resumes it there)
    OffPolicyLoop& Loop() { return loop_; }

    // Moves out what Train left (once, after Train). False with the reason.
    bool TakeTrained(TrainedState& out, std::string& error) {
        if (!learner_ || !loop_.Normalizer()) {
            error = loop_.Algorithm() + " has not trained a policy.";
            return false;
        }
        std::map<std::string, OptimizerState> optimizers;
        if (!learner_->ExportOptimizers(optimizers, error)) return false;
        out.optimizers = std::move(optimizers);
        TakeMore(*learner_, out);
        out.networks = std::move(learner_->Networks());
        out.normalizer = loop_.TakeNormalizer();
        learner_.reset();
        return true;
    }

    // A copy of the policy network now (the actor; DQN's Q network). Call
    // from on_report, on the thread that trains.
    bool CopyPolicy(ActorCopy& out, std::string& error) {
        if (!learner_ || !loop_.Normalizer()) {
            error = loop_.Algorithm() + " has no policy to copy yet.";
            return false;
        }
        out.parameters.clear();
        for (const auto& [name, tensor] : (learner_->Networks().*policy_)->GetParameters())
            out.parameters[name] = tensor.Clone();
        out.normalizer = std::make_unique<VecNormalize>(*loop_.Normalizer());
        return true;
    }

protected:
    // Which of the networks a preview copies
    using PolicyNetwork = std::unique_ptr<IExecutableModel> Networks::*;

    OffPolicyTrainer(TrainingEnvironment environment, Networks networks, Config config, std::string algorithm,
                     PolicyNetwork policy)
        : loop_(std::move(environment), LoopSettings(config), std::move(algorithm)), networks_(std::move(networks)),
          config_(std::move(config)), policy_(policy) {}

    // The trainer's own rules on its networks and config (CheckOffPolicySetup's among them)
    virtual bool CheckSetup(std::string& error) const = 0;
    // Its learner on the networks; throws when it cannot be built
    virtual std::unique_ptr<Learner> MakeLearner(Networks networks) = 0;
    // What the learner leaves besides the networks and optimizers (SAC: its entropy weight)
    virtual void TakeMore(Learner& /*learner*/, TrainedState& /*out*/) {}

    OffPolicyLoop loop_;
    Networks networks_;  // until Train hands them to the learner
    Config config_;

private:
    PolicyNetwork policy_;
    std::unique_ptr<Learner> learner_;
};

}  // namespace cyxwiz::rl
