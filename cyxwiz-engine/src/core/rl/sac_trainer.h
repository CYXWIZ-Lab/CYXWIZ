#pragma once

// Soft Actor-Critic in the Engine (TOFIX216 S3), as Stable-Baselines3 2.7's
// SAC does it, on the off-policy loop (off_policy_trainer.h): after the random
// steps the actor's own draws; each gradient step updates the entropy weight,
// the twin critics and the actor from a replay minibatch, with Polyak-averaged
// target critics (cyxwiz/rl/sac.h has the math). The networks are ordinary
// built models (IExecutableModel); updates go through ApplyOptimizerStep, or
// AdamOptimizer::StepTogether on a device whose provider steps Adam (S5).
// GUI-free.

#include "off_policy_trainer.h"
#include "sac_config.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {
class FusedMlp;
}

namespace cyxwiz::rl {

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
// Small Dense/activation networks run whole on the device's provider (the
// fused kernels: the twin critics and their targets in one launch each way,
// the critics' input gradient for the actor) and the optimizers step
// together (AdamOptimizer::StepTogether); otherwise, or once a fused call
// fails, the layers run as ArrayFire operations.
class SacLearner : public OffPolicyLearner {
public:
    // target_entropy: the config's, already resolved (auto = -actions)
    SacLearner(SacNetworks networks, size_t observation_size, size_t action_size, const SacConfig& config);
    ~SacLearner() override;

    // The loop's side: a draw with fresh noise; a Step with fresh noise and
    // the targets every target_update_interval gradient steps of the call;
    // the losses' and the entropy weight's means since the last report
    Tensor Explore(const Tensor& observations, const NextSeed& next_seed) override;
    void Learn(const ReplayBuffer::Batch& batch, int gradient_step, const NextSeed& next_seed) override;
    void Report(OffPolicyUpdateMetrics& metrics) override;

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
    // How the networks run: "fused small-network kernels (<provider>)" or
    // "ArrayFire operations" (with the reason a fused call failed)
    const std::string& NetworkPath() const override { return network_path_; }

private:
    // A step's critic gradients and the actor's draw on the batch (nothing
    // changed yet)
    struct CriticPass {
        Tensor critic_loss;
        std::map<std::string, Tensor> q1_grads, q2_grads;
        Tensor mean, log_std;    // the actor's outputs on the batch's observations
        Tensor actions, log_prob;  // its draw with noise_pi
        bool actor_cached = false;  // the actor's layers hold that forward (ArrayFire operations)
    };
    // The actor's loss and gradients against the updated critics
    struct ActorPass {
        Tensor loss, mean_q;
        std::map<std::string, Tensor> grads;
    };
    void DropFused(const std::exception& e);
    CriticPass CriticsFused(const ReplayBuffer::Batch& batch, const Tensor& noise_pi, const Tensor& noise_next,
                            const Tensor& alpha);
    CriticPass CriticsLayers(const ReplayBuffer::Batch& batch, const Tensor& noise_pi, const Tensor& noise_next,
                             const Tensor& alpha);
    ActorPass ActorFused(const ReplayBuffer::Batch& batch, const CriticPass& c, const Tensor& noise_pi,
                         const Tensor& alpha);
    ActorPass ActorLayers(const ReplayBuffer::Batch& batch, const CriticPass& c, const Tensor& noise_pi,
                          const Tensor& alpha);
    // One Adam step of each network's (and the entropy weight's) parameters
    void StepOptimizers(const std::vector<OffPolicyUpdate>& updates);

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
    // All five fused, or none
    std::unique_ptr<FusedMlp> actor_fused_, q1_fused_, q2_fused_, q1_target_fused_, q2_target_fused_;
    bool adam_together_ = true;
    std::string network_path_ = "ArrayFire operations";
    // Sums since the last report, on the device
    Tensor critic_sum_, actor_sum_, alpha_sum_, alpha_loss_sum_, q_sum_;
    int since_report_ = 0;
};

// What a run leaves to play or save.
struct SacTrainedState {
    SacNetworks networks;
    Tensor log_ent_coef;
    std::unique_ptr<VecNormalize> normalizer;
    std::map<std::string, OptimizerState> optimizers;
};

// The loop's settings of a SAC config
OffPolicySettings LoopSettings(const SacConfig& config);

class SacTrainer {
public:
    // The environment's actions must be continuous and bounded.
    SacTrainer(TrainingEnvironment environment, SacNetworks networks, SacConfig config);
    ~SacTrainer();

    // Runs to total_timesteps, or until Stop. False with the reason on an
    // environment or device failure (the replay buffer not fitting included).
    bool Train(const std::function<void(const OffPolicyUpdateMetrics&)>& on_report, std::string& error);

    void Stop() { loop_.Stop(); }
    void Pause() { loop_.Pause(); }
    void Resume() { loop_.Resume(); }
    // Its loop (a run stops, pauses and resumes it there)
    OffPolicyLoop& Loop() { return loop_; }

    // Moves out what Train left (once, after Train). False with the reason.
    bool TakeTrained(SacTrainedState& out, std::string& error);
    // A copy of the actor now. Call from on_report, on the thread that trains.
    bool CopyPolicy(ActorCopy& out, std::string& error);

private:
    bool CheckSetup(std::string& error) const;

    OffPolicyLoop loop_;
    SacNetworks networks_;  // until Train hands them to the learner
    SacConfig config_;
    std::unique_ptr<SacLearner> learner_;
};

}  // namespace cyxwiz::rl
