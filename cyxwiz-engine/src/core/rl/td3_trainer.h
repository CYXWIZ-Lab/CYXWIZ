#pragma once

// Twin Delayed DDPG in the Engine (TOFIX216 TD3), as Stable-Baselines3 2.7's
// TD3 does it, on the off-policy loop (off_policy_trainer.h): after the
// random steps the actor's actions plus Gaussian noise; every gradient step
// updates the twin critics toward the smaller target Q of smoothed target
// actions, and every policy_delay-th step the actor (from Q network 1) and,
// by Polyak averaging, the target critics and the target actor
// (cyxwiz/rl/td3.h has the math). The networks are ordinary built models
// (IExecutableModel), the actor ending in tanh; small Dense/activation
// networks run on the device's fused kernels as SAC's do. GUI-free.

#include "off_policy_trainer.h"
#include "td3_config.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz::rl {

// The actor gives actions in [-1, 1] [B, A] (it ends in tanh); each critic
// takes [observation | action] and gives [B, 1]; the targets start as copies
// of the actor and the critics.
struct Td3Networks {
    std::unique_ptr<IExecutableModel> actor;
    std::unique_ptr<IExecutableModel> actor_target;
    std::unique_ptr<IExecutableModel> q1;
    std::unique_ptr<IExecutableModel> q2;
    std::unique_ptr<IExecutableModel> q1_target;
    std::unique_ptr<IExecutableModel> q2_target;
};

// One gradient step's numbers, on the device ([1] each); the actor's only
// on a policy step
struct Td3StepLosses {
    Tensor critic_loss;
    bool policy_updated = false;
    Tensor actor_loss;
    Tensor mean_q;  // mean Q1 of the actor's actions
};

class Td3Learner : public OffPolicyLearner {
public:
    Td3Learner(Td3Networks networks, size_t observation_size, size_t action_size, const Td3Config& config);
    ~Td3Learner() override;

    // One gradient step on a replay batch (actions in [-1, 1]); noise [B, A]
    // standard normal, the target smoothing's draw. The actor and the targets
    // move on every policy_delay-th step this learner takes.
    Td3StepLosses Step(const ReplayBuffer::Batch& batch, const Tensor& noise);

    // The actor's actions in [-1, 1] [N, A] for observations [N, obs]
    Tensor Act(const Tensor& observations);

    // The loop's side: the actor's actions plus the exploration noise; the
    // random steps' actions plus the same noise (SB3 adds it to both); a Step
    // with a fresh smoothing draw; the losses' means since the last report
    Tensor Explore(const Tensor& observations, const NextSeed& next_seed) override;
    Tensor Warmup(const Tensor& actions, const NextSeed& next_seed) override;
    void Learn(const ReplayBuffer::Batch& batch, int gradient_step, const NextSeed& next_seed) override;
    void Report(OffPolicyUpdateMetrics& metrics) override;
    const NetworkRoute& Route() const override { return network_route_; }

    Td3Networks& Networks() { return networks_; }
    int64_t Updates() const { return updates_; }
    // "actor", "q1", "q2"
    bool ExportOptimizers(std::map<std::string, OptimizerState>& out, std::string& error) const;

private:
    void DropFused(const std::exception& e);
    // The critics' gradients and loss (nothing changed yet)
    struct CriticPass {
        Tensor loss;
        std::map<std::string, Tensor> q1_grads, q2_grads;
    };
    CriticPass CriticsFused(const ReplayBuffer::Batch& batch, const Tensor& noise);
    CriticPass CriticsLayers(const ReplayBuffer::Batch& batch, const Tensor& noise);
    // The actor's loss and gradients against the updated Q network 1
    struct ActorPass {
        Tensor loss, mean_q;
        std::map<std::string, Tensor> grads;
    };
    ActorPass ActorFused(const ReplayBuffer::Batch& batch);
    ActorPass ActorLayers(const ReplayBuffer::Batch& batch);
    Tensor Smoothed(const Tensor& target_actions, const Tensor& noise) const;

    Td3Networks networks_;
    size_t observation_size_;
    size_t action_size_;
    Td3Config config_;
    std::unique_ptr<Optimizer> actor_optimizer_;
    std::unique_ptr<Optimizer> q1_optimizer_;
    std::unique_ptr<Optimizer> q2_optimizer_;
    // All six fused, or none
    std::unique_ptr<FusedMlp> actor_fused_, actor_target_fused_, q1_fused_, q2_fused_, q1_target_fused_,
        q2_target_fused_;
    bool adam_together_ = true;
    NetworkRoute network_route_;
    int64_t updates_ = 0;  // gradient steps taken (SB3's _n_updates)
    // Sums since the last report, on the device
    Tensor critic_sum_, actor_sum_, q_sum_;
    int critic_steps_ = 0, actor_steps_ = 0;
};

// What a run leaves to play or save.
struct Td3TrainedState {
    Td3Networks networks;
    std::unique_ptr<VecNormalize> normalizer;
    std::map<std::string, OptimizerState> optimizers;
};

class Td3Trainer : public OffPolicyTrainer<Td3Learner, Td3Networks, Td3Config, Td3TrainedState> {
public:
    // The environment's actions must be continuous and bounded.
    Td3Trainer(TrainingEnvironment environment, Td3Networks networks, Td3Config config);

private:
    bool CheckSetup(std::string& error) const override;
    std::unique_ptr<Td3Learner> MakeLearner(Td3Networks networks) override;
};

}  // namespace cyxwiz::rl
