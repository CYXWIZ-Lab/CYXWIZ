#pragma once

// Deep Q-Network in the Engine (TOFIX216 DQN), as Stable-Baselines3 2.7's DQN
// does it, on the off-policy loop (off_policy_trainer.h): random choices for
// the first learning_starts steps, then epsilon-greedy (one draw for all the
// copies, SB3's predict); every gradient step moves the Q network toward
// r + gamma * the target network's best next Q (Huber loss, the gradients'
// global norm clipped, Adam), and every target_update_interval environment
// steps the target network moves to it by Polyak averaging (tau 1: a copy);
// epsilon falls linearly as the run goes (cyxwiz/rl/dqn.h has the math). The
// Q network is an ordinary built model (IExecutableModel) giving one value
// per choice; a small Dense/activation network runs on the device's fused
// kernels as SAC's and TD3's do. GUI-free.

#include "dqn_config.h"
#include "off_policy_trainer.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz::rl {

// SB3's epsilon-greedy (DQN.predict): one uniform draw for all the rows; below
// epsilon, a random choice of `count` for each row (one draw each, in order)
// into choices and true; else choices as they are (the greedy ones) and false.
// Training and playback both explore through it.
bool EpsilonGreedy(std::vector<float>& choices, float epsilon, size_t count, const std::function<uint64_t()>& next);

// The Q network gives [B, A] for observations [B, obs], one value per
// choice; the target starts as its copy.
struct DqnNetworks {
    std::unique_ptr<IExecutableModel> q;
    std::unique_ptr<IExecutableModel> q_target;
};

// One gradient step's numbers, on the device ([1] each)
struct DqnStepLosses {
    Tensor loss;
    Tensor mean_q;  // mean Q of the batch's chosen actions
};

class DqnLearner : public OffPolicyLearner {
public:
    // copies: the environment's copies (the target moves every
    // target_update_interval / copies steps of them, as SB3 counts)
    DqnLearner(DqnNetworks networks, size_t choices, const DqnConfig& config, int copies);
    ~DqnLearner() override;

    // One gradient step on a replay batch (actions: choice indices); the
    // target network does not move here
    DqnStepLosses Step(const ReplayBuffer::Batch& batch);
    // target = (1 - tau) * target + tau * Q (SB3's polyak_update)
    void UpdateTarget();

    // The greedy choices [N] (as floats) for observations [N, obs]
    Tensor Act(const Tensor& observations);

    // The loop's side: epsilon-greedy choices (one draw for all the copies:
    // all random, or all greedy); after each step of the copies, the target
    // on its interval and epsilon by the run's progress; a Step; the means
    // since the last report and the epsilon now
    Tensor Explore(const Tensor& observations, const NextSeed& next_seed) override;
    void AfterStep(int64_t timesteps, int64_t vector_steps) override;
    void Learn(const ReplayBuffer::Batch& batch, int gradient_step, const NextSeed& next_seed) override;
    void Report(OffPolicyUpdateMetrics& metrics) override;
    const NetworkRoute& Route() const override { return network_route_; }

    DqnNetworks& Networks() { return networks_; }
    int64_t Updates() const { return updates_; }
    int64_t TargetUpdates() const { return target_updates_; }
    // "q"
    bool ExportOptimizers(std::map<std::string, OptimizerState>& out, std::string& error) const;

private:
    void DropFused(const std::exception& e);
    // The Q network's gradients and the step's numbers (nothing changed yet)
    struct Pass {
        DqnStepLosses losses;
        std::map<std::string, Tensor> grads;
    };
    Pass PassFused(const ReplayBuffer::Batch& batch);
    Pass PassLayers(const ReplayBuffer::Batch& batch);

    DqnNetworks networks_;
    size_t choices_;
    DqnConfig config_;
    int64_t target_every_;  // steps of the copies between target updates
    std::unique_ptr<Optimizer> optimizer_;
    // Both fused, or neither
    std::unique_ptr<FusedMlp> q_fused_, q_target_fused_;
    bool adam_together_ = true;
    NetworkRoute network_route_;
    float epsilon_ = 0.0f;        // the share of random choices now (SB3's exploration_rate: 0 before the first step)
    int64_t updates_ = 0;         // gradient steps taken (SB3's _n_updates)
    int64_t target_updates_ = 0;
    // Sums since the last report, on the device
    Tensor loss_sum_, q_sum_;
    int steps_since_report_ = 0;
};

// What a run leaves to play or save.
struct DqnTrainedState {
    DqnNetworks networks;
    std::unique_ptr<VecNormalize> normalizer;
    std::map<std::string, OptimizerState> optimizers;
};

// CopyPolicy copies the Q network (the greedy policy).
class DqnTrainer : public OffPolicyTrainer<DqnLearner, DqnNetworks, DqnConfig, DqnTrainedState> {
public:
    // The environment's actions must be a choice (one index a step).
    DqnTrainer(TrainingEnvironment environment, DqnNetworks networks, DqnConfig config);

private:
    bool CheckSetup(std::string& error) const override;
    std::unique_ptr<DqnLearner> MakeLearner(DqnNetworks networks) override;
};

}  // namespace cyxwiz::rl
