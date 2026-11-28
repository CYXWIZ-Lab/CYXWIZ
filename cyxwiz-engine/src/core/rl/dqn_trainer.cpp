#include "dqn_trainer.h"

#include <cyxwiz/fused_mlp.h>
#include <cyxwiz/rl/dqn.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cyxwiz::rl {

// ---------------------------------------------------------------- DqnLearner

DqnLearner::DqnLearner(DqnNetworks networks, size_t choices, const DqnConfig& config, int copies)
    : networks_(std::move(networks)), choices_(choices), config_(config) {
    auto& n = networks_;
    if (!n.q || !n.q_target) throw std::invalid_argument("DQN needs a Q network and its target.");
    if (choices_ < 1) throw std::invalid_argument("DQN needs at least one choice.");
    // SB3: every max(target_update_interval // n_envs, 1) calls of _on_step
    target_every_ = std::max<int64_t>(config_.target_update_interval / std::max(copies, 1), 1);
    CopyParameters(*n.q, *n.q_target);
    n.q_target->SetTraining(false);
    optimizer_ = OffPolicyAdam(config_.learning_rate);

    // Both networks fused on the device's provider, or neither (one path to report)
    // The path and its reason are logged by the loop when the run starts
    // (Info, as PPO logs its own)
    FusedMlp::Refusal why;
    q_fused_ = FuseNetwork(*n.q, why);
    if (q_fused_) q_target_fused_ = FuseNetwork(*n.q_target, why);
    if (q_target_fused_) {
        network_route_ = FusedNetworkRoute(*q_fused_);
    } else {
        q_fused_.reset();
        network_route_ = ArrayFireNetworkRoute(why, "Q network");
    }
}

DqnLearner::~DqnLearner() = default;

void DqnLearner::DropFused(const std::exception& e) {
    RethrowProviderRouteError(e);
    spdlog::warn("DQN: the fused networks failed ({}); ArrayFire operations from now on", e.what());
    q_fused_.reset();
    q_target_fused_.reset();
    network_route_ = FailedNetworkRoute(e);
}

Tensor DqnLearner::Act(const Tensor& observations) {
    if (q_fused_) {
        try {
            return DqnGreedyActions(q_fused_->Forward(observations, false));
        } catch (const std::exception& e) {
            DropFused(e);
        }
    }
    networks_.q->SetTraining(false);
    return DqnGreedyActions(networks_.q->Forward(observations));
}

DqnLearner::Pass DqnLearner::PassFused(const ReplayBuffer::Batch& batch) {
    Pass p;
    // The target from the target network's best next Q (no gradient), then
    // the Q network on the batch, its gradient into the chosen columns
    const Tensor y = DqnTarget(q_target_fused_->Forward(batch.next_observations, false), batch.rewards, batch.dones,
                               config_.gamma);
    const DqnLoss loss = DqnLossAndGradient(q_fused_->Forward(batch.observations, true), batch.actions, y);
    p.grads = q_fused_->Backward(loss.grad_q);
    p.losses.loss = loss.loss;
    p.losses.mean_q = loss.mean_q;
    return p;
}

DqnLearner::Pass DqnLearner::PassLayers(const ReplayBuffer::Batch& batch) {
    auto& n = networks_;
    Pass p;
    const Tensor y = DqnTarget(n.q_target->Forward(batch.next_observations), batch.rewards, batch.dones, config_.gamma);
    const DqnLoss loss = DqnLossAndGradient(n.q->Forward(batch.observations), batch.actions, y);
    n.q->Backward(loss.grad_q);
    p.grads = n.q->GetGradients();
    p.losses.loss = loss.loss;
    p.losses.mean_q = loss.mean_q;
    return p;
}

DqnStepLosses DqnLearner::Step(const ReplayBuffer::Batch& batch) {
    networks_.q->SetTraining(true);
    ++updates_;
    // Nothing changes before the optimizer's step, so a failed fused call
    // redoes the pass as layers
    Pass p;
    bool done = false;
    if (q_fused_) {
        try {
            p = PassFused(batch);
            done = true;
        } catch (const std::exception& e) {
            DropFused(e);
        }
    }
    if (!done) p = PassLayers(batch);
    StepOffPolicyOptimizers({{optimizer_.get(), networks_.q.get(), nullptr, &p.grads}}, adam_, "DQN",
                            config_.max_grad_norm);
    return p.losses;
}

void DqnLearner::UpdateTarget() {
    PolyakUpdate(*networks_.q, *networks_.q_target, config_.tau);
    ++target_updates_;
}

bool EpsilonGreedy(std::vector<float>& choices, float epsilon, size_t count, const std::function<uint64_t()>& next) {
    if (!(UniformFromBits(next()) < epsilon)) return false;
    for (auto& c : choices) c = static_cast<float>(next() % static_cast<uint64_t>(count));
    return true;
}

Tensor DqnLearner::Explore(const Tensor& observations, const NextSeed& next_seed) {
    // SB3's DQN.predict: one draw for the whole vector of copies; the greedy
    // choices only when it keeps them
    std::vector<float> choices(observations.Shape().at(0));
    if (EpsilonGreedy(choices, epsilon_, choices_, next_seed)) return Tensor({choices.size()}, choices.data());
    return Act(observations);
}

void DqnLearner::AfterStep(int64_t timesteps, int64_t vector_steps) {
    if (vector_steps % target_every_ == 0) UpdateTarget();
    const double progress_remaining =
        1.0 - static_cast<double>(timesteps) / static_cast<double>(std::max<int64_t>(config_.total_timesteps, 1));
    epsilon_ = DqnExplorationRate(config_.exploration_initial_eps, config_.exploration_final_eps,
                                  config_.exploration_fraction, progress_remaining);
}

void DqnLearner::Learn(const ReplayBuffer::Batch& batch, int /*gradient_step*/, const NextSeed& /*next_seed*/) {
    const DqnStepLosses l = Step(batch);
    loss_sum_ = steps_since_report_ == 0 ? l.loss : loss_sum_ + l.loss;
    q_sum_ = steps_since_report_ == 0 ? l.mean_q : q_sum_ + l.mean_q;
    ++steps_since_report_;
}

void DqnLearner::Report(OffPolicyUpdateMetrics& metrics) {
    if (steps_since_report_ > 0) {
        const float k = static_cast<float>(steps_since_report_);
        metrics.critic_loss = FirstValue(loss_sum_) / k;
        metrics.mean_q = FirstValue(q_sum_) / k;
    }
    metrics.exploration_rate = epsilon_;
    steps_since_report_ = 0;
}

bool DqnLearner::ExportOptimizers(std::map<std::string, OptimizerState>& out, std::string& error) const {
    std::map<std::string, OptimizerState> states;
    if (!optimizer_->ExportState(states["q"], error)) return false;
    out = std::move(states);
    return true;
}

// ---------------------------------------------------------------- DqnTrainer

DqnTrainer::DqnTrainer(TrainingEnvironment environment, DqnNetworks networks, DqnConfig config)
    : OffPolicyTrainer(std::move(environment), std::move(networks), std::move(config), "DQN", &DqnNetworks::q) {}

bool DqnTrainer::CheckSetup(std::string& error) const {
    const auto& c = config_;
    const auto share = [](float x) { return x >= 0.0f && x <= 1.0f; };
    if (!networks_.q || !networks_.q_target)
        error = "DQN needs a Q network and its target.";
    else if (!CheckOffPolicySetup(loop_.Environment(), loop_.Settings(), "DQN", error, true))
        return false;
    else if (c.target_update_interval < 1)
        error = "DQN's counts must each be at least 1 (the random steps first at least 0).";
    else if (!(c.learning_rate > 0.0f) || !(c.gamma > 0.0f && c.gamma <= 1.0f) || !(c.tau > 0.0f && c.tau <= 1.0f) ||
             !(c.max_grad_norm > 0.0f))
        error = "DQN's learning rate and largest gradient norm must be above 0, the discount and tau in (0, 1].";
    else if (!share(c.exploration_fraction) || !share(c.exploration_initial_eps) || !share(c.exploration_final_eps))
        error = "DQN's exploration start, end and share of the run must each be from 0 to 1.";
    return error.empty();
}

std::unique_ptr<DqnLearner> DqnTrainer::MakeLearner(DqnNetworks networks) {
    const auto& env = loop_.Environment();
    return std::make_unique<DqnLearner>(std::move(networks), static_cast<size_t>(env.info.action_count), config_,
                                        env.count);
}

}  // namespace cyxwiz::rl
