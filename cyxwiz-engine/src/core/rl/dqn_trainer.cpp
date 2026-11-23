#include "dqn_trainer.h"

#include <cyxwiz/fused_mlp.h>
#include <cyxwiz/rl/dqn.h>
#include <cyxwiz/sequential.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cyxwiz::rl {

namespace {

// A seed's bits as a uniform draw in [0, 1)
double Uniform(uint64_t bits) { return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0); }

}  // namespace

// ---------------------------------------------------------------- DqnLearner

DqnLearner::DqnLearner(DqnNetworks networks, size_t observation_size, size_t choices, const DqnConfig& config,
                       int copies)
    : networks_(std::move(networks)), observation_size_(observation_size), choices_(choices), config_(config) {
    auto& n = networks_;
    if (!n.q || !n.q_target) throw std::invalid_argument("DQN needs a Q network and its target.");
    if (choices_ < 1) throw std::invalid_argument("DQN needs at least one choice.");
    // SB3: every max(target_update_interval // n_envs, 1) calls of _on_step
    target_every_ = std::max<int64_t>(config_.target_update_interval / std::max(copies, 1), 1);
    CopyParameters(*n.q, *n.q_target);
    n.q_target->SetTraining(false);
    optimizer_ = OffPolicyAdam(config_.learning_rate);

    // Both networks fused on the device's provider, or neither (one path to report)
    std::string why;
    const auto fuse = [&why](IExecutableModel& model) -> std::unique_ptr<FusedMlp> {
        SequentialModel* sequential = model.AsSequentialModel();
        if (!sequential) {
            why = "not a sequential network";
            return nullptr;
        }
        return FusedMlp::For(*sequential, why);
    };
    q_fused_ = fuse(*n.q);
    if (q_fused_) q_target_fused_ = fuse(*n.q_target);
    if (q_target_fused_) {
        network_path_ = "fused small-network kernels (" + q_fused_->ProviderId() + ")";
    } else {
        q_fused_.reset();
        spdlog::debug("DQN: the networks run as ArrayFire operations ({})", why);
    }
}

DqnLearner::~DqnLearner() = default;

void DqnLearner::DropFused(const std::exception& e) {
    spdlog::warn("DQN: the fused networks failed ({}); ArrayFire operations from now on", e.what());
    q_fused_.reset();
    q_target_fused_.reset();
    network_path_ = std::string("ArrayFire operations (the fused kernels failed: ") + e.what() + ")";
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
    StepOffPolicyOptimizers({{optimizer_.get(), networks_.q.get(), nullptr, &p.grads}}, adam_together_, "DQN",
                            config_.max_grad_norm);
    return p.losses;
}

void DqnLearner::UpdateTarget() {
    PolyakUpdate(*networks_.q, *networks_.q_target, config_.tau);
    ++target_updates_;
}

Tensor DqnLearner::Explore(const Tensor& observations, const NextSeed& next_seed) {
    // SB3's DQN.predict: one draw for the whole vector of copies
    if (Uniform(next_seed()) < epsilon_) {
        const size_t n = observations.Shape().at(0);
        std::vector<float> choices(n);
        for (auto& c : choices) c = static_cast<float>(next_seed() % static_cast<uint64_t>(choices_));
        return Tensor({n}, choices.data());
    }
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

OffPolicySettings LoopSettings(const DqnConfig& c) {
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

DqnTrainer::DqnTrainer(TrainingEnvironment environment, DqnNetworks networks, DqnConfig config)
    : loop_(std::move(environment), LoopSettings(config), "DQN"), networks_(std::move(networks)),
      config_(std::move(config)) {}

DqnTrainer::~DqnTrainer() = default;

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

bool DqnTrainer::TakeTrained(DqnTrainedState& out, std::string& error) {
    if (!learner_ || !loop_.Normalizer()) {
        error = "DQN has not trained a policy.";
        return false;
    }
    std::map<std::string, OptimizerState> optimizers;
    if (!learner_->ExportOptimizers(optimizers, error)) return false;
    out.optimizers = std::move(optimizers);
    out.networks = std::move(learner_->Networks());
    out.normalizer = loop_.TakeNormalizer();
    learner_.reset();
    return true;
}

bool DqnTrainer::CopyPolicy(ActorCopy& out, std::string& error) {
    if (!learner_ || !loop_.Normalizer()) {
        error = "DQN has no policy to copy yet.";
        return false;
    }
    out.parameters.clear();
    for (const auto& [name, tensor] : learner_->Networks().q->GetParameters()) out.parameters[name] = tensor.Clone();
    out.normalizer = std::make_unique<VecNormalize>(*loop_.Normalizer());
    return true;
}

bool DqnTrainer::Train(const std::function<void(const OffPolicyUpdateMetrics&)>& on_report, std::string& error) {
    error.clear();
    if (!CheckSetup(error)) return false;
    const auto& env = loop_.Environment();
    try {
        learner_ = std::make_unique<DqnLearner>(std::move(networks_), static_cast<size_t>(env.info.observation_size),
                                                static_cast<size_t>(env.info.action_count), config_, env.count);
    } catch (const std::exception& e) {
        error = std::string("DQN training failed: ") + e.what();
        return false;
    }
    return loop_.Run(*learner_, on_report, error);
}

}  // namespace cyxwiz::rl
