#include "td3_trainer.h"

#include <cyxwiz/fused_mlp.h>
#include <cyxwiz/rl/sac.h>
#include <cyxwiz/rl/td3.h>

#include <spdlog/spdlog.h>

#include <cmath>
#include <stdexcept>

namespace cyxwiz::rl {

// ---------------------------------------------------------------- Td3Learner

Td3Learner::Td3Learner(Td3Networks networks, size_t observation_size, size_t action_size, const Td3Config& config)
    : networks_(std::move(networks)), observation_size_(observation_size), action_size_(action_size),
      config_(config) {
    auto& n = networks_;
    if (!n.actor || !n.actor_target || !n.q1 || !n.q2 || !n.q1_target || !n.q2_target)
        throw std::invalid_argument("TD3 needs an actor, two critics and a target of each.");
    CopyParameters(*n.actor, *n.actor_target);
    CopyParameters(*n.q1, *n.q1_target);
    CopyParameters(*n.q2, *n.q2_target);
    n.actor_target->SetTraining(false);
    n.q1_target->SetTraining(false);
    n.q2_target->SetTraining(false);
    actor_optimizer_ = OffPolicyAdam(config_.learning_rate);
    q1_optimizer_ = OffPolicyAdam(config_.learning_rate);  // SB3: one Adam over both critics, the same per parameter
    q2_optimizer_ = OffPolicyAdam(config_.learning_rate);

    // The six networks fused on the device's provider, or none of them (one
    // path to report)
    std::string why;
    actor_fused_ = FuseNetwork(*n.actor, why);
    if (actor_fused_) actor_target_fused_ = FuseNetwork(*n.actor_target, why);
    if (actor_target_fused_) q1_fused_ = FuseNetwork(*n.q1, why);
    if (q1_fused_) q2_fused_ = FuseNetwork(*n.q2, why);
    if (q2_fused_) q1_target_fused_ = FuseNetwork(*n.q1_target, why);
    if (q1_target_fused_) q2_target_fused_ = FuseNetwork(*n.q2_target, why);
    if (q2_target_fused_) {
        network_path_ = "fused small-network kernels (" + actor_fused_->ProviderId() + ")";
    } else {
        actor_fused_.reset();
        actor_target_fused_.reset();
        q1_fused_.reset();
        q2_fused_.reset();
        q1_target_fused_.reset();
        spdlog::debug("TD3: the networks run as ArrayFire operations ({})", why);
    }
}

Td3Learner::~Td3Learner() = default;

void Td3Learner::DropFused(const std::exception& e) {
    RethrowProviderRouteError(e);
    spdlog::warn("TD3: the fused networks failed ({}); ArrayFire operations from now on", e.what());
    actor_fused_.reset();
    actor_target_fused_.reset();
    q1_fused_.reset();
    q2_fused_.reset();
    q1_target_fused_.reset();
    q2_target_fused_.reset();
    network_path_ = std::string("ArrayFire operations (the fused kernels failed: ") + e.what() + ")";
}

Tensor Td3Learner::Act(const Tensor& observations) {
    if (actor_fused_) {
        try {
            return actor_fused_->Forward(observations, false);
        } catch (const std::exception& e) {
            DropFused(e);
        }
    }
    networks_.actor->SetTraining(false);
    return networks_.actor->Forward(observations);
}

Tensor Td3Learner::Smoothed(const Tensor& target_actions, const Tensor& noise) const {
    return Td3TargetActions(target_actions, noise, config_.target_policy_noise, config_.target_noise_clip);
}

Td3Learner::CriticPass Td3Learner::CriticsFused(const ReplayBuffer::Batch& batch, const Tensor& noise) {
    CriticPass c;
    // The critics' target, from the next observations and the target actor's
    // smoothed actions (no gradient)
    const Tensor next_actions = Smoothed(actor_target_fused_->Forward(batch.next_observations, false), noise);
    const Tensor next_input = Tensor::Cat({batch.next_observations, next_actions}, 1);
    const auto [q1_next, q2_next] = FusedMlp::ForwardPair(*q1_target_fused_, *q2_target_fused_, next_input, false);
    const Tensor y = Td3Target(q1_next, q2_next, batch.rewards, batch.dones, config_.gamma);
    // The critics on the batch's own actions, both in one launch each way
    const Tensor input = Tensor::Cat({batch.observations, batch.actions}, 1);
    const auto [q1, q2] = FusedMlp::ForwardPair(*q1_fused_, *q2_fused_, input, true);
    const Td3CriticLoss critic = Td3CriticLossAndGradients(q1, q2, y);
    auto grads = FusedMlp::BackwardPair(*q1_fused_, critic.grad_q1, *q2_fused_, critic.grad_q2);
    c.loss = critic.loss;
    c.q1_grads = std::move(grads.first);
    c.q2_grads = std::move(grads.second);
    return c;
}

Td3Learner::CriticPass Td3Learner::CriticsLayers(const ReplayBuffer::Batch& batch, const Tensor& noise) {
    auto& n = networks_;
    CriticPass c;
    const Tensor next_actions = Smoothed(n.actor_target->Forward(batch.next_observations), noise);
    const Tensor next_input = Tensor::Cat({batch.next_observations, next_actions}, 1);
    const Tensor y = Td3Target(n.q1_target->Forward(next_input), n.q2_target->Forward(next_input), batch.rewards,
                               batch.dones, config_.gamma);
    const Tensor input = Tensor::Cat({batch.observations, batch.actions}, 1);
    const Td3CriticLoss critic = Td3CriticLossAndGradients(n.q1->Forward(input), n.q2->Forward(input), y);
    n.q1->Backward(critic.grad_q1);
    c.q1_grads = n.q1->GetGradients();
    n.q2->Backward(critic.grad_q2);
    c.q2_grads = n.q2->GetGradients();
    c.loss = critic.loss;
    return c;
}

Td3Learner::ActorPass Td3Learner::ActorFused(const ReplayBuffer::Batch& batch) {
    const int a = static_cast<int>(action_size_);
    const int o = static_cast<int>(observation_size_);
    ActorPass p;
    // The actor's actions, kept for its backward; the updated Q network 1 on
    // them, its input's gradient carrying the action's
    const Tensor actions = actor_fused_->Forward(batch.observations, true);
    const Tensor q1_pi = q1_fused_->Forward(Tensor::Cat({batch.observations, actions}, 1), true);
    const Td3ActorLoss actor = Td3ActorLossAndGradient(q1_pi);
    const Tensor through_q1 = q1_fused_->InputGradient(actor.grad_q1).Slice(1, o, o + a);
    p.grads = actor_fused_->Backward(through_q1);
    p.loss = actor.loss;
    p.mean_q = actor.mean_q;
    return p;
}

Td3Learner::ActorPass Td3Learner::ActorLayers(const ReplayBuffer::Batch& batch) {
    auto& n = networks_;
    const int a = static_cast<int>(action_size_);
    const int o = static_cast<int>(observation_size_);
    ActorPass p;
    const Tensor actions = n.actor->Forward(batch.observations);
    const Tensor q1_pi = n.q1->Forward(Tensor::Cat({batch.observations, actions}, 1));
    const Td3ActorLoss actor = Td3ActorLossAndGradient(q1_pi);
    // Q network 1's backward runs over its last forward (the actor's
    // actions); the parameter gradients it leaves are overwritten by its next step
    const Tensor through_q1 = n.q1->Backward(actor.grad_q1).Slice(1, o, o + a);
    n.actor->Backward(through_q1);
    p.grads = n.actor->GetGradients();
    p.loss = actor.loss;
    p.mean_q = actor.mean_q;
    return p;
}

Td3StepLosses Td3Learner::Step(const ReplayBuffer::Batch& batch, const Tensor& noise) {
    auto& n = networks_;
    n.actor->SetTraining(true);
    n.q1->SetTraining(true);
    n.q2->SetTraining(true);
    ++updates_;
    Td3StepLosses losses;

    // ---- The critics (nothing changes before their step, so a failed fused
    // call redoes them as layers)
    CriticPass c;
    bool done = false;
    if (actor_fused_) {
        try {
            c = CriticsFused(batch, noise);
            done = true;
        } catch (const std::exception& e) {
            DropFused(e);
        }
    }
    if (!done) c = CriticsLayers(batch, noise);
    losses.critic_loss = c.loss;
    StepOffPolicyOptimizers({{q1_optimizer_.get(), n.q1.get(), nullptr, &c.q1_grads},
                             {q2_optimizer_.get(), n.q2.get(), nullptr, &c.q2_grads}},
                            adam_together_, "TD3");

    // ---- Every policy_delay-th step: the actor against the updated Q network
    // 1, then every target (SB3's order)
    if (updates_ % config_.policy_delay != 0) return losses;
    ActorPass p;
    done = false;
    if (actor_fused_) {
        try {
            p = ActorFused(batch);
            done = true;
        } catch (const std::exception& e) {
            DropFused(e);
        }
    }
    if (!done) p = ActorLayers(batch);
    StepOffPolicyOptimizers({{actor_optimizer_.get(), n.actor.get(), nullptr, &p.grads}}, adam_together_, "TD3");
    PolyakUpdate(*n.q1, *n.q1_target, config_.tau);
    PolyakUpdate(*n.q2, *n.q2_target, config_.tau);
    PolyakUpdate(*n.actor, *n.actor_target, config_.tau);
    losses.policy_updated = true;
    losses.actor_loss = p.loss;
    losses.mean_q = p.mean_q;
    return losses;
}

Tensor Td3Learner::Explore(const Tensor& observations, const NextSeed& next_seed) {
    const Tensor actions = Act(observations);
    return Warmup(actions, next_seed);
}

Tensor Td3Learner::Warmup(const Tensor& actions, const NextSeed& next_seed) {
    if (!config_.action_noise) return actions;
    const Tensor noise = SacNoise(actions.Shape()[0], action_size_, next_seed());
    return Td3ExploreActions(actions, noise, config_.action_noise_std);
}

void Td3Learner::Learn(const ReplayBuffer::Batch& batch, int /*gradient_step*/, const NextSeed& next_seed) {
    const Tensor noise = SacNoise(batch.observations.Shape()[0], action_size_, next_seed());
    const Td3StepLosses l = Step(batch, noise);
    critic_sum_ = critic_steps_ == 0 ? l.critic_loss : critic_sum_ + l.critic_loss;
    ++critic_steps_;
    if (l.policy_updated) {
        actor_sum_ = actor_steps_ == 0 ? l.actor_loss : actor_sum_ + l.actor_loss;
        q_sum_ = actor_steps_ == 0 ? l.mean_q : q_sum_ + l.mean_q;
        ++actor_steps_;
    }
}

void Td3Learner::Report(OffPolicyUpdateMetrics& metrics) {
    if (critic_steps_ > 0) metrics.critic_loss = FirstValue(critic_sum_) / static_cast<float>(critic_steps_);
    if (actor_steps_ > 0) {
        const float k = static_cast<float>(actor_steps_);
        metrics.actor_loss = FirstValue(actor_sum_) / k;
        metrics.mean_q = FirstValue(q_sum_) / k;
    }
    critic_steps_ = actor_steps_ = 0;
}

bool Td3Learner::ExportOptimizers(std::map<std::string, OptimizerState>& out, std::string& error) const {
    std::map<std::string, OptimizerState> states;
    if (!actor_optimizer_->ExportState(states["actor"], error) || !q1_optimizer_->ExportState(states["q1"], error) ||
        !q2_optimizer_->ExportState(states["q2"], error))
        return false;
    out = std::move(states);
    return true;
}

// ---------------------------------------------------------------- Td3Trainer

Td3Trainer::Td3Trainer(TrainingEnvironment environment, Td3Networks networks, Td3Config config)
    : OffPolicyTrainer(std::move(environment), std::move(networks), std::move(config), "TD3", &Td3Networks::actor) {}

bool Td3Trainer::CheckSetup(std::string& error) const {
    const auto& c = config_;
    const auto& n = networks_;
    if (!n.actor || !n.actor_target || !n.q1 || !n.q2 || !n.q1_target || !n.q2_target)
        error = "TD3 needs an actor, two critics and a target of each.";
    else if (!CheckOffPolicySetup(loop_.Environment(), loop_.Settings(), "TD3", error))
        return false;
    else if (c.policy_delay < 1)
        error = "TD3's counts must each be at least 1 (the random steps first at least 0).";
    else if (!(c.learning_rate > 0.0f) || !(c.gamma > 0.0f && c.gamma <= 1.0f) || !(c.tau > 0.0f && c.tau <= 1.0f))
        error = "TD3's learning rate must be above 0, the discount and tau in (0, 1].";
    else if ((c.action_noise && !(c.action_noise_std > 0.0f)) || !(c.target_policy_noise >= 0.0f) ||
             !(c.target_noise_clip >= 0.0f))
        error = "TD3's noise std must be above 0, the target smoothing noise and its clip at least 0.";
    return error.empty();
}

std::unique_ptr<Td3Learner> Td3Trainer::MakeLearner(Td3Networks networks) {
    const auto& info = loop_.Environment().info;
    return std::make_unique<Td3Learner>(std::move(networks), static_cast<size_t>(info.observation_size),
                                        static_cast<size_t>(info.action_size), config_);
}

}  // namespace cyxwiz::rl
