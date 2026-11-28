#include "sac_trainer.h"

#include <cyxwiz/fused_mlp.h>
#include <cyxwiz/rl/sac.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cyxwiz::rl {

// ---------------------------------------------------------------- SacLearner

SacLearner::SacLearner(SacNetworks networks, size_t observation_size, size_t action_size, const SacConfig& config)
    : networks_(std::move(networks)), observation_size_(observation_size), action_size_(action_size),
      config_(config) {
    if (!networks_.actor || !networks_.q1 || !networks_.q2 || !networks_.q1_target || !networks_.q2_target)
        throw std::invalid_argument("SAC needs an actor, two critics and their targets.");
    if (config_.target_entropy_auto) config_.target_entropy = -static_cast<float>(action_size_);
    CopyParameters(*networks_.q1, *networks_.q1_target);
    CopyParameters(*networks_.q2, *networks_.q2_target);
    networks_.q1_target->SetTraining(false);
    networks_.q2_target->SetTraining(false);
    actor_optimizer_ = OffPolicyAdam(config_.learning_rate);
    q1_optimizer_ = OffPolicyAdam(config_.learning_rate);  // SB3: one Adam over both critics, the same per parameter
    q2_optimizer_ = OffPolicyAdam(config_.learning_rate);
    // SB3: log_ent_coef starts at log(ent_coef) when learned
    const float start = std::log(config_.ent_coef > 0.0f ? config_.ent_coef : 1.0f);
    log_ent_coef_ = Tensor({1}, &start);
    fixed_ent_coef_ = Tensor({1}, &config_.ent_coef);
    if (config_.ent_coef_auto) ent_coef_optimizer_ = OffPolicyAdam(config_.learning_rate);

    // The five networks fused on the device's provider, or none of them (one
    // path to report)
    // The path and its reason are logged by the loop when the run starts
    // (Info, as PPO logs its own)
    FusedMlp::Refusal why;
    const char* refused = "actor";
    actor_fused_ = FuseNetwork(*networks_.actor, why);
    if (actor_fused_) {
        refused = "critic";
        q1_fused_ = FuseNetwork(*networks_.q1, why);
    }
    if (q1_fused_) q2_fused_ = FuseNetwork(*networks_.q2, why);
    if (q2_fused_) q1_target_fused_ = FuseNetwork(*networks_.q1_target, why);
    if (q1_target_fused_) q2_target_fused_ = FuseNetwork(*networks_.q2_target, why);
    if (q2_target_fused_) {
        network_route_ = FusedNetworkRoute(*actor_fused_);
    } else {
        actor_fused_.reset();
        q1_fused_.reset();
        q2_fused_.reset();
        q1_target_fused_.reset();
        network_route_ = ArrayFireNetworkRoute(why, refused);
    }
}

SacLearner::~SacLearner() = default;

void SacLearner::DropFused(const std::exception& e) {
    RethrowProviderRouteError(e);
    spdlog::warn("SAC: the fused networks failed ({}); ArrayFire operations from now on", e.what());
    actor_fused_.reset();
    q1_fused_.reset();
    q2_fused_.reset();
    q1_target_fused_.reset();
    q2_target_fused_.reset();
    network_route_ = FailedNetworkRoute(e);
}

Tensor SacActorActions(IExecutableModel& actor, const Tensor& observations, size_t action_size, const Tensor* noise) {
    actor.SetTraining(false);
    const Tensor out = actor.Forward(observations);
    const int a = static_cast<int>(action_size);
    const Tensor mean = out.Slice(1, 0, a);
    if (!noise) return SacDeterministicAction(mean);
    return SacSquash(mean, out.Slice(1, a, 2 * a), *noise).actions;
}

Tensor SacLearner::Act(const Tensor& observations, const Tensor* noise) {
    if (actor_fused_) {
        try {
            const Tensor out = actor_fused_->Forward(observations, false);
            const int a = static_cast<int>(action_size_);
            const Tensor mean = out.Slice(1, 0, a);
            if (!noise) return SacDeterministicAction(mean);
            return SacSquash(mean, out.Slice(1, a, 2 * a), *noise).actions;
        } catch (const std::exception& e) {
            DropFused(e);
        }
    }
    return SacActorActions(*networks_.actor, observations, action_size_, noise);
}

SacLearner::CriticPass SacLearner::CriticsFused(const ReplayBuffer::Batch& batch, const Tensor& noise_pi,
                                                const Tensor& noise_next, const Tensor& alpha) {
    const int a = static_cast<int>(action_size_);
    CriticPass c;
    // The critics' target, from the next observations (no gradient)
    const Tensor next_out = actor_fused_->Forward(batch.next_observations, false);
    const SacSquashed next = SacSquash(next_out.Slice(1, 0, a), next_out.Slice(1, a, 2 * a), noise_next);
    const Tensor next_input = Tensor::Cat({batch.next_observations, next.actions}, 1);
    const auto [q1_next, q2_next] = FusedMlp::ForwardPair(*q1_target_fused_, *q2_target_fused_, next_input, false);
    const Tensor y = SacTarget(q1_next, q2_next, next.log_prob, batch.rewards, batch.dones, alpha, config_.gamma);
    // The critics on the batch's own actions, both in one launch each way
    const Tensor input = Tensor::Cat({batch.observations, batch.actions}, 1);
    const auto [q1, q2] = FusedMlp::ForwardPair(*q1_fused_, *q2_fused_, input, true);
    const SacCriticLoss critic = SacCriticLossAndGradients(q1, q2, y);
    auto grads = FusedMlp::BackwardPair(*q1_fused_, critic.grad_q1, *q2_fused_, critic.grad_q2);
    c.critic_loss = critic.loss;
    c.q1_grads = std::move(grads.first);
    c.q2_grads = std::move(grads.second);
    // The actor's draw on the batch, kept for its backward
    const Tensor out = actor_fused_->Forward(batch.observations, true);
    c.mean = out.Slice(1, 0, a);
    c.log_std = out.Slice(1, a, 2 * a);
    const SacSquashed pi = SacSquash(c.mean, c.log_std, noise_pi);
    c.actions = pi.actions;
    c.log_prob = pi.log_prob;
    return c;
}

SacLearner::CriticPass SacLearner::CriticsLayers(const ReplayBuffer::Batch& batch, const Tensor& noise_pi,
                                                 const Tensor& noise_next, const Tensor& alpha) {
    auto& n = networks_;
    const int a = static_cast<int>(action_size_);
    CriticPass c;
    // The critics' target, from the next observations (no gradient). The actor
    // runs here first so that its last forward is the batch's own below.
    const Tensor next_out = n.actor->Forward(batch.next_observations);
    const SacSquashed next = SacSquash(next_out.Slice(1, 0, a), next_out.Slice(1, a, 2 * a), noise_next);
    const Tensor next_input = Tensor::Cat({batch.next_observations, next.actions}, 1);
    const Tensor y = SacTarget(n.q1_target->Forward(next_input), n.q2_target->Forward(next_input), next.log_prob,
                               batch.rewards, batch.dones, alpha, config_.gamma);
    // The critics on the batch's own actions
    const Tensor input = Tensor::Cat({batch.observations, batch.actions}, 1);
    const SacCriticLoss critic = SacCriticLossAndGradients(n.q1->Forward(input), n.q2->Forward(input), y);
    n.q1->Backward(critic.grad_q1);
    c.q1_grads = n.q1->GetGradients();
    n.q2->Backward(critic.grad_q2);
    c.q2_grads = n.q2->GetGradients();
    c.critic_loss = critic.loss;
    // The actor's draw on the batch (the critics' update does not change it)
    const Tensor out = n.actor->Forward(batch.observations);
    c.mean = out.Slice(1, 0, a);
    c.log_std = out.Slice(1, a, 2 * a);
    const SacSquashed pi = SacSquash(c.mean, c.log_std, noise_pi);
    c.actions = pi.actions;
    c.log_prob = pi.log_prob;
    c.actor_cached = true;
    return c;
}

SacLearner::ActorPass SacLearner::ActorFused(const ReplayBuffer::Batch& batch, const CriticPass& c,
                                             const Tensor& noise_pi, const Tensor& alpha) {
    const int a = static_cast<int>(action_size_);
    const int o = static_cast<int>(observation_size_);
    ActorPass p;
    // The updated critics on the actor's draw; its action's gradient comes
    // back through their input (their parameters' gradients are not computed)
    const Tensor pi_input = Tensor::Cat({batch.observations, c.actions}, 1);
    const auto [q1_pi, q2_pi] = FusedMlp::ForwardPair(*q1_fused_, *q2_fused_, pi_input, true);
    const SacActorLoss actor = SacActorLossAndGradients(q1_pi, q2_pi, c.log_prob, alpha);
    const auto [through_q1, through_q2] =
        FusedMlp::InputGradientPair(*q1_fused_, actor.grad_q1, *q2_fused_, actor.grad_q2);
    const SacSquashGradients g = SacSquashBackward(c.mean, c.log_std, noise_pi,
                                                   through_q1.Slice(1, o, o + a) + through_q2.Slice(1, o, o + a),
                                                   actor.grad_log_prob);
    // The actor's kept forward is the batch's observations (CriticsFused)
    p.grads = actor_fused_->Backward(Tensor::Cat({g.grad_mean, g.grad_log_std}, 1));
    p.loss = actor.loss;
    p.mean_q = actor.mean_min_q;
    return p;
}

SacLearner::ActorPass SacLearner::ActorLayers(const ReplayBuffer::Batch& batch, const CriticPass& c,
                                              const Tensor& noise_pi, const Tensor& alpha) {
    auto& n = networks_;
    const int a = static_cast<int>(action_size_);
    const int o = static_cast<int>(observation_size_);
    ActorPass p;
    // The actor against the updated critics: its action's gradient comes back
    // through each critic's input
    const Tensor pi_input = Tensor::Cat({batch.observations, c.actions}, 1);
    const Tensor q1_pi = n.q1->Forward(pi_input);
    const Tensor q2_pi = n.q2->Forward(pi_input);
    const SacActorLoss actor = SacActorLossAndGradients(q1_pi, q2_pi, c.log_prob, alpha);
    // Each critic's backward runs over its own last forward (pi_input); the
    // critics' parameter gradients it leaves are overwritten by their next step
    const Tensor through_q1 = n.q1->Backward(actor.grad_q1).Slice(1, o, o + a);
    const Tensor through_q2 = n.q2->Backward(actor.grad_q2).Slice(1, o, o + a);
    const SacSquashGradients g = SacSquashBackward(c.mean, c.log_std, noise_pi, through_q1 + through_q2,
                                                   actor.grad_log_prob);
    // The actor's backward needs its forward of the batch's observations
    if (!c.actor_cached) n.actor->Forward(batch.observations);
    n.actor->Backward(Tensor::Cat({g.grad_mean, g.grad_log_std}, 1));
    p.grads = n.actor->GetGradients();
    p.loss = actor.loss;
    p.mean_q = actor.mean_min_q;
    return p;
}

void SacLearner::StepOptimizers(const std::vector<OffPolicyUpdate>& updates) {
    // The entropy weight goes as a loose tensor and comes back the same way
    std::map<std::string, Tensor> ent{{"log_ent_coef", log_ent_coef_}};
    std::vector<OffPolicyUpdate> all = updates;
    for (auto& u : all)
        if (!u.model) u.parameters = &ent;
    StepOffPolicyOptimizers(all, adam_, "SAC");
    log_ent_coef_ = ent.at("log_ent_coef");
}

SacStepLosses SacLearner::Step(const ReplayBuffer::Batch& batch, const Tensor& noise_pi, const Tensor& noise_next,
                               bool update_targets) {
    auto& n = networks_;
    n.actor->SetTraining(true);
    n.q1->SetTraining(true);
    n.q2->SetTraining(true);
    SacStepLosses losses;
    // The weight this step uses: alpha before its own update (SB3 detaches it first)
    const Tensor alpha = config_.ent_coef_auto ? log_ent_coef_.Exp() : fixed_ent_coef_;
    losses.ent_coef = alpha;

    // ---- The critics' gradients and the actor's draw (nothing changes yet, so
    // a failed fused call redoes them as layers)
    CriticPass c;
    bool done = false;
    if (actor_fused_) {
        try {
            c = CriticsFused(batch, noise_pi, noise_next, alpha);
            done = true;
        } catch (const std::exception& e) {
            DropFused(e);
        }
    }
    if (!done) c = CriticsLayers(batch, noise_pi, noise_next, alpha);
    losses.critic_loss = c.critic_loss;

    // ---- The entropy weight, then the critics (SB3's order; one launch
    // together when the provider steps Adam)
    std::map<std::string, Tensor> ent_grads;
    std::vector<OffPolicyUpdate> updates;
    if (config_.ent_coef_auto) {
        const SacEntCoefLoss ent = SacEntCoefLossAndGradient(log_ent_coef_, c.log_prob, config_.target_entropy);
        losses.ent_coef_loss = ent.loss;
        ent_grads["log_ent_coef"] = ent.grad_log_ent_coef;
        updates.push_back({ent_coef_optimizer_.get(), nullptr, nullptr, &ent_grads});
    }
    updates.push_back({q1_optimizer_.get(), n.q1.get(), nullptr, &c.q1_grads});
    updates.push_back({q2_optimizer_.get(), n.q2.get(), nullptr, &c.q2_grads});
    StepOptimizers(updates);

    // ---- The actor against the updated critics
    ActorPass p;
    done = false;
    if (actor_fused_) {
        try {
            p = ActorFused(batch, c, noise_pi, alpha);
            done = true;
        } catch (const std::exception& e) {
            DropFused(e);
        }
    }
    if (!done) p = ActorLayers(batch, c, noise_pi, alpha);
    losses.actor_loss = p.loss;
    losses.mean_q = p.mean_q;
    StepOptimizers({{actor_optimizer_.get(), n.actor.get(), nullptr, &p.grads}});

    if (update_targets) {
        PolyakUpdate(*n.q1, *n.q1_target, config_.tau);
        PolyakUpdate(*n.q2, *n.q2_target, config_.tau);
    }
    return losses;
}

bool SacLearner::ExportOptimizers(std::map<std::string, OptimizerState>& out, std::string& error) const {
    std::map<std::string, OptimizerState> states;
    if (!actor_optimizer_->ExportState(states["actor"], error) || !q1_optimizer_->ExportState(states["q1"], error) ||
        !q2_optimizer_->ExportState(states["q2"], error))
        return false;
    if (ent_coef_optimizer_ && !ent_coef_optimizer_->ExportState(states["ent_coef"], error)) return false;
    out = std::move(states);
    return true;
}

Tensor SacLearner::Explore(const Tensor& observations, const NextSeed& next_seed) {
    const Tensor noise = SacNoise(observations.Shape()[0], action_size_, next_seed());
    return Act(observations, &noise);
}

void SacLearner::Learn(const ReplayBuffer::Batch& batch, int gradient_step, const NextSeed& next_seed) {
    const size_t rows = batch.observations.Shape()[0];
    const Tensor noise_pi = SacNoise(rows, action_size_, next_seed());
    const Tensor noise_next = SacNoise(rows, action_size_, next_seed());
    // SB3 counts the target interval within each train() call
    const SacStepLosses l = Step(batch, noise_pi, noise_next, gradient_step % config_.target_update_interval == 0);
    const bool first = since_report_ == 0;
    critic_sum_ = first ? l.critic_loss : critic_sum_ + l.critic_loss;
    actor_sum_ = first ? l.actor_loss : actor_sum_ + l.actor_loss;
    alpha_sum_ = first ? l.ent_coef : alpha_sum_ + l.ent_coef;
    q_sum_ = first ? l.mean_q : q_sum_ + l.mean_q;
    if (config_.ent_coef_auto) alpha_loss_sum_ = first ? l.ent_coef_loss : alpha_loss_sum_ + l.ent_coef_loss;
    ++since_report_;
}

void SacLearner::Report(OffPolicyUpdateMetrics& metrics) {
    if (since_report_ > 0) {
        const float k = static_cast<float>(since_report_);
        metrics.critic_loss = FirstValue(critic_sum_) / k;
        metrics.actor_loss = FirstValue(actor_sum_) / k;
        metrics.ent_coef = FirstValue(alpha_sum_) / k;
        metrics.mean_q = FirstValue(q_sum_) / k;
        metrics.ent_coef_loss = config_.ent_coef_auto ? FirstValue(alpha_loss_sum_) / k : 0.0f;
    } else {
        metrics.ent_coef = config_.ent_coef_auto ? std::exp(FirstValue(log_ent_coef_)) : config_.ent_coef;
    }
    since_report_ = 0;
}

// ---------------------------------------------------------------- SacTrainer

SacTrainer::SacTrainer(TrainingEnvironment environment, SacNetworks networks, SacConfig config)
    : OffPolicyTrainer(std::move(environment), std::move(networks), std::move(config), "SAC", &SacNetworks::actor) {}

bool SacTrainer::CheckSetup(std::string& error) const {
    const auto& c = config_;
    if (!networks_.actor || !networks_.q1 || !networks_.q2 || !networks_.q1_target || !networks_.q2_target)
        error = "SAC needs an actor, two critics and their targets.";
    else if (!CheckOffPolicySetup(loop_.Environment(), loop_.Settings(), "SAC", error))
        return false;
    else if (c.target_update_interval < 1)
        error = "SAC's counts must each be at least 1 (the random steps first at least 0).";
    else if (!(c.learning_rate > 0.0f) || !(c.gamma > 0.0f && c.gamma <= 1.0f) || !(c.tau > 0.0f && c.tau <= 1.0f) ||
             !(c.ent_coef > 0.0f))
        error = "SAC's learning rate and entropy weight must be above 0, the discount and tau in (0, 1].";
    return error.empty();
}

std::unique_ptr<SacLearner> SacTrainer::MakeLearner(SacNetworks networks) {
    const auto& info = loop_.Environment().info;
    return std::make_unique<SacLearner>(std::move(networks), static_cast<size_t>(info.observation_size),
                                        static_cast<size_t>(info.action_size), config_);
}

void SacTrainer::TakeMore(SacLearner& learner, SacTrainedState& out) { out.log_ent_coef = learner.LogEntCoef(); }

}  // namespace cyxwiz::rl
