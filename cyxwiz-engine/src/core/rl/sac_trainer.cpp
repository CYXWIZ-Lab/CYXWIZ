#include "sac_trainer.h"

#include "../optimizer_step.h"

#include <cyxwiz/fused_mlp.h>
#include <cyxwiz/optimizers/adam.h>
#include <cyxwiz/rl/sac.h>
#include <cyxwiz/sequential.h>

#include <arrayfire.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <thread>
#include <vector>

namespace cyxwiz::rl {

namespace {

// SplitMix64 for the trainer's own draws (noise, action and sample seeds)
class Random {
public:
    explicit Random(uint64_t seed) : state_(seed) {}
    uint64_t Next() {
        uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

private:
    uint64_t state_;
};

std::vector<float> Host(const Tensor& t) {
    const float* data = t.ReadData<float>();
    return std::vector<float>(data, data + t.NumElements());
}

Tensor Rows(const std::vector<float>& values, size_t rows, size_t width) {
    return Tensor({rows, width}, values.data());
}

float First(const Tensor& t) { return Host(t).at(0); }

std::unique_ptr<Optimizer> Adam(float learning_rate) {
    // SB3 SAC's optimizer: th.optim.Adam with its defaults
    return std::make_unique<AdamOptimizer>(learning_rate, 0.9, 0.999, 1e-8);
}

// The critics' own parameters into the targets
void CopyParameters(IExecutableModel& from, IExecutableModel& to) {
    std::map<std::string, Tensor> copy;
    for (const auto& [name, tensor] : from.GetParameters()) copy[name] = tensor.Clone();
    to.SetParameters(copy);
}

void Polyak(IExecutableModel& critic, IExecutableModel& target, float tau) {
    auto params = critic.GetParameters();
    auto targets = target.GetParameters();
    for (auto& [name, t] : targets) {
        const auto it = params.find(name);
        if (it == params.end()) throw std::runtime_error("SAC: a target critic's parameter " + name + " has no critic one");
        t = PolyakAverage(it->second, t, tau);
    }
    target.SetParameters(targets);
}

// Observations normalised with the statistics as they are now, as SB3's
// replay buffer does at sample time with a VecNormalize
Tensor NormalizeOnDevice(const Tensor& observations, const VecNormalize& normalizer) {
    const auto& stats = normalizer.ObservationStats();
    const auto& s = normalizer.Settings();
    const size_t rows = observations.Shape()[0], width = observations.Shape()[1];
    std::vector<float> mean(width), scale(width);
    for (size_t i = 0; i < width; ++i) {
        mean[i] = static_cast<float>(stats.mean[i]);
        scale[i] = static_cast<float>(1.0 / std::sqrt(stats.var[i] + s.epsilon));
    }
    const unsigned r = static_cast<unsigned>(rows);
    const af::array m = af::tile(af::array(1, static_cast<dim_t>(width), mean.data()), r, 1);
    const af::array k = af::tile(af::array(1, static_cast<dim_t>(width), scale.data()), r, 1);
    af::array out = af::clamp((observations.GetSemanticArray() - m) * k, -static_cast<double>(s.clip_observations),
                              static_cast<double>(s.clip_observations));
    out.eval();
    return Tensor::FromSemanticArray(out, {rows, width});
}

}  // namespace

void InitializeLikeTorchLinear(IExecutableModel& model) {
    auto params = model.GetParameters();
    // fan_in of each layer: its weight's columns ([out, in]); "layer<K>.weight" / "layer<K>.bias"
    std::map<std::string, size_t> fan_in;
    for (const auto& [name, tensor] : params) {
        const auto dot = name.rfind('.');
        if (dot != std::string::npos && name.substr(dot + 1) == "weight" && tensor.Shape().size() == 2)
            fan_in[name.substr(0, dot)] = tensor.Shape()[1];
    }
    for (auto& [name, tensor] : params) {
        const auto dot = name.rfind('.');
        const auto it = fan_in.find(dot == std::string::npos ? name : name.substr(0, dot));
        if (it == fan_in.end() || it->second == 0) continue;
        const float bound = 1.0f / std::sqrt(static_cast<float>(it->second));
        af::array u = (af::randu(af::dim4(static_cast<dim_t>(tensor.NumElements())), f32) * 2.0f - 1.0f) * bound;
        u.eval();
        std::vector<float> host(tensor.NumElements());
        u.host(host.data());
        tensor = Tensor(tensor.Shape(), host.data());
    }
    model.SetParameters(params);
}

// ---------------------------------------------------------------- SacLearner

SacLearner::SacLearner(SacNetworks networks, size_t observation_size, size_t action_size, const SacConfig& config)
    : networks_(std::move(networks)), observation_size_(observation_size), action_size_(action_size),
      config_(config) {
    if (!networks_.actor || !networks_.q1 || !networks_.q2 || !networks_.q1_target || !networks_.q2_target)
        throw std::invalid_argument("SAC needs an actor, two critics and their targets.");
    CopyParameters(*networks_.q1, *networks_.q1_target);
    CopyParameters(*networks_.q2, *networks_.q2_target);
    networks_.q1_target->SetTraining(false);
    networks_.q2_target->SetTraining(false);
    actor_optimizer_ = Adam(config_.learning_rate);
    q1_optimizer_ = Adam(config_.learning_rate);  // SB3: one Adam over both critics, the same per parameter
    q2_optimizer_ = Adam(config_.learning_rate);
    // SB3: log_ent_coef starts at log(ent_coef) when learned
    const float start = std::log(config_.ent_coef > 0.0f ? config_.ent_coef : 1.0f);
    log_ent_coef_ = Tensor({1}, &start);
    fixed_ent_coef_ = Tensor({1}, &config_.ent_coef);
    if (config_.ent_coef_auto) ent_coef_optimizer_ = Adam(config_.learning_rate);

    // The five networks fused on the device's provider, or none of them (one
    // path to report)
    std::string why;
    const auto fuse = [&why](IExecutableModel& model) -> std::unique_ptr<FusedMlp> {
        SequentialModel* sequential = model.AsSequentialModel();
        if (!sequential) {
            why = "not a sequential network";
            return nullptr;
        }
        return FusedMlp::For(*sequential, why);
    };
    actor_fused_ = fuse(*networks_.actor);
    if (actor_fused_) q1_fused_ = fuse(*networks_.q1);
    if (q1_fused_) q2_fused_ = fuse(*networks_.q2);
    if (q2_fused_) q1_target_fused_ = fuse(*networks_.q1_target);
    if (q1_target_fused_) q2_target_fused_ = fuse(*networks_.q2_target);
    if (q2_target_fused_) {
        network_path_ = "fused small-network kernels (" + actor_fused_->ProviderId() + ")";
    } else {
        actor_fused_.reset();
        q1_fused_.reset();
        q2_fused_.reset();
        q1_target_fused_.reset();
        spdlog::debug("SAC: the networks run as ArrayFire operations ({})", why);
    }
}

SacLearner::~SacLearner() = default;

void SacLearner::DropFused(const std::exception& e) {
    spdlog::warn("SAC: the fused networks failed ({}); ArrayFire operations from now on", e.what());
    actor_fused_.reset();
    q1_fused_.reset();
    q2_fused_.reset();
    q1_target_fused_.reset();
    q2_target_fused_.reset();
    network_path_ = std::string("ArrayFire operations (the fused kernels failed: ") + e.what() + ")";
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

void SacLearner::StepOptimizers(const std::vector<Update>& updates) {
    if (adam_together_) {
        std::vector<std::map<std::string, Tensor>> params(updates.size());
        std::vector<AdamOptimizer::Group> groups;
        for (size_t i = 0; i < updates.size(); ++i) {
            const Update& u = updates[i];
            params[i] = u.model ? u.model->GetParameters() : std::map<std::string, Tensor>{{"log_ent_coef", log_ent_coef_}};
            groups.push_back({dynamic_cast<AdamOptimizer*>(u.optimizer), &params[i], u.gradients});
        }
        std::string why;
        if (AdamOptimizer::StepTogether(groups, 0.0f, why)) {
            for (size_t i = 0; i < updates.size(); ++i) {
                if (updates[i].model) updates[i].model->SetParameters(params[i]);
                else log_ent_coef_ = params[i].at("log_ent_coef");
            }
            return;
        }
        adam_together_ = false;
        spdlog::debug("SAC: the optimizers step per tensor ({})", why);
    }
    for (const Update& u : updates) {
        if (u.model) {
            ApplyOptimizerStep(*u.model, *u.optimizer, *u.gradients, {});
            continue;
        }
        std::map<std::string, Tensor> params{{"log_ent_coef", log_ent_coef_}};
        u.optimizer->Step(params, *u.gradients);
        log_ent_coef_ = params.at("log_ent_coef");
    }
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
    std::vector<Update> updates;
    if (config_.ent_coef_auto) {
        const SacEntCoefLoss ent = SacEntCoefLossAndGradient(log_ent_coef_, c.log_prob, config_.target_entropy);
        losses.ent_coef_loss = ent.loss;
        ent_grads["log_ent_coef"] = ent.grad_log_ent_coef;
        updates.push_back({ent_coef_optimizer_.get(), nullptr, &ent_grads});
    }
    updates.push_back({q1_optimizer_.get(), n.q1.get(), &c.q1_grads});
    updates.push_back({q2_optimizer_.get(), n.q2.get(), &c.q2_grads});
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
    StepOptimizers({{actor_optimizer_.get(), n.actor.get(), &p.grads}});

    if (update_targets) {
        Polyak(*n.q1, *n.q1_target, config_.tau);
        Polyak(*n.q2, *n.q2_target, config_.tau);
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

// ---------------------------------------------------------------- SacTrainer

SacTrainer::SacTrainer(TrainingEnvironment environment, SacNetworks networks, SacConfig config)
    : environment_(std::move(environment)), networks_(std::move(networks)), config_(std::move(config)) {}

SacTrainer::~SacTrainer() = default;

bool SacTrainer::CheckSetup(std::string& error) const {
    const auto& info = environment_.info;
    const auto& c = config_;
    if (!networks_.actor || !networks_.q1 || !networks_.q2 || !networks_.q1_target || !networks_.q2_target)
        error = "SAC needs an actor, two critics and their targets.";
    else if (environment_.count < 1 || !environment_.reset || !environment_.step) error = "SAC needs an environment.";
    else if (info.discrete_actions) error = "SAC learns continuous actions; " + info.name + " has a choice of actions.";
    else if (info.observation_size < 1 || info.action_size < 1) error = "The environment observes or acts on nothing.";
    else if (info.action_low.size() != static_cast<size_t>(info.action_size) ||
             info.action_high.size() != static_cast<size_t>(info.action_size) ||
             !std::all_of(info.action_low.begin(), info.action_low.end(), [](float v) { return std::isfinite(v); }) ||
             !std::all_of(info.action_high.begin(), info.action_high.end(), [](float v) { return std::isfinite(v); }))
        error = "SAC needs bounded actions; " + info.name + " has actions without finite bounds.";
    else if (c.total_timesteps < 1 || c.buffer_size < 1 || c.learning_starts < 0 || c.batch_size < 1 ||
             c.train_freq < 1 || c.gradient_steps < 1 || c.target_update_interval < 1 || c.report_every < 1)
        error = "SAC's counts must each be at least 1 (the random steps first at least 0).";
    else if (!(c.learning_rate > 0.0f) || !(c.gamma > 0.0f && c.gamma <= 1.0f) || !(c.tau > 0.0f && c.tau <= 1.0f) ||
             !(c.ent_coef > 0.0f))
        error = "SAC's learning rate and entropy weight must be above 0, the discount and tau in (0, 1].";
    return error.empty();
}

bool SacTrainer::TakeTrained(SacTrainedState& out, std::string& error) {
    if (!learner_ || !normalizer_) {
        error = "SAC has not trained a policy.";
        return false;
    }
    std::map<std::string, OptimizerState> optimizers;
    if (!learner_->ExportOptimizers(optimizers, error)) return false;
    out.optimizers = std::move(optimizers);
    out.log_ent_coef = learner_->LogEntCoef();
    out.networks = std::move(learner_->Networks());
    out.normalizer = std::move(normalizer_);
    learner_.reset();
    return true;
}

bool SacTrainer::CopyPolicy(SacPolicyCopy& out, std::string& error) {
    if (!learner_ || !normalizer_) {
        error = "SAC has no policy to copy yet.";
        return false;
    }
    out.parameters.clear();
    for (const auto& [name, tensor] : learner_->Networks().actor->GetParameters()) out.parameters[name] = tensor.Clone();
    out.normalizer = std::make_unique<VecNormalize>(*normalizer_);
    return true;
}

bool SacTrainer::Train(const std::function<void(const SacUpdateMetrics&)>& on_report, std::string& error) {
    error.clear();
    if (!CheckSetup(error)) return false;
    const auto& info = environment_.info;
    const size_t n = static_cast<size_t>(environment_.count);
    const size_t obs_size = static_cast<size_t>(info.observation_size);
    const size_t act_size = static_cast<size_t>(info.action_size);
    const size_t batch = static_cast<size_t>(config_.batch_size);
    Random random(config_.seed ^ 0x5AC5AC5AC5ull);

    std::vector<float> obs(n * obs_size), final_obs(n * obs_size), rewards(n), env_actions(n * act_size);
    std::vector<uint8_t> terminated(n), truncated(n);
    const plugin::PluginEnvironmentStepBuffers out{obs.data(), rewards.data(), terminated.data(), truncated.data(),
                                                   final_obs.data()};
    std::vector<float> raw_obs(n * obs_size), raw_next(n * obs_size);
    std::vector<double> episode_return(n, 0.0);
    std::vector<int> episode_length(n, 0);
    std::deque<std::pair<float, int>> recent;  // the last 100 episodes
    SacUpdateMetrics metrics;

    try {
        learner_ = std::make_unique<SacLearner>(std::move(networks_), obs_size, act_size, config_);
        spdlog::info("SAC: the networks run as {}", learner_->NetworkPath());
        ReplayBuffer buffer(config_.buffer_size, obs_size, act_size);
        metrics.buffer_capacity = buffer.Capacity();
        metrics.buffer_bytes = buffer.Bytes();
        VecNormalizeSettings norm;
        norm.observations = config_.normalize_observations;
        norm.rewards = false;  // SAC: observations only (RL Zoo normalises none)
        norm.gamma = config_.gamma;
        normalizer_ = std::make_unique<VecNormalize>(n, obs_size, norm);
        spdlog::info("SAC: training on {} x {} for {} steps; replay buffer {} steps ({:.0f} MB on the device)", info.id,
                     n, config_.total_timesteps, buffer.Capacity(), static_cast<double>(buffer.Bytes()) / 1e6);

        if (!environment_.reset(config_.seed, obs.data(), error)) return false;
        raw_obs = obs;
        normalizer_->Reset(obs.data());

        Tensor critic_sum, actor_sum, alpha_sum, alpha_loss_sum, q_sum;
        int since_report = 0;
        int64_t vec_steps = 0, last_report = 0;
        auto report_start = std::chrono::steady_clock::now();
        // The numbers since the last report: the losses averaged over its gradient steps
        const auto report = [&]() {
            ++metrics.report;
            metrics.compute_path = learner_->NetworkPath();
            if (since_report > 0) {
                const float k = static_cast<float>(since_report);
                metrics.critic_loss = First(critic_sum) / k;
                metrics.actor_loss = First(actor_sum) / k;
                metrics.ent_coef = First(alpha_sum) / k;
                metrics.mean_q = First(q_sum) / k;
                metrics.ent_coef_loss = config_.ent_coef_auto ? First(alpha_loss_sum) / k : 0.0f;
            } else {
                metrics.ent_coef = config_.ent_coef_auto ? std::exp(First(learner_->LogEntCoef())) : config_.ent_coef;
            }
            since_report = 0;
            if (recent.empty()) {
                metrics.mean_episode_return = metrics.mean_episode_length = std::numeric_limits<float>::quiet_NaN();
            } else {
                double r = 0.0, l = 0.0;
                for (const auto& [ret, len] : recent) {
                    r += ret;
                    l += len;
                }
                metrics.mean_episode_return = static_cast<float>(r / recent.size());
                metrics.mean_episode_length = static_cast<float>(l / recent.size());
            }
            metrics.buffer_size = buffer.Size();
            const auto now = std::chrono::steady_clock::now();
            const double seconds = std::chrono::duration<double>(now - report_start).count();
            metrics.steps_per_second =
                seconds > 0.0 ? static_cast<float>(static_cast<double>(metrics.timesteps - last_report) / seconds) : 0.0f;
            report_start = now;
            last_report = metrics.timesteps;
            metrics.reached_target = recent.size() >= 100 && metrics.mean_episode_return >= config_.target_return;
            if (on_report) on_report(metrics);
        };
        while (metrics.timesteps < config_.total_timesteps && !stop_) {
            while (paused_ && !stop_) std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (stop_) break;
            // ---- Act: uniform random until learning starts, then the actor's draw
            std::vector<float> actions;
            if (metrics.timesteps < config_.learning_starts) {
                actions = Host(SacUniformActions(n, act_size, random.Next()));
            } else {
                const Tensor noise = SacNoise(n, act_size, random.Next());
                actions = Host(learner_->Act(Rows(obs, n, obs_size), &noise));
            }
            for (size_t i = 0; i < n * act_size; ++i) {  // [-1, 1] to the environment's bounds
                const size_t j = i % act_size;
                env_actions[i] = info.action_low[j] + 0.5f * (actions[i] + 1.0f) * (info.action_high[j] - info.action_low[j]);
            }
            if (!environment_.step(env_actions.data(), out, error)) return false;

            // ---- The transitions, raw: an ended copy's next observation is its final one
            for (size_t i = 0; i < n; ++i) {
                const bool done = terminated[i] || truncated[i];
                std::copy_n((done ? final_obs.begin() : obs.begin()) + i * obs_size, obs_size,
                            raw_next.begin() + i * obs_size);
                episode_return[i] += rewards[i];
                ++episode_length[i];
                if (done) {
                    recent.emplace_back(static_cast<float>(episode_return[i]), episode_length[i]);
                    if (recent.size() > 100) recent.pop_front();
                    ++metrics.episodes;
                    episode_return[i] = 0.0;
                    episode_length[i] = 0;
                }
            }
            buffer.Add(raw_obs.data(), actions.data(), rewards.data(), raw_next.data(), terminated.data(), n);
            raw_obs = obs;  // the next step's raw observations
            normalizer_->Step(obs.data(), rewards.data(), terminated.data(), truncated.data(), final_obs.data());
            metrics.timesteps += static_cast<int64_t>(n);
            ++vec_steps;

            // ---- Train every train_freq steps once past learning_starts
            metrics.learning = metrics.timesteps > config_.learning_starts;
            if (metrics.learning && vec_steps % config_.train_freq == 0) {
                for (int g = 0; g < config_.gradient_steps && !stop_; ++g) {
                    ReplayBuffer::Batch b = buffer.Sample(batch, random.Next());
                    if (config_.normalize_observations) {
                        b.observations = NormalizeOnDevice(b.observations, *normalizer_);
                        b.next_observations = NormalizeOnDevice(b.next_observations, *normalizer_);
                    }
                    const Tensor noise_pi = SacNoise(batch, act_size, random.Next());
                    const Tensor noise_next = SacNoise(batch, act_size, random.Next());
                    // SB3 counts the target interval within each train() call
                    const SacStepLosses l = learner_->Step(b, noise_pi, noise_next, g % config_.target_update_interval == 0);
                    const bool first = since_report == 0;
                    critic_sum = first ? l.critic_loss : critic_sum + l.critic_loss;
                    actor_sum = first ? l.actor_loss : actor_sum + l.actor_loss;
                    alpha_sum = first ? l.ent_coef : alpha_sum + l.ent_coef;
                    q_sum = first ? l.mean_q : q_sum + l.mean_q;
                    if (config_.ent_coef_auto) alpha_loss_sum = first ? l.ent_coef_loss : alpha_loss_sum + l.ent_coef_loss;
                    ++since_report;
                    ++metrics.updates;
                }
            }

            // ---- Report
            const bool last = metrics.timesteps >= config_.total_timesteps;
            if (!last && metrics.timesteps - last_report < config_.report_every) continue;
            report();
            if (metrics.reached_target) break;
        }
        // A Stop reports where it stopped, so the run's last numbers say so
        if (stop_ && metrics.timesteps > last_report) report();
    } catch (const std::exception& e) {
        error = std::string("SAC training failed: ") + e.what();
        return false;
    }
    return true;
}

}  // namespace cyxwiz::rl
