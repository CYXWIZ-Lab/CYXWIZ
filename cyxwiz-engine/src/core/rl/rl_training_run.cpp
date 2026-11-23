#include "rl_training_run.h"

#include "../../plugin/registries/plugin_environment_registry.h"
#include "../rl_playback_presentation.h"
#include "../run_execution_device.h"
#include "../training_trace_collector.h"
#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/device.h>

#include <spdlog/spdlog.h>

#include <cctype>
#include <chrono>

namespace cyxwiz::rl {

RLTrainingRun::~RLTrainingRun() {
    Stop();
    if (worker_.joinable()) worker_.join();
}

bool RLTrainingRun::Start(std::shared_ptr<const RLTrainingConfiguration> config, std::string graph_json,
                          std::string& error) {
    if (Active()) {
        error = "An RL run is already training: stop it first.";
        return false;
    }
    if (!config) {
        error = "The graph is not a compiled RL graph.";
        return false;
    }
    if (worker_.joinable()) worker_.join();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        updates_.clear();
        reports_.clear();
        message_.clear();
        trained_.reset();
        device_ = {};
        preview_.reset();
        preview_next_.reset();
        preview_holds_update_ = 0;
        preview_holds_steps_ = 0;
    }
    config_ = std::move(config);
    graph_json_ = std::move(graph_json);
    stop_requested_ = false;
    paused_ = false;
    state_ = State::Running;
    worker_ = std::thread(&RLTrainingRun::Run, this);
    return true;
}

void RLTrainingRun::Stop() {
    stop_requested_ = true;
    std::lock_guard<std::mutex> lock(mutex_);
    if (trainer_) trainer_->Stop();
    if (off_policy_) off_policy_->Stop();
}

void RLTrainingRun::StopAndWait() {
    Stop();
    if (worker_.joinable()) worker_.join();
}

void RLTrainingRun::Pause() {
    std::lock_guard<std::mutex> lock(mutex_);
    paused_ = true;
    if (trainer_) trainer_->Pause();
    if (off_policy_) off_policy_->Pause();
}

void RLTrainingRun::Resume() {
    std::lock_guard<std::mutex> lock(mutex_);
    paused_ = false;
    if (trainer_) trainer_->Resume();
    if (off_policy_) off_policy_->Resume();
}

std::string RLTrainingRun::Message() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return message_;
}

std::string RLTrainingRun::Waiting() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return waiting_;
}

ExecutionDeviceContext RLTrainingRun::Device() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return device_;
}

std::shared_ptr<TrainedPolicy> RLTrainingRun::Trained() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return trained_;
}

void RLTrainingRun::SetWatch(Watch watch, int every_updates) {
    watch_ = static_cast<int>(watch);
    watch_every_ = std::max(1, every_updates);
}

void RLTrainingRun::TakeNetworksNow() { take_now_ = true; }

std::shared_ptr<TrainedPolicy> RLTrainingRun::PreviewPolicy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return preview_;
}

void RLTrainingRun::PreviewHolds(int& update, long long& steps) const {
    std::lock_guard<std::mutex> lock(mutex_);
    update = preview_holds_update_;
    steps = preview_holds_steps_;
}

bool RLTrainingRun::RefreshPreview(TrainedPolicy& policy, int& update, long long& steps) {
    std::unique_ptr<PreviewNetworks> next;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!preview_next_ || preview_next_->update <= update) return false;
        next = std::move(preview_next_);
        preview_holds_update_ = next->update;
        preview_holds_steps_ = next->steps;
    }
    const std::lock_guard<std::mutex> in_use(policy.in_use);
    const PolicyDeviceScope on_its_device(policy);  // the copies live on the run's device
    policy.policy->SetParameters(next->parameters);
    policy.log_std = next->log_std;
    policy.normalizer = std::move(next->normalizer);
    policy.steps = next->steps;
    update = next->update;
    steps = next->steps;
    return true;
}

const char* RLTrainingRun::Learner() const {
    return config_ ? RLAlgorithmName(config_->algorithm) : "PPO";
}

bool RLTrainingRun::WatchBetweenUpdates(plugin::EnvironmentBatch& batch, int number, int& last_watch,
                                        bool& raw_shown) {
    const int watch = watch_.load();
    if (watch != last_watch) {  // the training copy in the plugin's view, or not
        std::string show_error;
        if (watch == static_cast<int>(Watch::RawCopy)) {
            raw_shown = batch.ShowCopy(0, show_error);
            if (raw_shown) batch.LabelShownCopy(rl_playback::TrainingCopyLabel(0));
        } else if (raw_shown) {
            batch.ShowCopy(-1, show_error);
            raw_shown = false;
        }
        if (!show_error.empty()) spdlog::warn("{}: {}", Learner(), show_error);
    }
    const bool preview = watch == static_cast<int>(Watch::Preview);
    const bool due = take_now_.exchange(false) || (preview && last_watch != watch) ||
                     (preview && number % std::max(1, watch_every_.load()) == 0);
    last_watch = watch;
    return preview && due;
}

void RLTrainingRun::EndRawWatch(plugin::EnvironmentBatch& batch, bool raw_shown) {
    if (!raw_shown) return;
    std::string show_error;
    batch.ShowCopy(-1, show_error);
}

void RLTrainingRun::CopyForPreview(std::unique_ptr<PreviewNetworks> next, const RLTrainingConfiguration& c,
                                   std::unique_ptr<IExecutableModel>& network) {
    std::shared_ptr<TrainedPolicy> made;
    if (!PreviewPolicy()) {
        if (!network) return;  // it did not build (Run logged why)
        // Its own policy network on this run's device (nothing else: it only plays)
        made = std::make_shared<TrainedPolicy>();
        made->config = c;
        made->policy = std::move(network);
        made->log_std = next->log_std;
        made->normalizer = std::make_unique<VecNormalize>(*next->normalizer);
        if (const auto* active = cyxwiz::Device::GetCurrentDevice()) {
            made->device_type = active->GetType();
            made->device_id = active->GetDeviceId();
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (made) preview_ = std::move(made);
    preview_next_ = std::move(next);
}

std::vector<PpoUpdateMetrics> RLTrainingRun::TakeUpdates() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<PpoUpdateMetrics> out;
    out.swap(updates_);
    return out;
}

std::vector<OffPolicyUpdateMetrics> RLTrainingRun::TakeOffPolicyReports() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<OffPolicyUpdateMetrics> out;
    out.swap(reports_);
    return out;
}

namespace {
std::atomic<int> active_runs{0};

// How the run computes, in its trace at the start and on each change
void TraceComputePath(const char* learner, const std::string& path, std::string& traced) {
    if (path == traced) return;
    // Falling back from the fused kernels mid-run is worth a look (a warning)
    const bool fell_back = !traced.empty();
    TrainingTraceCollector::Instance().RecordRuntimeEvent(std::string(learner) + ".ComputePath", path,
                                                          fell_back ? "warning" : "ok");
    traced = path;
}
}  // namespace

bool RLTrainingRun::AnyActive() { return active_runs.load() > 0; }

namespace {
// "ppo", "sac", "td3", "dqn": a run id's start
std::string LowerName(const char* name) {
    std::string out(name);
    for (auto& ch : out) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return out;
}

// The preview's own policy network: PPO's policy, SAC's or TD3's actor, DQN's Q network
std::unique_ptr<IExecutableModel> PreviewNetwork(const RLTrainingConfiguration& c, std::string& error) {
    switch (c.algorithm) {
        case RLAlgorithm::SAC:
            return BuildSacActorNetwork(c, error);
        case RLAlgorithm::TD3:
            return BuildTd3ActorNetwork(c, error);
        case RLAlgorithm::DQN:
            return BuildDqnQNetwork(c, error);
        case RLAlgorithm::PPO:
            break;
    }
    return BuildRLPolicyNetwork(c, error);
}
}  // namespace

bool RLTrainingRun::TrainPpo(const std::shared_ptr<plugin::EnvironmentBatch>& batch, TrainingEnvironment env,
                             std::unique_ptr<IExecutableModel>& preview_network,
                             std::shared_ptr<TrainedPolicy>& trained, std::string& error) {
    const auto& c = *config_;
    std::unique_ptr<IExecutableModel> policy, value;
    if (!BuildRLNetworks(c, policy, value, error)) return false;
    PpoTrainer trainer(std::move(env), std::move(policy), std::move(value), c.ppo);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        trainer_ = &trainer;
        if (paused_) trainer.Pause();
        if (stop_requested_) trainer.Stop();
    }
    spdlog::info("PPO: training on {} x {} for {} steps", c.environment.id, c.copies, c.ppo.total_timesteps);
    std::vector<PpoUpdateMetrics> history;
    std::string traced_path;
    int last_watch = static_cast<int>(Watch::Off);
    bool raw_shown = false;
    const bool ok = trainer.Train(
        [&](const PpoUpdateMetrics& update) {
            PpoUpdateMetrics m = update;
            m.environment_stats = batch->Stats();  // between rollouts: the batch is not stepping
            if (WatchBetweenUpdates(*batch, update.update, last_watch, raw_shown)) {
                auto next = std::make_unique<PreviewNetworks>();
                PpoPolicyCopy copy;
                std::string copy_error;
                if (trainer.CopyPolicy(copy, copy_error)) {
                    next->parameters = std::move(copy.parameters);
                    next->log_std = copy.log_std;
                    next->normalizer = std::move(copy.normalizer);
                    next->update = update.update;
                    next->steps = update.timesteps;
                    CopyForPreview(std::move(next), c, preview_network);
                } else {
                    spdlog::warn("PPO preview: {}", copy_error);
                }
            }
            TraceComputePath("PPO", m.compute_path, traced_path);
            history.push_back(m);
            std::lock_guard<std::mutex> lock(mutex_);
            updates_.push_back(m);
        },
        error);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        trainer_ = nullptr;
    }
    EndRawWatch(*batch, raw_shown);
    if (!ok) return false;
    // The policy stays for playback and Save (a run stopped before its first update has none)
    if (history.empty()) return true;
    PpoTrainedState state;
    if (!trainer.TakeTrained(state, error)) return false;
    trained = MakeTrainedPolicy(c, graph_json_, std::move(state), std::move(history));
    return true;
}

bool RLTrainingRun::TrainOffPolicy(const std::shared_ptr<plugin::EnvironmentBatch>& batch, TrainingEnvironment env,
                                   std::unique_ptr<IExecutableModel>& preview_network,
                                   std::shared_ptr<TrainedPolicy>& trained, std::string& error) {
    const auto& c = *config_;
    const char* learner = Learner();
    std::vector<OffPolicyUpdateMetrics> history;
    // SAC's, TD3's or DQN's trainer through the same loop: the reports, the
    // preview's copies of the actor (DQN: the Q network), Stop / Pause; true
    // when it trained without failing
    const auto train = [&](auto& trainer) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            off_policy_ = &trainer.Loop();
            if (paused_) trainer.Pause();
            if (stop_requested_) trainer.Stop();
        }
        std::string traced_path;
        int last_watch = static_cast<int>(Watch::Off);
        bool raw_shown = false;
        const bool ok = trainer.Train(
            [&](const OffPolicyUpdateMetrics& report) {
                OffPolicyUpdateMetrics m = report;
                m.environment_stats = batch->Stats();  // between steps: the batch is not stepping
                if (WatchBetweenUpdates(*batch, report.report, last_watch, raw_shown)) {
                    auto next = std::make_unique<PreviewNetworks>();
                    ActorCopy copy;
                    std::string copy_error;
                    if (trainer.CopyPolicy(copy, copy_error)) {
                        next->parameters = std::move(copy.parameters);
                        next->normalizer = std::move(copy.normalizer);
                        next->update = report.report;
                        next->steps = report.timesteps;
                        CopyForPreview(std::move(next), c, preview_network);
                    } else {
                        spdlog::warn("{} preview: {}", learner, copy_error);
                    }
                }
                TraceComputePath(learner, m.compute_path, traced_path);
                history.push_back(m);
                std::lock_guard<std::mutex> lock(mutex_);
                reports_.push_back(m);
            },
            error);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            off_policy_ = nullptr;
        }
        EndRawWatch(*batch, raw_shown);
        return ok;
    };
    // A run stopped before its first gradient step has no policy (a Stop reports where it stopped)
    const auto learned = [&history]() { return !history.empty() && history.back().updates > 0; };
    if (c.algorithm == RLAlgorithm::DQN) {
        DqnNetworks networks;
        if (!BuildDqnNetworks(c, networks, error)) return false;
        DqnTrainer trainer(std::move(env), std::move(networks), c.dqn);
        if (!train(trainer)) return false;
        if (!learned()) return true;
        DqnTrainedState state;
        if (!trainer.TakeTrained(state, error)) return false;
        trained = MakeDqnTrainedPolicy(c, graph_json_, std::move(state), std::move(history));
        return true;
    }
    if (c.algorithm == RLAlgorithm::TD3) {
        Td3Networks networks;
        if (!BuildTd3Networks(c, networks, error)) return false;
        Td3Trainer trainer(std::move(env), std::move(networks), c.td3);
        if (!train(trainer)) return false;
        if (!learned()) return true;
        Td3TrainedState state;
        if (!trainer.TakeTrained(state, error)) return false;
        trained = MakeTd3TrainedPolicy(c, graph_json_, std::move(state), std::move(history));
        return true;
    }
    SacNetworks networks;
    if (!BuildSacNetworks(c, networks, error)) return false;
    SacTrainer trainer(std::move(env), std::move(networks), c.sac);
    if (!train(trainer)) return false;
    if (!learned()) return true;
    SacTrainedState state;
    if (!trainer.TakeTrained(state, error)) return false;
    trained = MakeSacTrainedPolicy(c, graph_json_, std::move(state), std::move(history));
    return true;
}

void RLTrainingRun::Run() {
    // Counted for the worker's whole life, every return included
    struct Counted {
        Counted() { ++active_runs; }
        ~Counted() { --active_runs; }
    } counted;
    const auto& c = *config_;
    const bool off_policy = IsOffPolicy(c.algorithm);
    const char* learner = Learner();
    bool traced = false;
    const auto fail = [this, &traced, learner](const std::string& why) {
        spdlog::error("{}: {}", learner, why);
        if (traced) TrainingTraceCollector::Instance().FinishRun("failed");
        std::lock_guard<std::mutex> lock(mutex_);
        message_ = why;
        state_ = State::Failed;
    };
    try {
        // The device: the user's route under the next-run policy, as supervised
        // training takes it; held on this worker for the whole run, so the
        // Engine knows a training uses it (Preferences > Devices waits)
        ExecutionDeviceContext device;
        try {
            device = PrepareRunExecutionDevice(ResolveRunExecutionPolicy(false, learner), learner);
        } catch (const std::exception& e) {
            return fail(std::string("The training device did not start: ") + e.what());
        }
        const ScopedExecutionDeviceContext device_scope(device);
        // One training run per GPU on this machine: wait for it, Stop ends the wait
        auto lease = AcquireGpuRunLease(
            device, [this]() { return stop_requested_.load(); },
            [this](const std::string& message) {
                std::lock_guard<std::mutex> lock(mutex_);
                waiting_ = message;
            });
        {
            std::lock_guard<std::mutex> lock(mutex_);
            waiting_.clear();
        }
        if (lease.cancelled) {
            std::lock_guard<std::mutex> lock(mutex_);
            message_ = "Stopped while waiting for the GPU";
            state_ = State::Stopped;
            return;
        }
        if (!lease.error.empty()) spdlog::warn("{}: {}; the run goes on without it", learner, lease.error);
        const ScopedActiveExecutionDeviceContext active_device_scope;
        const std::string run_id =
            LowerName(learner) + "-" +
            std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count());
        TrainingTraceCollector::Instance().StartRun(run_id);
        traced = true;
        RecordRunExecutionDeviceBind(device, run_id, learner);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            device_ = device;
        }

        std::string error;
        std::shared_ptr<plugin::EnvironmentBatch> batch(
            plugin::PluginEnvironmentRegistry::Instance().Create(c.environment_key, c.environment_settings, c.copies,
                                                                 error));
        if (!batch) return fail(error);

        // The preview's own policy network (P5), made before the run seeds its
        // random engine and made for every run: what training draws from that
        // engine (Dropout masks) is then the same whether anyone watches or not
        std::unique_ptr<IExecutableModel> preview_network = PreviewNetwork(c, error);
        if (!preview_network) spdlog::warn("{} preview: {}", learner, error);
        error.clear();
        // The weights from the run's seed, as the supervised trainer seeds its model
        SeedCurrentArrayFireRandomEngine(RLSeed(c));

        TrainingEnvironment env;
        env.info = batch->Info();
        env.count = batch->Count();
        env.reset = [batch](uint64_t seed, float* obs, std::string& e) { return batch->Reset(seed, obs, e); };
        env.step = [batch](const float* actions, const plugin::PluginEnvironmentStepBuffers& out, std::string& e) {
            return batch->Step(actions, out, e);
        };
        std::shared_ptr<TrainedPolicy> trained;
        const bool ok = off_policy ? TrainOffPolicy(batch, std::move(env), preview_network, trained, error)
                                   : TrainPpo(batch, std::move(env), preview_network, trained, error);
        if (!ok) return fail(error);
        if (trained) {
            if (const auto* active = cyxwiz::Device::GetCurrentDevice()) {  // where its networks live
                trained->device_type = active->GetType();
                trained->device_id = active->GetDeviceId();
            }
        }
        TrainingTraceCollector::Instance().FinishRun(stop_requested_ ? "stopped" : "completed");
        traced = false;
        std::lock_guard<std::mutex> lock(mutex_);
        trained_ = std::move(trained);
        if (stop_requested_) {
            message_ = "Stopped";
            state_ = State::Stopped;
        } else {
            message_ = "Finished";
            state_ = State::Finished;
        }
        spdlog::info("{}: {}", learner, message_);
    } catch (const std::exception& e) {
        fail(std::string(learner) + " training failed: " + e.what());
    }
}

}  // namespace cyxwiz::rl
