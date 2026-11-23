#include "rl_training_run.h"

#include "../../plugin/registries/plugin_environment_registry.h"
#include "../rl_playback_presentation.h"
#include "../run_execution_device.h"
#include "../training_trace_collector.h"
#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/device.h>

#include <spdlog/spdlog.h>

#include <chrono>

namespace cyxwiz::rl {

RLTrainingRun::~RLTrainingRun() {
    Stop();
    if (worker_.joinable()) worker_.join();
}

bool RLTrainingRun::Start(std::shared_ptr<const RLTrainingConfiguration> config, std::string graph_json,
                          std::string& error) {
    if (Active()) {
        error = "A PPO run is already training: stop it first.";
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
}

void RLTrainingRun::StopAndWait() {
    Stop();
    if (worker_.joinable()) worker_.join();
}

void RLTrainingRun::Pause() {
    std::lock_guard<std::mutex> lock(mutex_);
    paused_ = true;
    if (trainer_) trainer_->Pause();
}

void RLTrainingRun::Resume() {
    std::lock_guard<std::mutex> lock(mutex_);
    paused_ = false;
    if (trainer_) trainer_->Resume();
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
    policy.policy->SetParameters(next->copy.parameters);
    policy.log_std = next->copy.log_std;
    policy.normalizer = std::move(next->copy.normalizer);
    policy.steps = next->steps;
    update = next->update;
    steps = next->steps;
    return true;
}

void RLTrainingRun::CopyForPreview(PpoTrainer& trainer, const RLTrainingConfiguration& c, int update,
                                   long long steps, std::unique_ptr<IExecutableModel>& network) {
    auto next = std::make_unique<PreviewNetworks>();
    std::string error;
    if (!trainer.CopyPolicy(next->copy, error)) {
        spdlog::warn("PPO preview: {}", error);
        return;
    }
    next->update = update;
    next->steps = steps;
    std::shared_ptr<TrainedPolicy> made;
    if (!PreviewPolicy()) {
        if (!network) return;  // it did not build (Run logged why)
        // Its own policy network on this run's device (no value network: it only plays)
        made = std::make_shared<TrainedPolicy>();
        made->config = c;
        made->policy = std::move(network);
        made->log_std = next->copy.log_std;
        made->normalizer = std::make_unique<VecNormalize>(*next->copy.normalizer);
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

namespace {
std::atomic<int> active_runs{0};
}  // namespace

bool RLTrainingRun::AnyActive() { return active_runs.load() > 0; }

void RLTrainingRun::Run() {
    // Counted for the worker's whole life, every return included
    struct Counted {
        Counted() { ++active_runs; }
        ~Counted() { --active_runs; }
    } counted;
    const auto& c = *config_;
    bool traced = false;
    const auto fail = [this, &traced](const std::string& why) {
        spdlog::error("PPO: {}", why);
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
            device = PrepareRunExecutionDevice(ResolveRunExecutionPolicy(false, "PPO"), "PPO");
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
        if (!lease.error.empty()) spdlog::warn("PPO: {}; the run goes on without it", lease.error);
        const ScopedActiveExecutionDeviceContext active_device_scope;
        const std::string run_id =
            "ppo-" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::system_clock::now().time_since_epoch())
                                        .count());
        TrainingTraceCollector::Instance().StartRun(run_id);
        traced = true;
        RecordRunExecutionDeviceBind(device, run_id, "PPO");
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
        std::unique_ptr<IExecutableModel> preview_network = BuildRLPolicyNetwork(c, error);
        if (!preview_network) spdlog::warn("PPO preview: {}", error);
        error.clear();
        // The weights from the run's seed, as the supervised trainer seeds its model
        SeedCurrentArrayFireRandomEngine(c.ppo.seed);
        std::unique_ptr<IExecutableModel> policy, value;
        if (!BuildRLNetworks(c, policy, value, error)) return fail(error);

        PpoEnvironment env;
        env.info = batch->Info();
        env.count = batch->Count();
        env.reset = [batch](uint64_t seed, float* obs, std::string& e) { return batch->Reset(seed, obs, e); };
        env.step = [batch](const float* actions, const plugin::PluginEnvironmentStepBuffers& out, std::string& e) {
            return batch->Step(actions, out, e);
        };
        PpoTrainer trainer(std::move(env), std::move(policy), std::move(value), c.ppo);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            trainer_ = &trainer;
            if (paused_) trainer.Pause();
            if (stop_requested_) trainer.Stop();
        }
        spdlog::info("PPO: training on {} x {} for {} steps", c.environment.id, c.copies, c.ppo.total_timesteps);
        std::vector<PpoUpdateMetrics> history;
        std::string traced_path;  // how the run computes, in its trace at the start and on each change
        int last_watch = static_cast<int>(Watch::Off);
        bool raw_shown = false;
        const bool ok = trainer.Train(
            [this, &history, &traced_path, &batch, &trainer, &c, &last_watch, &raw_shown,
             &preview_network](const PpoUpdateMetrics& update) {
                PpoUpdateMetrics m = update;
                m.environment_stats = batch->Stats();  // between rollouts: the batch is not stepping
                // Watching (P5), between rollouts too
                const int watch = watch_.load();
                if (watch != last_watch) {  // the training copy in the plugin's view, or not
                    std::string show_error;
                    if (watch == static_cast<int>(Watch::RawCopy)) {
                        raw_shown = batch->ShowCopy(0, show_error);
                        if (raw_shown) batch->LabelShownCopy(rl_playback::TrainingCopyLabel(0));
                    } else if (raw_shown) {
                        batch->ShowCopy(-1, show_error);
                        raw_shown = false;
                    }
                    if (!show_error.empty()) spdlog::warn("PPO: {}", show_error);
                }
                const bool preview = watch == static_cast<int>(Watch::Preview);
                const bool due = take_now_.exchange(false) || (preview && last_watch != watch) ||
                                 (preview && update.update % std::max(1, watch_every_.load()) == 0);
                last_watch = watch;
                if (preview && due) CopyForPreview(trainer, c, update.update, update.timesteps, preview_network);
                if (m.compute_path != traced_path) {
                    // Falling back from the fused kernels mid-run is worth a look (a warning)
                    const bool fell_back = !traced_path.empty();
                    TrainingTraceCollector::Instance().RecordRuntimeEvent("PPO.ComputePath", m.compute_path,
                                                                          fell_back ? "warning" : "ok");
                    traced_path = m.compute_path;
                }
                history.push_back(m);
                std::lock_guard<std::mutex> lock(mutex_);
                updates_.push_back(m);
            },
            error);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            trainer_ = nullptr;
        }
        if (raw_shown) {  // the view no longer follows the training
            std::string show_error;
            batch->ShowCopy(-1, show_error);
        }
        if (!ok) return fail(error);
        // The policy stays for playback and Save (a run stopped before its
        // first update has none)
        std::shared_ptr<TrainedPolicy> trained;
        PpoTrainedState state;
        if (!history.empty()) {
            if (!trainer.TakeTrained(state, error)) return fail(error);
            trained = MakeTrainedPolicy(c, graph_json_, std::move(state), std::move(history));
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
        spdlog::info("PPO: {}", message_);
    } catch (const std::exception& e) {
        fail(std::string("PPO training failed: ") + e.what());
    }
}

}  // namespace cyxwiz::rl
