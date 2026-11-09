#include "rl_training_run.h"

#include "../../plugin/registries/plugin_environment_registry.h"
#include "algorithms/arrayfire_backend_utils.h"

#include <spdlog/spdlog.h>

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

std::shared_ptr<TrainedPolicy> RLTrainingRun::Trained() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return trained_;
}

std::vector<PpoUpdateMetrics> RLTrainingRun::TakeUpdates() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<PpoUpdateMetrics> out;
    out.swap(updates_);
    return out;
}

void RLTrainingRun::Run() {
    const auto& c = *config_;
    const auto fail = [this](const std::string& why) {
        spdlog::error("PPO: {}", why);
        std::lock_guard<std::mutex> lock(mutex_);
        message_ = why;
        state_ = State::Failed;
    };
    try {
        std::string error;
        std::shared_ptr<plugin::EnvironmentBatch> batch(
            plugin::PluginEnvironmentRegistry::Instance().Create(c.environment_key, c.environment_settings, c.copies,
                                                                 error));
        if (!batch) return fail(error);

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
        const bool ok = trainer.Train(
            [this, &history](const PpoUpdateMetrics& m) {
                history.push_back(m);
                std::lock_guard<std::mutex> lock(mutex_);
                updates_.push_back(m);
            },
            error);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            trainer_ = nullptr;
        }
        if (!ok) return fail(error);
        // The policy stays for playback and Save (a run stopped before its
        // first update has none)
        std::shared_ptr<TrainedPolicy> trained;
        PpoTrainedState state;
        if (!history.empty()) {
            if (!trainer.TakeTrained(state, error)) return fail(error);
            trained = MakeTrainedPolicy(c, graph_json_, std::move(state), std::move(history));
        }
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
