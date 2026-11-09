#include "rl_training_run.h"

#include "../model_builder.h"
#include "../../plugin/registries/plugin_environment_registry.h"
#include "algorithms/arrayfire_backend_utils.h"

#include <spdlog/spdlog.h>

#include <cstdio>

namespace cyxwiz::rl {

namespace {

// The graph's layers plus the head's output layer
std::unique_ptr<IExecutableModel> Network(const std::vector<CompiledLayer>& hidden, size_t inputs, size_t outputs,
                                          float output_gain, const char* what, std::string& error) {
    std::vector<CompiledLayer> layers = hidden;
    CompiledLayer head;
    head.type = gui::NodeType::Dense;
    head.name = std::string(what) + " output";
    head.units = static_cast<int>(outputs);
    head.parameters["units"] = std::to_string(outputs);
    layers.push_back(head);
    auto built = BuildNetworkFromLayers(layers, inputs);
    if (!built.ok()) {
        error = std::string("The ") + what + " network could not be built: " + built.error_message;
        return nullptr;
    }
    InitializeLikeStableBaselines3(*built.model, output_gain);
    return std::move(built.model);
}

}  // namespace

RLTrainingRun::~RLTrainingRun() {
    Stop();
    if (worker_.joinable()) worker_.join();
}

bool RLTrainingRun::Start(std::shared_ptr<const RLTrainingConfiguration> config, std::string& error) {
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
    }
    config_ = std::move(config);
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
        auto policy = Network(c.policy_layers, c.observation_size, c.policy_outputs, 0.01f, "policy", error);
        if (!policy) return fail(error);
        auto value = Network(c.value_layers, c.observation_size, 1, 1.0f, "value", error);
        if (!value) return fail(error);

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
        const bool ok = trainer.Train(
            [this](const PpoUpdateMetrics& m) {
                std::lock_guard<std::mutex> lock(mutex_);
                updates_.push_back(m);
            },
            error);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            trainer_ = nullptr;
        }
        if (!ok) return fail(error);
        std::lock_guard<std::mutex> lock(mutex_);
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
