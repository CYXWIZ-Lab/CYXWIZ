#pragma once

// One PPO run of a compiled RL graph (TOFIX195 S4.3): on a worker thread it
// creates the environment copies from the plugin environment registry,
// builds the policy and value networks from the graph's layers plus the
// heads' output layers (Stable-Baselines3's initialisation), and trains them
// with PpoTrainer. The UI thread starts, pauses and stops it and drains its
// per-update metrics. A finished or stopped run keeps its trained policy
// (TOFIX195 S5) until the next Start. Engine side: it links the plugin host.

#include "../rl_graph_compile.h"
#include "ppo_trainer.h"
#include "trained_policy.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace cyxwiz::rl {

class RLTrainingRun {
public:
    enum class State { Idle, Running, Finished, Stopped, Failed };

    RLTrainingRun() = default;
    ~RLTrainingRun();
    RLTrainingRun(const RLTrainingRun&) = delete;
    RLTrainingRun& operator=(const RLTrainingRun&) = delete;

    // False with the reason while a run is active. graph_json: the graph as
    // saved, kept with the trained policy.
    bool Start(std::shared_ptr<const RLTrainingConfiguration> config, std::string graph_json, std::string& error);
    void Stop();
    void Pause();
    void Resume();

    bool Active() const { return state_ == State::Running; }
    bool Paused() const { return paused_; }
    State GetState() const { return state_; }
    std::string Message() const;  // why it failed, or how it ended
    const RLTrainingConfiguration* Config() const { return config_.get(); }

    // The updates since the last call (UI thread)
    std::vector<PpoUpdateMetrics> TakeUpdates();

    // The policy of the last finished or stopped run (null otherwise)
    std::shared_ptr<TrainedPolicy> Trained() const;

private:
    void Run();

    std::shared_ptr<const RLTrainingConfiguration> config_;
    std::string graph_json_;
    std::thread worker_;
    std::atomic<State> state_{State::Idle};
    std::atomic<bool> paused_{false};
    std::atomic<bool> stop_requested_{false};
    mutable std::mutex mutex_;  // guards the members below and trainer_
    std::vector<PpoUpdateMetrics> updates_;
    std::string message_;
    PpoTrainer* trainer_ = nullptr;  // while Run trains
    std::shared_ptr<TrainedPolicy> trained_;
};

}  // namespace cyxwiz::rl
