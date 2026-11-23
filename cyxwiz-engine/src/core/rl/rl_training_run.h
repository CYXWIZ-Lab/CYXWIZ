#pragma once

// One run of a compiled RL graph (TOFIX195 S4.3; SAC since TOFIX216 S4, TD3
// and DQN after it): on a worker thread it creates the environment copies from
// the plugin environment registry, builds the graph's networks (PPO: policy
// and value, SB3's initialisation; SAC, TD3: the actor, twin critics and their
// targets; DQN: the Q network and its target; PyTorch's initialisation) and
// trains them with PpoTrainer, SacTrainer, Td3Trainer or DqnTrainer. The UI thread starts, pauses
// and stops it and drains its reports. A finished or stopped run keeps its
// trained policy (TOFIX195 S5) until the next Start. Engine side: it links the
// plugin host.

#include "../execution_device_context.h"
#include "../rl_graph_compile.h"
#include "ppo_trainer.h"
#include "sac_trainer.h"
#include "trained_policy.h"

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace cyxwiz::plugin {
class EnvironmentBatch;
}

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
    // Stop, and return once the worker has ended (nothing steps the
    // environment any more: its plugin may shut down)
    void StopAndWait();
    void Pause();
    void Resume();

    bool Active() const { return state_ == State::Running; }
    // True while any run in this process trains (its worker is between start
    // and end): what the runtime truth reports as an RL training
    static bool AnyActive();
    bool Paused() const { return paused_; }
    State GetState() const { return state_; }
    std::string Message() const;  // why it failed, or how it ended
    // While the run waits for its GPU (another training run holds it): what to
    // tell the user; empty otherwise
    std::string Waiting() const;
    const RLTrainingConfiguration* Config() const { return config_.get(); }

    // A PPO run's updates since the last call (UI thread)
    std::vector<PpoUpdateMetrics> TakeUpdates();
    // An off-policy run's reports (SAC's, TD3's or DQN's) since the last call (UI thread)
    std::vector<OffPolicyUpdateMetrics> TakeOffPolicyReports();

    // The policy of the last finished or stopped run (null otherwise)
    std::shared_ptr<TrainedPolicy> Trained() const;

    // Watching the run while it trains (TOFIX212 P5), applied by the worker
    // between updates (a SAC run: between reports). Preview: the policy is
    // copied every `every_updates` updates (reports) (and at the first one
    // after Preview is chosen) for a preview to play beside training; training
    // is not changed. RawCopy: the plugin's view follows copy 0 of the
    // training itself. UI thread.
    enum class Watch { Off = 0, Preview = 1, RawCopy = 2 };
    void SetWatch(Watch watch, int every_updates);
    void TakeNetworksNow();  // copy at the next update, whatever the interval
    // The preview's own policy on the run's device, made at the first copy
    // (null before, and after the next Start)
    std::shared_ptr<TrainedPolicy> PreviewPolicy() const;
    // At a preview episode's start (the preview's thread): the newest copy
    // into `policy` when it is newer than `update`; then update and steps say
    // which networks it plays.
    bool RefreshPreview(TrainedPolicy& policy, int& update, long long& steps);
    // The update and steps of the networks the preview policy holds now (0:
    // none put in yet): a preview started again plays them until the next copy
    void PreviewHolds(int& update, long long& steps) const;

    // The device the run trains on: the user's route from Preferences >
    // Devices under the next-run policy, as supervised training resolves it
    // (run_execution_device). Not valid before the run's preflight.
    ExecutionDeviceContext Device() const;

private:
    // A copy of the policy for the preview (PPO's policy or SAC's actor)
    struct PreviewNetworks {
        std::map<std::string, Tensor> parameters;
        Tensor log_std;  // a continuous PPO policy's; empty otherwise
        std::unique_ptr<VecNormalize> normalizer;
        int update = 0;
        long long steps = 0;
    };

    void Run();
    // The learner's own training, between the shared start (device, trace,
    // environment) and end. `preview_network`: the preview's own policy
    // network, built before the run seeded its random engine. False with the
    // reason; `trained` stays null for a run stopped before it trained (PPO:
    // its first update; SAC, TD3, DQN: its first gradient step).
    bool TrainPpo(const std::shared_ptr<plugin::EnvironmentBatch>& batch, TrainingEnvironment env,
                  std::unique_ptr<IExecutableModel>& preview_network, std::shared_ptr<TrainedPolicy>& trained,
                  std::string& error);
    bool TrainOffPolicy(const std::shared_ptr<plugin::EnvironmentBatch>& batch, TrainingEnvironment env,
                        std::unique_ptr<IExecutableModel>& preview_network, std::shared_ptr<TrainedPolicy>& trained,
                        std::string& error);
    // Between two updates (reports), on the worker: the plugin's view follows
    // copy 0 of the training or not; true when the preview is due a copy
    bool WatchBetweenUpdates(plugin::EnvironmentBatch& batch, int number, int& last_watch, bool& raw_shown);
    // The view no longer follows the training (the run's end)
    static void EndRawWatch(plugin::EnvironmentBatch& batch, bool raw_shown);
    // The policy now, for the preview (the worker, between updates). The
    // first copy makes the preview policy from `network`; each copy then
    // waits for the preview's next episode (RefreshPreview).
    void CopyForPreview(std::unique_ptr<PreviewNetworks> next, const RLTrainingConfiguration& c,
                        std::unique_ptr<IExecutableModel>& network);
    // "PPO", "SAC", "TD3" or "DQN": the run's logs and trace
    const char* Learner() const;

    std::shared_ptr<const RLTrainingConfiguration> config_;
    std::string graph_json_;
    std::thread worker_;
    std::atomic<State> state_{State::Idle};
    std::atomic<bool> paused_{false};
    std::atomic<bool> stop_requested_{false};
    mutable std::mutex mutex_;  // guards the members below and the trainers
    std::vector<PpoUpdateMetrics> updates_;
    std::vector<OffPolicyUpdateMetrics> reports_;
    std::string message_;
    std::string waiting_;  // what to tell the user while it waits for its GPU
    PpoTrainer* trainer_ = nullptr;  // while Run trains PPO
    OffPolicyLoop* off_policy_ = nullptr;  // while Run trains SAC, TD3 or DQN
    std::shared_ptr<TrainedPolicy> trained_;
    ExecutionDeviceContext device_;

    // Watching (P5): the UI's choice, read by the worker
    std::atomic<int> watch_{0};
    std::atomic<int> watch_every_{20};
    std::atomic<bool> take_now_{false};
    // Guarded by mutex_: the preview's policy and the newest copy waiting for it
    std::shared_ptr<TrainedPolicy> preview_;
    std::unique_ptr<PreviewNetworks> preview_next_;
    int preview_holds_update_ = 0;  // the networks preview_ holds (RefreshPreview)
    long long preview_holds_steps_ = 0;
};

}  // namespace cyxwiz::rl
