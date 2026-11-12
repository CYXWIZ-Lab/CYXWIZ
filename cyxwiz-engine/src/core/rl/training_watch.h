#pragma once

// Watching a PPO run while it trains (TOFIX212 P5, approved board 7). Live
// preview: the run copies its policy every N updates; a PolicyPlayback plays
// one copy of the environment at real time with the best actions, and takes
// the newest networks at each episode's start (the plugin's view says which
// update it plays). Raw training copy: the run shows its own copy 0. The UI
// thread calls Update each frame while a run exists. Engine side: it links
// the plugin host.

#include "policy_playback.h"
#include "rl_training_run.h"

#include <atomic>
#include <memory>
#include <string>

namespace cyxwiz::rl {

class TrainingWatch {
public:
    struct Status {
        bool previewing = false;     // the preview plays
        int networks_update = 0;     // the update its networks were taken at (0: none yet)
        long long networks_steps = 0;
        int episodes = 0;            // finished by the preview
        bool has_last_return = false;
        double last_return = 0.0;    // the last finished episode's
        int last_return_update = 0;  // the update of the networks that played it
        std::string message;         // why the preview failed
    };

    TrainingWatch() = default;
    ~TrainingWatch() { Stop(); }
    TrainingWatch(const TrainingWatch&) = delete;
    TrainingWatch& operator=(const TrainingWatch&) = delete;

    // Applies the choice to `run`; starts the preview once the run has copied
    // its networks, stops it when the run ends or the choice changes.
    void Update(RLTrainingRun* run, RLTrainingRun::Watch watch, int every_updates);
    // The next update's networks, whatever the interval
    void TakeNow();
    // Stops the preview (waits for its worker)
    void Stop();
    // The environment the preview plays (empty when none plays)
    std::string EnvironmentKey() const;
    Status Get() const;

private:
    PolicyPlayback playback_;
    RLTrainingRun* run_ = nullptr;
    std::shared_ptr<TrainedPolicy> previewed_;  // the run's preview policy being played
    // Written by the preview's thread at an episode's start
    std::atomic<int> update_{0};
    std::atomic<long long> steps_{0};
    std::atomic<int> started_{0};          // episodes started by the preview
    std::atomic<int> finished_update_{0};  // the networks' update of the last finished episode
};

}  // namespace cyxwiz::rl
