#pragma once

// Playback of a trained policy (TOFIX195 S5.3): one copy of its environment,
// stepped on a worker with the policy's actions at the environment's own
// pace (real time), four times faster, or as fast as it goes. The copy is
// shown in the plugin's own view when it has one (IEnvironmentBatch::
// ShowCopy, the MuJoCo Viewport); otherwise the Engine draws what it can
// from the copy's observation. Engine side: it links the plugin host.

#include "trained_policy.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace cyxwiz::rl {

enum class PlaybackSpeed { RealTime = 0, Times4 = 1, Fastest = 2 };

class PolicyPlayback {
public:
    enum class State { Idle, Playing, Paused, Stopped, Failed };

    // What the panel shows, copied out under the lock
    struct Snapshot {
        State state = State::Idle;
        std::string message;            // why it failed
        int episode = 0;                // 1 = the first
        int step = 0;                   // in this episode
        double episode_return = 0.0;    // so far
        std::vector<double> returns;    // finished episodes, oldest first (the last 50)
        std::vector<float> action;      // the last action given
        std::vector<float> observation; // the copy's last observation, raw
        bool shown_in_view = false;     // the plugin's own view shows the copy
    };

    PolicyPlayback() = default;
    ~PolicyPlayback();
    PolicyPlayback(const PolicyPlayback&) = delete;
    PolicyPlayback& operator=(const PolicyPlayback&) = delete;

    // Stops a playback that runs, then plays `policy`. False with the reason
    // when there is no policy.
    bool Start(std::shared_ptr<TrainedPolicy> policy, PlaybackSpeed speed, bool best, std::string& error);
    // The next Start plays in these conditions instead of the policy's own:
    // environment settings and the first episode's seed (a stress test's level,
    // to watch its first episode again). Used once.
    void SetConditions(std::map<std::string, std::string> settings, uint64_t first_seed);
    // Called on the playback's thread before every episode (the first one
    // too), never while an action is computed: it may change the policy's
    // networks (a training run's preview) and give the plugin's view a new
    // label (nullopt: the label stays). For the next Start only, like
    // SetConditions.
    using EpisodeStart = std::function<std::optional<plugin::PluginShownCopyLabel>(TrainedPolicy& policy)>;
    void SetEpisodeStart(EpisodeStart hook);
    void Pause();
    void Resume();
    void NewEpisode();
    // Ends it and gives the plugin's view back (waits for the worker)
    void Stop();

    void SetBest(bool best) { best_ = best; }
    void SetSpeed(PlaybackSpeed speed) { speed_ = static_cast<int>(speed); }
    bool Best() const { return best_; }
    PlaybackSpeed Speed() const { return static_cast<PlaybackSpeed>(speed_.load()); }

    Snapshot Get() const;
    // The policy it plays (null before Start)
    std::shared_ptr<TrainedPolicy> Policy() const;

private:
    void Run();

    std::shared_ptr<TrainedPolicy> policy_;
    bool has_conditions_ = false;  // for this Start (SetConditions)
    EpisodeStart episode_start_;   // for this Start (SetEpisodeStart)
    EpisodeStart pending_episode_start_;
    std::map<std::string, std::string> settings_;
    uint64_t first_seed_ = 0;
    bool pending_conditions_ = false;  // for the next Start
    std::map<std::string, std::string> pending_settings_;
    uint64_t pending_seed_ = 0;
    std::thread worker_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> new_episode_{false};
    std::atomic<bool> best_{true};
    std::atomic<int> speed_{0};
    mutable std::mutex mutex_;
    Snapshot snapshot_;
};

}  // namespace cyxwiz::rl
