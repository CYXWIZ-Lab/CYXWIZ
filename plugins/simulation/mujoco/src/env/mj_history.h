#pragma once

// What the Viewport's timeline and plots read (TOFIX212 P3b, P3c): the states
// of a run, newest last, each with the controls, the reward of its step, the
// root body's speed and the first sensor values. The Viewport's own run keeps
// the last N seconds; a played copy keeps its episode whole. No ImGui. The
// owner guards it (the physics lock).

#include <mujoco/mujoco.h>

#include <deque>
#include <vector>

namespace cyxwiz::plugin::mujoco {

constexpr int kHistoryState = mjSTATE_FULLPHYSICS;  // what a step back puts back (with the controls)
constexpr size_t kHistorySensors = 8;               // sensor values kept a step

struct MjHistoryEntry {
    int episode = 1;
    int step = 0;          // in its episode
    double time = 0.0;     // simulated, in its episode
    double reward = 0.0;   // of the step that led here
    double episode_return = 0.0;  // the episode's reward summed up to here
    double root_speed = 0.0;  // m/s, of the robot's root body
    std::vector<mjtNum> state;     // kHistoryState
    std::vector<double> controls;  // d->ctrl
    std::vector<double> sensors;   // the first kHistorySensors values of d->sensordata
    std::vector<mjtNum> ghost_state;  // a ghost's beside it (mjSTATE_FULLPHYSICS); empty: none
    int ghost_episode = 0;            // ... and its numbers then
    int ghost_step = 0;
    double ghost_return = 0.0;
};

class MjHistory {
public:
    // How much it keeps: entries spanning `seconds` of simulated time (0: no
    // limit) and at most `max_entries`
    void SetLimit(double seconds, size_t max_entries);
    void Clear() {
        entries_.clear();
        span_ = 0.0;
    }
    // A new step; drops the oldest past the limit
    void Add(MjHistoryEntry entry);
    // Keeps the entries up to `index` (a run going on from a step back)
    void TruncateAfter(size_t index);
    // The state of `m` and `d` now as an entry (the caller holds the physics lock)
    static MjHistoryEntry Capture(const mjModel* m, const mjData* d, int episode, int step, double reward,
                                  int root_body);
    // Puts an entry's state and controls back and computes the state again;
    // false when it does not fit the model
    static bool Apply(const mjModel* m, mjData* d, const MjHistoryEntry& entry);

    size_t Size() const { return entries_.size(); }
    bool Empty() const { return entries_.empty(); }
    const MjHistoryEntry& At(size_t i) const { return entries_[i]; }
    const MjHistoryEntry& Newest() const { return entries_.back(); }
    // Entries counted back from the newest that cover `seconds` of one episode's
    // simulated time (the plots' window); at least 1 when not empty
    size_t CountWithin(double seconds) const;

private:
    std::deque<MjHistoryEntry> entries_;
    double seconds_ = 30.0;
    size_t max_entries_ = 100000;
    double span_ = 0.0;  // simulated time the entries cover (summed over their episodes)
};

}  // namespace cyxwiz::plugin::mujoco
