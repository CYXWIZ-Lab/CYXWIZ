// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// Environments for reinforcement learning (plugin API 1.4). The Engine finds
// this through IPlugin::QueryInterface("IEnvironmentProvider") when the plugin
// starts (Training permission). The Engine is the learner: it asks for many
// copies of one environment, sends one action per copy and reads back what
// each copy observed and earned. The plugin only simulates.
//
// Every buffer is flat and row-major, one row per copy, so the Engine moves
// all copies' data in one piece per step:
//   observations [count x observation_size]
//   actions      [count x action_size]   (discrete: one action index per copy)
//   rewards, terminated, truncated [count]

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz::plugin {

// A setting the environment takes, as a graph node parameter would be.
struct PluginEnvironmentSetting {
    std::string key;            // "frame_skip"
    std::string label;          // "Frame skip"
    std::string default_value;  // "5"
    std::string description;    // what it changes
};

struct PluginEnvironmentInfo {
    std::string id;           // unique within the plugin ("InvertedPendulum-v5")
    std::string name;         // shown to the user ("Inverted Pendulum")
    std::string description;
    std::string category;     // "Classic Control", "Locomotion", ...

    int observation_size = 0;

    // Continuous actions: action_size values per copy, each in
    // [action_low[i], action_high[i]]. Discrete: action_size is 1 and the one
    // value is an index in [0, action_count).
    bool discrete_actions = false;
    int action_size = 0;
    int action_count = 0;
    std::vector<float> action_low;
    std::vector<float> action_high;

    int max_episode_steps = 0;  // an episode is truncated after this many steps; 0 = never
    double step_seconds = 0.0;  // simulated time one Step covers (for playback at real speed)
    std::vector<PluginEnvironmentSetting> settings;
};

// Where Step writes. final_observations holds, for each copy that ended in
// this Step (terminated or truncated), the observation it ended on; the row
// of `observations` is then already the first observation of its next
// episode. Rows of copies that did not end are left as they were.
struct PluginEnvironmentStepBuffers {
    float* observations = nullptr;
    float* rewards = nullptr;
    uint8_t* terminated = nullptr;  // 1: the episode reached a terminal state
    uint8_t* truncated = nullptr;   // 1: the episode was cut (step limit)
    float* final_observations = nullptr;
};

// Copies of one environment. It is the plugin's object but stands alone: it
// must not use the plugin object, because it may outlive the plugin's
// OnShutdown (the Engine keeps the library loaded until Release). One
// thread at a time calls it.
class IEnvironmentBatch {
public:
    virtual int Count() const = 0;
    virtual int ObservationSize() const = 0;
    virtual int ActionSize() const = 0;

    // Starts every copy. Copy i's randomness comes from (seed, i) alone, so a
    // seed gives the same runs whatever the number of threads. Writes
    // [count x observation_size].
    virtual bool Reset(uint64_t seed, float* observations) = 0;

    // Applies one action per copy and advances every copy by one step. A
    // copy whose episode ends starts its next episode in the same call.
    // False: the step failed; LastError says why.
    virtual bool Step(const float* actions, const PluginEnvironmentStepBuffers& out) = 0;

    virtual std::string LastError() const = 0;

    // Optional: shows copy `copy` in the plugin's own view (the MuJoCo
    // plugin's viewport) as later Resets and Steps move it, until
    // ShowCopy(-1) or Release. False: these environments have no picture
    // (the Engine then draws what it can, or says there is no view).
    virtual bool ShowCopy(int /*copy*/) { return false; }

    // Frees the batch inside the plugin's own module.
    virtual void Release() = 0;

protected:
    ~IEnvironmentBatch() = default;
};

struct PluginEnvironmentCreateResult {
    IEnvironmentBatch* batch = nullptr;  // null: error says why
    std::string error;
};

class IEnvironmentProvider {
public:
    virtual ~IEnvironmentProvider() = default;

    virtual std::vector<PluginEnvironmentInfo> GetEnvironments() = 0;

    // count copies of environment `id`. Settings it does not name keep their
    // defaults; a value it cannot use is an error, not a silent default.
    virtual PluginEnvironmentCreateResult CreateEnvironments(const std::string& id,
                                                             const std::map<std::string, std::string>& settings,
                                                             int count) = 0;
};

} // namespace cyxwiz::plugin
