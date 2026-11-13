#pragma once

// One copy of a training batch shown in the MuJoCo Viewport (TOFIX195 S5.2,
// IEnvironmentBatch::ShowCopy): the batch publishes that copy's physics state
// after every Reset and Step; the viewport loads the same model file and
// applies the latest state each frame. The plugin and its batches share it
// through shared_ptr, so a batch that outlives the plugin still holds it.

#include "mj_stress.h"
#include "plugin/interfaces/i_environment_provider.h"

#include <mujoco/mujoco.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace cyxwiz::plugin::mujoco {

// A second policy's copy drawn see-through beside the shown one
// (IEnvironmentBatch::ShowCopyAsGhost, TOFIX212 P3d): its batch publishes it
// as the shown copy's batch does; the Viewport draws it when it plays the
// same model file and scene.
struct MjGhostView {
    const void* owner = nullptr;  // the ghost's batch; null: no ghost
    std::string model_path;
    MjScene scene;
    std::string name;             // the policy (the Engine's words)
    std::vector<mjtNum> state;    // mj_getState(mjSTATE_FULLPHYSICS)
    int episode = 1;
    int step = 0;
    double episode_return = 0.0;
    bool has_last_return = false;
    double last_return = 0.0;
    uint64_t version = 0;         // grows with every published state
};

struct MjViewSlot {
    std::mutex mutex;
    const void* owner = nullptr;  // the showing batch; null: nothing shown
    std::string model_path;
    std::string title;            // the task's name ("Inverted Pendulum")
    int copy = 0;
    std::vector<mjtNum> state;    // mj_getState(mjSTATE_FULLPHYSICS)
    // The shown copy's episode since it was shown (the Viewport's live
    // numbers): episode 1 starts at ShowCopy or Reset
    int episode = 1;
    int step = 0;                 // steps into the episode
    int max_steps = 0;            // 0: no limit
    double step_seconds = 0.0;    // simulated time of one environment step (model step x frame skip)
    double episode_return = 0.0;  // rewards summed over this episode
    bool has_last_return = false;
    double last_return = 0.0;     // the episode before
    std::vector<float> action;    // the last action given (empty before the first step)
    bool last_ended_by_task = false;  // the episode before ended by the task's own end, not the step limit
    // The Viewport's Stress panel (TOFIX212 P4): requests for the shown copy,
    // applied at the batch's next Step; the scene the copy runs in (when it
    // differs from the Viewport's, the Viewport builds the same model before it
    // applies the state); why a request could not be done (the Viewport takes it)
    std::vector<MjStressRequest> requests;
    MjScene scene;
    std::string stress_message;
    // How the Engine names the shown copy (LabelShownCopy: a training preview,
    // a training copy); empty: "Policy playback"
    PluginShownCopyLabel label;
    uint64_t version = 0;         // grows with every published state
    MjGhostView ghost;            // a second policy beside it (ShowCopyAsGhost)
};

}  // namespace cyxwiz::plugin::mujoco
