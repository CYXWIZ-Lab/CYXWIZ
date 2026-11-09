#pragma once

// One copy of a training batch shown in the MuJoCo Viewport (TOFIX195 S5.2,
// IEnvironmentBatch::ShowCopy): the batch publishes that copy's physics state
// after every Reset and Step; the viewport loads the same model file and
// applies the latest state each frame. The plugin and its batches share it
// through shared_ptr, so a batch that outlives the plugin still holds it.

#include <mujoco/mujoco.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace cyxwiz::plugin::mujoco {

struct MjViewSlot {
    std::mutex mutex;
    const void* owner = nullptr;  // the showing batch; null: nothing shown
    std::string model_path;
    std::string title;            // the task's name ("Inverted Pendulum")
    int copy = 0;
    std::vector<mjtNum> state;    // mj_getState(mjSTATE_FULLPHYSICS)
    uint64_t version = 0;         // grows with every published state
};

}  // namespace cyxwiz::plugin::mujoco
