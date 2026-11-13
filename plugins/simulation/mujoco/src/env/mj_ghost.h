#pragma once

// The ghost's physics state on the Viewport's model (TOFIX212 P3d): its own
// mjData, set from the ghost batch's published state or from a timeline
// entry, and drawn see-through by MjRenderer. The UI thread uses it alone.

#include <mujoco/mujoco.h>

#include <vector>

namespace cyxwiz::plugin::mujoco {

class MjGhost {
public:
    MjGhost() = default;
    ~MjGhost();
    MjGhost(const MjGhost&) = delete;
    MjGhost& operator=(const MjGhost&) = delete;

    // Puts `state` (mjSTATE_FULLPHYSICS) on model `m` and computes where its
    // bodies are; false when it does not fit the model
    bool Apply(const mjModel* m, const std::vector<mjtNum>& state);
    // Nothing to draw (until the next Apply)
    void Clear() { state_.clear(); }
    // The ghost to draw on model `m`; null when none was applied for it
    mjData* Data(const mjModel* m) const;
    // The state last applied; empty: none
    const std::vector<mjtNum>& State() const { return state_; }

private:
    bool Fits(const mjModel* m) const;

    const mjModel* model_ = nullptr;
    // The model's sizes: a model built later at the same address is another one
    mjtSize nq_ = 0;
    mjtSize nv_ = 0;
    mjtSize nbody_ = 0;
    mjData* data_ = nullptr;
    std::vector<mjtNum> state_;
};

}  // namespace cyxwiz::plugin::mujoco
