#pragma once

// The physics options the Viewport's Inspect panel changes (TOFIX212 P3a):
// the switches it shows as chips, and what differs from the model file's
// values. No ImGui: the panel draws them, the tests check them.

#include <mujoco/mujoco.h>

#include <string>
#include <vector>

namespace cyxwiz::plugin::mujoco {

// One switch: a disable bit (on while not disabled) or an enable bit
struct MjPhysicsSwitch {
    const char* label;
    int bit;
    bool enable_bit;  // mjtEnableBit; otherwise mjtDisableBit
};
const std::vector<MjPhysicsSwitch>& PhysicsSwitches();
bool SwitchOn(const mjOption& option, const MjPhysicsSwitch& s);
void SetSwitch(mjOption& option, const MjPhysicsSwitch& s, bool on);

// The integrator and solver choices, in mjtIntegrator / mjtSolver order
extern const char* const kIntegratorNames[4];  // Euler, RK4, Implicit, Implicit fast
extern const char* const kSolverNames[3];      // PGS, CG, Newton

// What differs from the file's values, in words: "timestep, integrator";
// empty when nothing does
std::string PhysicsChanges(const mjOption& now, const mjOption& file);

}  // namespace cyxwiz::plugin::mujoco
