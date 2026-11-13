#include "mj_physics.h"

#include <cctype>

namespace cyxwiz::plugin::mujoco {

const char* const kIntegratorNames[4] = {"Euler", "RK4", "Implicit", "Implicit fast"};
const char* const kSolverNames[3] = {"PGS", "CG", "Newton"};

const std::vector<MjPhysicsSwitch>& PhysicsSwitches() {
    static const std::vector<MjPhysicsSwitch> switches = {
        {"Contacts", mjDSBL_CONTACT, false},          {"Joint limits", mjDSBL_LIMIT, false},
        {"Friction loss", mjDSBL_FRICTIONLOSS, false}, {"Actuation", mjDSBL_ACTUATION, false},
        {"Gravity", mjDSBL_GRAVITY, false},           {"Springs", mjDSBL_SPRING, false},
        {"Dampers", mjDSBL_DAMPER, false},            {"Warm start", mjDSBL_WARMSTART, false},
        {"Energy", mjENBL_ENERGY, true},
    };
    return switches;
}

bool SwitchOn(const mjOption& option, const MjPhysicsSwitch& s) {
    return s.enable_bit ? (option.enableflags & s.bit) != 0 : (option.disableflags & s.bit) == 0;
}

void SetSwitch(mjOption& option, const MjPhysicsSwitch& s, bool on) {
    int& flags = s.enable_bit ? option.enableflags : option.disableflags;
    const bool set_bit = s.enable_bit ? on : !on;  // a disable bit is set to turn the switch off
    if (set_bit) flags |= s.bit;
    else flags &= ~s.bit;
}

std::string PhysicsChanges(const mjOption& now, const mjOption& file) {
    std::string text;
    auto add = [&text](const char* what) {
        if (!text.empty()) text += ", ";
        text += what;
    };
    if (now.timestep != file.timestep) add("timestep");
    if (now.integrator != file.integrator) add("integrator");
    if (now.solver != file.solver) add("solver");
    if (now.iterations != file.iterations) add("iterations");
    if (now.gravity[0] != file.gravity[0] || now.gravity[1] != file.gravity[1] || now.gravity[2] != file.gravity[2])
        add("gravity");
    for (const auto& s : PhysicsSwitches()) {
        if (SwitchOn(now, s) == SwitchOn(file, s)) continue;
        std::string word = s.label;
        for (auto& c : word) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        add(word.c_str());
    }
    return text;
}

}  // namespace cyxwiz::plugin::mujoco
