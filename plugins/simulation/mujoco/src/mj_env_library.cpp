#include "mj_env_library.h"

#include <algorithm>
#include <filesystem>

namespace cyxwiz::plugin::mujoco {

void MjEnvLibrary::PopulateBuiltins() {
    envs_ = {
        // ===== Classic Control =====
        {"inverted_pendulum", "Inverted Pendulum", "inverted_pendulum.xml",
         "Balance a pole on a sliding cart. Simple continuous control benchmark.",
         "Classic Control", 4, 1, 1000, 950.0f},

        {"cartpole", "Cart-Pole Swing-Up", "cartpole.xml",
         "Swing up and balance a pole attached to a cart via a single actuator.",
         "Classic Control", 4, 1, 500, 475.0f},

        {"reacher", "Reacher 2D", "reacher.xml",
         "Move a 2-joint arm to reach a target position in 2D space.",
         "Classic Control", 11, 2, 200, -3.75f},

        // ===== Locomotion =====
        {"hopper", "Hopper", "hopper.xml",
         "Make a single-legged robot hop forward as fast as possible.",
         "Locomotion", 11, 3, 1000, 3800.0f},

        {"walker2d", "Walker 2D", "walker2d.xml",
         "Make a planar bipedal robot walk forward without falling.",
         "Locomotion", 17, 6, 1000, 4800.0f},

        {"half_cheetah", "Half Cheetah", "half_cheetah.xml",
         "Make a 2D cheetah-like robot run forward as fast as possible.",
         "Locomotion", 17, 6, 1000, 4800.0f},

        // ===== Manipulation =====
        {"pusher", "Pusher", "pusher.xml",
         "Use a 3-joint arm to push an object to a target location.",
         "Manipulation", 23, 7, 200, 0.0f},
    };
}

const EnvInfo* MjEnvLibrary::FindById(const std::string& id) const {
    for (const auto& e : envs_) {
        if (e.id == id) return &e;
    }
    return nullptr;
}

std::string MjEnvLibrary::GetAssetPath(const EnvInfo& info) const {
    if (!assets_dir_.empty()) {
        std::filesystem::path p = std::filesystem::path(assets_dir_) / info.filename;
        if (std::filesystem::exists(p)) return p.string();
    }
    // Fallback: try common locations relative to CWD
    for (const auto& dir : {
        "plugins/simulation/mujoco/assets",
        "plugins/mujoco_simulation/assets",
    }) {
        std::filesystem::path p = std::filesystem::path(dir) / info.filename;
        if (std::filesystem::exists(p)) return p.string();
    }
    // Last resort: return with configured dir or bare filename
    if (assets_dir_.empty()) return info.filename;
    return (std::filesystem::path(assets_dir_) / info.filename).string();
}

std::vector<std::string> MjEnvLibrary::GetCategories() const {
    std::vector<std::string> cats;
    for (const auto& e : envs_) {
        if (std::find(cats.begin(), cats.end(), e.category) == cats.end()) {
            cats.push_back(e.category);
        }
    }
    return cats;
}

} // namespace cyxwiz::plugin::mujoco
