#include "mj_env_library.h"

#include <algorithm>
#include <filesystem>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

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

void MjEnvLibrary::AddMenagerieEnv(const std::string& id, const std::string& name,
                                    const std::string& repo_path, const std::string& description,
                                    const std::string& category, int act_dim, int obs_dim,
                                    const std::string& scene_file) {
    EnvInfo info;
    info.id = id;
    info.name = name;
    info.filename = scene_file;
    info.description = description;
    info.category = category;
    info.act_dim = act_dim;
    info.obs_dim = obs_dim;
    info.source = EnvSource::Menagerie;
    info.repo_path = repo_path;
    info.scene_file = scene_file;
    info.requires_download = true;
    envs_.push_back(std::move(info));
}

void MjEnvLibrary::PopulateMenagerie() {
    // ===== Robot Arms =====
    AddMenagerieEnv("franka_emika_panda", "Franka Panda", "franka_emika_panda",
        "7-DOF research arm. Popular for manipulation research.",
        "Arms", 7, 14);

    AddMenagerieEnv("universal_robots_ur5e", "UR5e", "universal_robots_ur5e",
        "6-DOF collaborative robot arm by Universal Robots.",
        "Arms", 6, 12);

    AddMenagerieEnv("universal_robots_ur10e", "UR10e", "universal_robots_ur10e",
        "6-DOF heavy-duty collaborative arm by Universal Robots.",
        "Arms", 6, 12);

    AddMenagerieEnv("kuka_iiwa_14", "KUKA iiwa 14", "kuka_iiwa_14",
        "7-DOF lightweight robot arm by KUKA.",
        "Arms", 7, 14);

    AddMenagerieEnv("kinova_gen3", "Kinova Gen3", "kinova_gen3",
        "7-DOF lightweight arm with integrated vision.",
        "Arms", 7, 14);

    AddMenagerieEnv("sawyer", "Sawyer", "rethink_robotics_sawyer",
        "7-DOF collaborative robot arm by Rethink Robotics.",
        "Arms", 7, 14);

    AddMenagerieEnv("aloha", "ALOHA", "aloha",
        "Dual-arm manipulation system for learning from demonstrations.",
        "Arms", 14, 28);

    // ===== Quadrupeds =====
    AddMenagerieEnv("unitree_go2", "Unitree Go2", "unitree_go2",
        "Compact quadruped robot by Unitree. Popular for locomotion research.",
        "Quadrupeds", 12, 36);

    AddMenagerieEnv("unitree_go1", "Unitree Go1", "unitree_go1",
        "Quadruped robot by Unitree. Predecessor to Go2.",
        "Quadrupeds", 12, 36);

    AddMenagerieEnv("unitree_a1", "Unitree A1", "unitree_a1",
        "High-performance quadruped by Unitree.",
        "Quadrupeds", 12, 36);

    AddMenagerieEnv("anymal_c", "ANYmal C", "anymal_c",
        "Ruggedized quadruped by ANYbotics for industrial inspection.",
        "Quadrupeds", 12, 36);

    AddMenagerieEnv("anymal_b", "ANYmal B", "anymal_b",
        "Quadruped by ANYbotics. Previous generation.",
        "Quadrupeds", 12, 36);

    AddMenagerieEnv("spot", "Boston Dynamics Spot", "boston_dynamics_spot",
        "Agile quadruped by Boston Dynamics.",
        "Quadrupeds", 12, 36);

    // ===== Humanoids =====
    AddMenagerieEnv("unitree_g1", "Unitree G1", "unitree_g1",
        "Full-size humanoid robot by Unitree.",
        "Humanoids", 23, 69);

    AddMenagerieEnv("unitree_h1", "Unitree H1", "unitree_h1",
        "Tall humanoid robot by Unitree with powerful actuators.",
        "Humanoids", 19, 57);

    AddMenagerieEnv("robotis_op3", "Robotis OP3", "robotis_op3",
        "Small humanoid platform for bipedal walking research.",
        "Humanoids", 20, 60);

    // ===== Bipeds =====
    AddMenagerieEnv("agility_cassie", "Agility Cassie", "agility_cassie",
        "Bipedal robot by Agility Robotics. Underactuated walking/running.",
        "Bipeds", 10, 32);

    AddMenagerieEnv("berkeley_humanoid", "Berkeley Humanoid", "berkeley_humanoid",
        "Open-source humanoid platform from UC Berkeley.",
        "Bipeds", 12, 36);

    // ===== Dexterous Hands =====
    AddMenagerieEnv("shadow_hand", "Shadow Hand", "shadow_dexee",
        "24-DOF dexterous robotic hand by Shadow Robot Company.",
        "Hands", 24, 48);

    AddMenagerieEnv("wonik_allegro", "Allegro Hand", "wonik_allegro",
        "16-DOF dexterous robotic hand by Wonik Robotics.",
        "Hands", 16, 32);

    AddMenagerieEnv("leap_hand", "LEAP Hand", "leap_hand",
        "Low-cost dexterous hand for research.",
        "Hands", 16, 32);

    // ===== Grippers =====
    AddMenagerieEnv("robotiq_2f85", "Robotiq 2F-85", "robotiq_2f85",
        "Industrial 2-finger adaptive gripper by Robotiq.",
        "Grippers", 1, 2);

    AddMenagerieEnv("robotis_rh_p12_rn", "Robotis RH-P12-RN", "robotis_rh_p12_rn",
        "Modular gripper by Robotis.",
        "Grippers", 1, 2);

    // ===== Drones / UAVs =====
    AddMenagerieEnv("bitcraze_crazyflie_2", "Crazyflie 2", "bitcraze_crazyflie_2",
        "Nano quadcopter by Bitcraze. Lightweight drone platform.",
        "Drones", 4, 12);

    AddMenagerieEnv("skydio_x2", "Skydio X2", "skydio_x2",
        "Autonomous drone by Skydio.",
        "Drones", 4, 12);

    // ===== Mobile Manipulators =====
    AddMenagerieEnv("google_robot", "Google Robot", "google_robot",
        "Mobile manipulator used in Google's robotics research.",
        "Mobile Manipulators", 8, 20);

    AddMenagerieEnv("google_barkour_vb", "Google Barkour vB", "google_barkour_vb",
        "Quadruped platform used in Google's Barkour agility research.",
        "Mobile Manipulators", 12, 36);

    // ===== Biomechanical =====
    AddMenagerieEnv("myo_hand", "MyoHand", "myohand",
        "Musculotendon-driven hand model for biomechanics research.",
        "Biomechanical", 39, 63);

    AddMenagerieEnv("myo_leg", "MyoLeg", "myoleg",
        "Musculotendon-driven leg model for locomotion biomechanics.",
        "Biomechanical", 80, 120);
}

const EnvInfo* MjEnvLibrary::FindById(const std::string& id) const {
    for (const auto& e : envs_) {
        if (e.id == id) return &e;
    }
    return nullptr;
}

std::string MjEnvLibrary::GetAssetPath(const EnvInfo& info) const {
    // Menagerie models: look in menagerie download dir
    if (info.source == EnvSource::Menagerie) {
        std::string model_dir = GetMenagerieModelDir(info);
        fs::path scene = fs::path(model_dir) / info.scene_file;
        if (fs::exists(scene)) return scene.string();
        // Fallback: maybe it was extracted differently
        return scene.string();
    }

    // Builtin models: look in assets dir
    if (!assets_dir_.empty()) {
        fs::path p = fs::path(assets_dir_) / info.filename;
        if (fs::exists(p)) return p.string();
    }
    // Fallback: try common locations relative to CWD
    for (const auto& dir : {
        "plugins/simulation/mujoco/assets",
        "plugins/mujoco_simulation/assets",
    }) {
        fs::path p = fs::path(dir) / info.filename;
        if (fs::exists(p)) return p.string();
    }
    if (assets_dir_.empty()) return info.filename;
    return (fs::path(assets_dir_) / info.filename).string();
}

bool MjEnvLibrary::IsModelAvailable(const EnvInfo& info) const {
    if (info.source == EnvSource::Builtin) return true;

    std::string model_dir = GetMenagerieModelDir(info);
    fs::path scene = fs::path(model_dir) / info.scene_file;
    return fs::exists(scene);
}

std::string MjEnvLibrary::GetMenagerieModelDir(const EnvInfo& info) const {
    if (menagerie_dir_.empty()) {
        // Default: ~/.cyxwiz/menagerie/
#ifdef _WIN32
        const char* home = std::getenv("USERPROFILE");
#else
        const char* home = std::getenv("HOME");
#endif
        if (home) {
            return (fs::path(home) / ".cyxwiz" / "menagerie" / info.repo_path).string();
        }
        return (fs::path("menagerie") / info.repo_path).string();
    }
    return (fs::path(menagerie_dir_) / info.repo_path).string();
}

void MjEnvLibrary::MarkDownloaded(const std::string& id) {
    for (auto& e : envs_) {
        if (e.id == id) {
            e.requires_download = false;
            break;
        }
    }
}

void MjEnvLibrary::ScanDownloadedModels() {
    for (auto& e : envs_) {
        if (e.source == EnvSource::Menagerie) {
            e.requires_download = !IsModelAvailable(e);
        }
    }
    int available = 0;
    for (const auto& e : envs_) {
        if (e.source == EnvSource::Menagerie && !e.requires_download) available++;
    }
    if (available > 0) {
        spdlog::info("MjEnvLibrary: {} Menagerie models available locally", available);
    }
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
