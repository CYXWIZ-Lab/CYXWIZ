#include "mj_env_library.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <map>
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
         "Classic Control", 10, 2, 50, -3.75f},

        // ===== Locomotion =====
        {"hopper", "Hopper", "hopper.xml",
         "Make a single-legged robot hop forward as fast as possible.",
         "Locomotion", 11, 3, 1000, 3800.0f},

        {"walker2d", "Walker 2D", "walker2d.xml",
         "Make a planar bipedal robot walk forward without falling.",
         "Locomotion", 17, 6, 1000, 0.0f},

        {"half_cheetah", "Half Cheetah", "half_cheetah.xml",
         "Make a 2D cheetah-like robot run forward as fast as possible.",
         "Locomotion", 17, 6, 1000, 4800.0f},

        // ===== Manipulation =====
        {"pusher", "Pusher", "pusher.xml",
         "Use a 3-joint arm to push an object to a target location.",
         "Manipulation", 23, 7, 100, 0.0f},
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

std::string MjEnvLibrary::GetAssetPath(const EnvInfo& info) const {
    // Custom imported models: use absolute path directly
    if (!info.custom_asset_path.empty()) {
        return info.custom_asset_path;
    }

    // Menagerie models: the project's copy first, then the shared download of the version found
    if (info.source == EnvSource::Menagerie) return ResolveModelRef(RefFor(info), roots_).file.string();

    // Builtin models: the plugin's assets dir
    if (assets_dir_.empty()) return info.filename;
    return (fs::path(assets_dir_) / info.filename).string();
}

namespace {

fs::path CyxWizHome() {
#ifdef _WIN32
    const char* home = std::getenv("USERPROFILE");
#else
    const char* home = std::getenv("HOME");
#endif
    return home ? fs::path(home) / ".cyxwiz" : fs::path(".cyxwiz");
}

}  // namespace

std::string MjEnvLibrary::GetMenagerieRoot() { return (CyxWizHome() / "menagerie").string(); }

std::string MjEnvLibrary::GetImportedRoot() { return (CyxWizHome() / "imported").string(); }

MjModelRef MjEnvLibrary::RefFor(const EnvInfo& info) const {
    MjModelRef ref;
    switch (info.source) {
        case EnvSource::Builtin:
            ref.kind = MjRefKind::Builtin;
            ref.file = info.filename;
            return ref;
        case EnvSource::Menagerie:
            ref.kind = MjRefKind::Menagerie;
            ref.robot = info.repo_path;
            ref.file = info.scene_file;
            ref.commit = info.commit;
            return ref;
        default:
            return RefForFile(info.custom_asset_path, roots_);  // project: in it; imported elsewhere: its path
    }
}

void MjEnvLibrary::SetNeeded(std::vector<MjModelRef> needed) {
    needed_ = std::move(needed);
    for (auto& e : envs_) {
        e.needed_commit.clear();
        if (e.source != EnvSource::Menagerie) continue;
        for (const auto& ref : needed_) {
            if (ref.kind != MjRefKind::Menagerie || ref.robot != e.repo_path || ref.commit.empty()) continue;
            if (ResolveModelRef(ref, roots_).where == MjModelWhere::Missing) {
                e.needed_commit = ref.commit;
                break;
            }
        }
    }
}

void MjEnvLibrary::Rescan() {
    std::error_code ec;
    // The project's own models are listed again (another project may be open)
    envs_.erase(std::remove_if(envs_.begin(), envs_.end(),
                               [](const EnvInfo& e) { return e.source == EnvSource::Project; }),
                envs_.end());
    // Each shared download with its manifest: robot -> its newest version
    struct Found {
        std::string commit, date;
        uint64_t bytes = 0;
    };
    std::map<std::string, Found> shared;
    for (fs::directory_iterator it(roots_.menagerie, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_directory(ec)) continue;
        const auto manifest = ReadModelManifest(it->path());
        if (!manifest || manifest->source != "menagerie" || manifest->robot.empty()) continue;
        auto& f = shared[manifest->robot];
        if (f.commit.empty() || manifest->date > f.date) f = {manifest->commit, manifest->date, manifest->bytes};
    }
    for (auto& e : envs_) {
        if (e.source != EnvSource::Menagerie) continue;
        e.where = MjModelWhere::Missing;
        e.commit.clear();
        e.date.clear();
        e.bytes = 0;
        const fs::path copy = roots_.project.empty() ? fs::path() : ProjectModelsFolder(roots_.project) / e.repo_path;
        const auto copy_manifest = copy.empty() ? std::nullopt : ReadModelManifest(copy);
        if (copy_manifest && copy_manifest->robot == e.repo_path && fs::exists(copy / e.scene_file, ec)) {
            e.where = MjModelWhere::ProjectCopy;
            e.commit = copy_manifest->commit;
            e.date = copy_manifest->date;
            e.bytes = copy_manifest->bytes;
        } else if (const auto it = shared.find(e.repo_path);
                   it != shared.end() &&
                   fs::exists(MenagerieFolder(roots_.menagerie, e.repo_path, it->second.commit) / e.scene_file, ec)) {
            e.where = MjModelWhere::SharedDownload;
            e.commit = it->second.commit;
            e.date = it->second.date;
            e.bytes = it->second.bytes;
        } else if (fs::exists(roots_.menagerie / e.repo_path / e.scene_file, ec)) {
            e.where = MjModelWhere::SharedDownload;  // downloaded before versions were kept
        }
        e.requires_download = e.where == MjModelWhere::Missing;
    }
    // The project's own models (a Menagerie copy shows on its robot's card)
    if (!roots_.project.empty()) {
        const fs::path models = ProjectModelsFolder(roots_.project);
        for (fs::directory_iterator it(models, ec), end; !ec && it != end; it.increment(ec)) {
            if (!it->is_directory(ec)) continue;
            const auto manifest = ReadModelManifest(it->path());
            if (manifest && manifest->source == "menagerie") continue;
            AddImportedModel(it->path().string(), it->path().filename().string(),
                             manifest && !manifest->url.empty() ? "Imported from " + manifest->url
                                                                : "A model of this project",
                             EnvSource::Project);
        }
    }
    // Models imported outside any project, in earlier sessions
    for (fs::directory_iterator it(GetImportedRoot(), ec), end; !ec && it != end; it.increment(ec)) {
        if (it->is_directory(ec))
            AddImportedModel(it->path().string(), it->path().filename().string(), "Saved in " + it->path().string());
    }
    SetNeeded(needed_);
}

bool MjEnvLibrary::AddImportedModel(const std::string& dir, const std::string& name,
                                    const std::string& description, EnvSource source) {
    const std::string id = (source == EnvSource::Project ? "project_" : "imported_") + name;
    for (const auto& e : envs_) {
        if (e.id == id) return true;
    }
    std::error_code ec;
    fs::path model;
    for (fs::recursive_directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path().extension() != ".xml") continue;
        if (it->path().filename() == "scene.xml") {
            model = it->path();
            break;
        }
        if (model.empty()) model = it->path();
    }
    if (model.empty()) return false;

    EnvInfo info;
    info.id = id;
    info.name = name;
    info.filename = model.filename().string();
    info.description = description;
    info.category = source == EnvSource::Project ? "In this project" : "Imported";
    info.source = source;
    info.custom_asset_path = model.string();
    envs_.push_back(std::move(info));
    spdlog::info("MjEnvLibrary: Added imported environment '{}'", name);
    return true;
}

} // namespace cyxwiz::plugin::mujoco
