#pragma once

#include <string>
#include <vector>

namespace cyxwiz::plugin::mujoco {

enum class EnvSource {
    Builtin,     // Shipped with plugin (simple XMLs in assets/)
    Menagerie,   // From MuJoCo Menagerie (may need download)
};

struct EnvInfo {
    std::string id;             // "inverted_pendulum" or "franka_emika_panda"
    std::string name;           // "Inverted Pendulum" or "Franka Panda"
    std::string filename;       // "inverted_pendulum.xml" or "scene.xml"
    std::string description;
    std::string category;       // "Classic Control", "Locomotion", "Arms", etc.
    int obs_dim = 0;
    int act_dim = 0;
    int max_steps = 1000;
    float reward_threshold = 0.0f;

    // Menagerie fields
    EnvSource source = EnvSource::Builtin;
    std::string repo_path;      // Menagerie GitHub dir name (e.g. "franka_emika_panda")
    std::string scene_file;     // File to load within model dir (e.g. "scene.xml")
    bool requires_download = false;
};

class MjEnvLibrary {
public:
    MjEnvLibrary() { PopulateBuiltins(); PopulateMenagerie(); }

    void SetAssetsDir(const std::string& dir) { assets_dir_ = dir; }
    void SetMenagerieDir(const std::string& dir) { menagerie_dir_ = dir; }

    const std::vector<EnvInfo>& GetAll() const { return envs_; }
    const EnvInfo* FindById(const std::string& id) const;
    std::string GetAssetPath(const EnvInfo& info) const;

    // Check if a Menagerie model's files exist locally
    bool IsModelAvailable(const EnvInfo& info) const;

    // Mark a model as downloaded (clears requires_download)
    void MarkDownloaded(const std::string& id);

    // Get the local directory for a Menagerie model
    std::string GetMenagerieModelDir(const EnvInfo& info) const;

    // Unique categories present in the library
    std::vector<std::string> GetCategories() const;

    // Scan menagerie dir for previously downloaded models
    void ScanDownloadedModels();

private:
    void PopulateBuiltins();
    void PopulateMenagerie();

    void AddMenagerieEnv(const std::string& id, const std::string& name,
                         const std::string& repo_path, const std::string& description,
                         const std::string& category, int act_dim, int obs_dim,
                         const std::string& scene_file = "scene.xml");

    std::vector<EnvInfo> envs_;
    std::string assets_dir_;
    std::string menagerie_dir_;  // ~/.cyxwiz/menagerie/ or custom path
};

} // namespace cyxwiz::plugin::mujoco
