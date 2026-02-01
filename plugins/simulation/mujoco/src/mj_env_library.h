#pragma once

#include <string>
#include <vector>

namespace cyxwiz::plugin::mujoco {

struct EnvInfo {
    std::string id;             // "inverted_pendulum"
    std::string name;           // "Inverted Pendulum"
    std::string filename;       // "inverted_pendulum.xml"
    std::string description;
    std::string category;       // "Classic Control", "Locomotion", "Manipulation"
    int obs_dim = 0;
    int act_dim = 0;
    int max_steps = 1000;
    float reward_threshold = 0.0f;
};

class MjEnvLibrary {
public:
    MjEnvLibrary() { PopulateBuiltins(); }

    void SetAssetsDir(const std::string& dir) { assets_dir_ = dir; }

    const std::vector<EnvInfo>& GetAll() const { return envs_; }
    const EnvInfo* FindById(const std::string& id) const;
    std::string GetAssetPath(const EnvInfo& info) const;

    // Unique categories present in the library
    std::vector<std::string> GetCategories() const;

private:
    void PopulateBuiltins();

    std::vector<EnvInfo> envs_;
    std::string assets_dir_;
};

} // namespace cyxwiz::plugin::mujoco
