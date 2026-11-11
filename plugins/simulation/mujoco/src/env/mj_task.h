#pragma once

// Gymnasium's MuJoCo tasks (v5) in C++: the same model file, observation,
// reward, termination and start-state noise, so a policy trained here and
// its returns compare with Gymnasium's. The randomness is our own (MjRandom),
// the same on every platform; Gymnasium's numpy streams are not reproduced.

#include <mujoco/mujoco.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz::plugin::mujoco {

// SplitMix64: small, fast, and identical on every compiler (the standard
// library's distributions are not).
class MjRandom {
public:
    explicit MjRandom(uint64_t seed = 0) : state_(seed) {}
    uint64_t Next() {
        uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    // [0, 1)
    double Uniform() { return static_cast<double>(Next() >> 11) * 0x1.0p-53; }
    double Uniform(double low, double high) { return low + (high - low) * Uniform(); }
    double Normal();  // standard normal (Box-Muller)

    // Copy i of a batch reset with `seed` draws from this stream only.
    static uint64_t CopySeed(uint64_t seed, int copy);

private:
    uint64_t state_;
    bool has_spare_ = false;
    double spare_ = 0.0;
};

// What one step of a task gives.
struct MjTaskStep {
    double reward = 0.0;
    bool terminated = false;
};

// One task's rules. The batch owns the model and the data; a task reads them.
class MjTask {
public:
    virtual ~MjTask() = default;
    virtual int ObservationSize(const mjModel* m) const = 0;
    virtual void Observe(const mjModel* m, const mjData* d, float* out) const = 0;
    // After mj_resetData: the start state (qpos0 + noise) and mj_forward.
    virtual void ResetState(const mjModel* m, mjData* d, MjRandom& random) const = 0;
    // ctrl is set; advance frame_skip model steps and score the step.
    virtual MjTaskStep Step(const mjModel* m, mjData* d, const float* action) const = 0;
    virtual int FrameSkip() const = 0;
};

// A setting of a task: a number, at least `min`, whole when `integer`.
struct MjTaskSetting {
    std::string key;
    std::string label;
    double default_value = 0.0;
    double min = 0.0;
    bool integer = false;
    std::string description;
};

using MjTaskValues = std::map<std::string, double>;

struct MjTaskSpec {
    std::string id;           // Gymnasium's id ("InvertedPendulum-v5")
    std::string name;
    std::string description;
    std::string category;
    std::string model_file;   // in the plugin's assets folder
    int max_episode_steps = 1000;
    std::vector<MjTaskSetting> settings;  // the task's own (frame skip, noise, weights)
    std::unique_ptr<MjTask> (*create)(const MjTaskValues& values) = nullptr;
};

const std::vector<MjTaskSpec>& MjTasks();
const MjTaskSpec* FindMjTask(const std::string& id);

// The settings every task takes besides its own: episode limit and threads.
const std::vector<MjTaskSetting>& MjBatchSettings();

// Reads `given` against the task's settings and the batch settings: a value
// that is not a number, below its minimum, or not whole when it must be is
// an error; a key neither knows is an error; missing keys take defaults.
bool ReadMjTaskValues(const MjTaskSpec& spec, const std::map<std::string, std::string>& given, MjTaskValues& values,
                      std::string& error);

}  // namespace cyxwiz::plugin::mujoco
