#pragma once

// MjEnvManager — MuJoCo physics engine wrapper
//
// Manages mjModel + mjData lifecycle and provides a Gymnasium-compatible
// step/reset interface for reinforcement learning environments.

#include "env/mj_scene.h"

#include <mujoco/mujoco.h>

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <mutex>

namespace cyxwiz::plugin::mujoco {

struct StepResult {
    std::vector<float> observation;
    float reward = 0.0f;
    bool terminated = false;   // Environment-defined terminal state
    bool truncated = false;    // Max episode steps reached
    std::map<std::string, float> info;
};

// Reward function type — receives model, data, action; returns scalar reward.
// Default reward functions are provided per environment.
using RewardFn = std::function<float(const mjModel*, const mjData*, const std::vector<float>& action)>;

// Termination function type — returns true if episode should end.
using TerminationFn = std::function<bool(const mjModel*, const mjData*)>;

// Observation function type — extracts observation vector from state.
using ObservationFn = std::function<std::vector<float>(const mjModel*, const mjData*)>;

class MjEnvManager {
public:
    MjEnvManager() = default;
    ~MjEnvManager();

    // Non-copyable, movable
    MjEnvManager(const MjEnvManager&) = delete;
    MjEnvManager& operator=(const MjEnvManager&) = delete;
    MjEnvManager(MjEnvManager&& other) noexcept;
    MjEnvManager& operator=(MjEnvManager&& other) noexcept;

    // --- Lifecycle ---
    // The model file with a stress scene added (env/mj_scene.h; none: the file as it is)
    bool LoadModel(const std::string& mjcf_path, const MjScene& scene = {});
    // The same model file with another scene, the state carried over. The
    // caller holds the physics lock. False with the reason: the model is kept.
    bool ChangeScene(const MjScene& scene, std::string& error);
    const MjScene& Scene() const { return scene_; }
    const MjRobotSize& Robot() const { return robot_; }
    const std::string& ModelPath() const { return model_path_; }
    bool LoadModelFromString(const std::string& xml);
    void Close();
    bool IsLoaded() const { return model_ != nullptr; }

    // --- Simulation ---
    StepResult Step(const std::vector<float>& action);
    StepResult Reset();

    // --- State access ---
    std::vector<float> GetObservation() const;
    int GetObservationDim() const;
    int GetActionDim() const;
    int GetCurrentStep() const { return current_step_; }
    int GetMaxEpisodeSteps() const { return max_episode_steps_; }
    void SetMaxEpisodeSteps(int steps) { max_episode_steps_ = steps; }
    int GetFrameSkip() const { return frame_skip_; }
    void SetFrameSkip(int skip) { frame_skip_ = skip; }

    // --- Custom functions (optional overrides) ---
    void SetRewardFn(RewardFn fn) { reward_fn_ = std::move(fn); }
    void SetTerminationFn(TerminationFn fn) { termination_fn_ = std::move(fn); }
    void SetObservationFn(ObservationFn fn) { observation_fn_ = std::move(fn); }

    // --- Episode statistics ---
    float GetEpisodeReward() const { return episode_reward_; }
    int GetEpisodeCount() const { return episode_count_; }
    // The last episode ended by its termination, not its step limit
    bool LastEpisodeTerminated() const { return last_terminated_; }

    // --- Raw MuJoCo access (for renderer) ---
    const mjModel* GetModel() const { return model_; }
    const mjData* GetData() const { return data_; }
    mjData* GetMutableData() { return data_; }

    // Thread safety: lock before accessing mjData from render thread
    std::mutex& GetPhysicsMutex() { return physics_mutex_; }

private:
    float ComputeDefaultReward(const std::vector<float>& action) const;
    bool CheckDefaultTermination() const;
    std::vector<float> ComputeDefaultObservation() const;

    mjModel* model_ = nullptr;
    mjData* data_ = nullptr;
    mutable std::mutex physics_mutex_;

    std::string model_path_;
    MjScene scene_;
    MjRobotSize robot_;
    bool last_terminated_ = false;

    int max_episode_steps_ = 1000;
    int frame_skip_ = 1;
    int current_step_ = 0;

    // Episode tracking
    float episode_reward_ = 0.0f;
    int episode_count_ = 0;

    // Optional custom functions
    RewardFn reward_fn_;
    TerminationFn termination_fn_;
    ObservationFn observation_fn_;
};

} // namespace cyxwiz::plugin::mujoco
