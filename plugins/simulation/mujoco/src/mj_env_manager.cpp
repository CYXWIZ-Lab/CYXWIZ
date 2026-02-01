#include "mj_env_manager.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>

namespace cyxwiz::plugin::mujoco {

MjEnvManager::~MjEnvManager() {
    Close();
}

MjEnvManager::MjEnvManager(MjEnvManager&& other) noexcept
    : model_(other.model_),
      data_(other.data_),
      max_episode_steps_(other.max_episode_steps_),
      frame_skip_(other.frame_skip_),
      current_step_(other.current_step_),
      episode_reward_(other.episode_reward_),
      episode_count_(other.episode_count_),
      recent_episode_rewards_(std::move(other.recent_episode_rewards_)),
      recent_episode_lengths_(std::move(other.recent_episode_lengths_)),
      reward_fn_(std::move(other.reward_fn_)),
      termination_fn_(std::move(other.termination_fn_)),
      observation_fn_(std::move(other.observation_fn_)) {
    other.model_ = nullptr;
    other.data_ = nullptr;
    other.max_episode_steps_ = 1000;
    other.frame_skip_ = 1;
    other.current_step_ = 0;
    other.episode_reward_ = 0.0f;
    other.episode_count_ = 0;
}

MjEnvManager& MjEnvManager::operator=(MjEnvManager&& other) noexcept {
    if (this != &other) {
        Close();
        model_ = other.model_;
        data_ = other.data_;
        max_episode_steps_ = other.max_episode_steps_;
        frame_skip_ = other.frame_skip_;
        current_step_ = other.current_step_;
        episode_reward_ = other.episode_reward_;
        episode_count_ = other.episode_count_;
        recent_episode_rewards_ = std::move(other.recent_episode_rewards_);
        recent_episode_lengths_ = std::move(other.recent_episode_lengths_);
        reward_fn_ = std::move(other.reward_fn_);
        termination_fn_ = std::move(other.termination_fn_);
        observation_fn_ = std::move(other.observation_fn_);
        other.model_ = nullptr;
        other.data_ = nullptr;
        other.max_episode_steps_ = 1000;
        other.frame_skip_ = 1;
        other.current_step_ = 0;
        other.episode_reward_ = 0.0f;
        other.episode_count_ = 0;
    }
    return *this;
}

// =============================================================================
// Lifecycle
// =============================================================================

bool MjEnvManager::LoadModel(const std::string& mjcf_path) {
    Close();

    char error[1000] = "";
    model_ = mj_loadXML(mjcf_path.c_str(), nullptr, error, sizeof(error));
    if (!model_) {
        spdlog::error("[MuJoCo] Failed to load model '{}': {}", mjcf_path, error);
        return false;
    }

    data_ = mj_makeData(model_);
    if (!data_) {
        spdlog::error("[MuJoCo] Failed to create mjData for '{}'", mjcf_path);
        mj_deleteModel(model_);
        model_ = nullptr;
        return false;
    }

    spdlog::info("[MuJoCo] Loaded model '{}' — nq={}, nv={}, nu={}, nsensordata={}",
                 mjcf_path, model_->nq, model_->nv, model_->nu, model_->nsensordata);
    return true;
}

bool MjEnvManager::LoadModelFromString(const std::string& xml) {
    // TODO: Use mjVFS (mj_defaultVFS + mj_addBufferVFS) for in-memory loading.
    // mj_loadXML requires a file path — passing nullptr would crash.
    (void)xml;
    spdlog::warn("[MuJoCo] LoadModelFromString not yet implemented — use LoadModel with file path");
    return false;
}

void MjEnvManager::Close() {
    if (data_) {
        mj_deleteData(data_);
        data_ = nullptr;
    }
    if (model_) {
        mj_deleteModel(model_);
        model_ = nullptr;
    }
    current_step_ = 0;
    episode_reward_ = 0.0f;
}

// =============================================================================
// Simulation
// =============================================================================

StepResult MjEnvManager::Step(const std::vector<float>& action) {
    StepResult result;

    if (!model_ || !data_) {
        spdlog::error("[MuJoCo] Step() called on unloaded environment");
        result.terminated = true;
        return result;
    }

    // Apply action to actuators (clamp to model limits)
    int nu = model_->nu;
    for (int i = 0; i < nu && i < static_cast<int>(action.size()); ++i) {
        float a = action[i];
        // Clamp to actuator range if defined
        if (model_->actuator_ctrllimited[i]) {
            float lo = static_cast<float>(model_->actuator_ctrlrange[2 * i]);
            float hi = static_cast<float>(model_->actuator_ctrlrange[2 * i + 1]);
            a = std::clamp(a, lo, hi);
        }
        data_->ctrl[i] = static_cast<mjtNum>(a);
    }

    // Step physics (frame_skip times)
    for (int i = 0; i < frame_skip_; ++i) {
        mj_step(model_, data_);
    }

    current_step_++;

    // Compute reward
    float reward = reward_fn_ ? reward_fn_(model_, data_, action)
                              : ComputeDefaultReward(action);

    // Check termination
    bool terminated = termination_fn_ ? termination_fn_(model_, data_)
                                      : CheckDefaultTermination();
    bool truncated = (current_step_ >= max_episode_steps_);

    // Get observation
    result.observation = observation_fn_ ? observation_fn_(model_, data_)
                                         : ComputeDefaultObservation();
    result.reward = reward;
    result.terminated = terminated;
    result.truncated = truncated;
    result.info["step"] = static_cast<float>(current_step_);
    result.info["sim_time"] = static_cast<float>(data_->time);

    episode_reward_ += reward;

    // Track episode stats on terminal
    if (terminated || truncated) {
        recent_episode_rewards_.push_back(episode_reward_);
        recent_episode_lengths_.push_back(current_step_);
        if (static_cast<int>(recent_episode_rewards_.size()) > kRecentBufferSize) {
            recent_episode_rewards_.pop_front();
            recent_episode_lengths_.pop_front();
        }
        episode_count_++;
        result.info["episode_reward"] = episode_reward_;
        result.info["episode_length"] = static_cast<float>(current_step_);
    }

    return result;
}

StepResult MjEnvManager::Reset() {
    if (!model_ || !data_) {
        spdlog::error("[MuJoCo] Reset() called on unloaded environment");
        return {};
    }

    mj_resetData(model_, data_);
    mj_forward(model_, data_);  // Compute derived quantities

    current_step_ = 0;
    episode_reward_ = 0.0f;

    StepResult result;
    result.observation = observation_fn_ ? observation_fn_(model_, data_)
                                         : ComputeDefaultObservation();
    result.reward = 0.0f;
    result.terminated = false;
    result.truncated = false;
    return result;
}

// =============================================================================
// State Access
// =============================================================================

std::vector<float> MjEnvManager::GetObservation() const {
    if (observation_fn_) return observation_fn_(model_, data_);
    return ComputeDefaultObservation();
}

int MjEnvManager::GetObservationDim() const {
    if (!model_) return 0;
    // Default observation: qpos + qvel + sensor data
    return model_->nq + model_->nv + model_->nsensordata;
}

int MjEnvManager::GetActionDim() const {
    if (!model_) return 0;
    return model_->nu;
}

float MjEnvManager::GetMeanEpisodeReward() const {
    if (recent_episode_rewards_.empty()) return 0.0f;
    float sum = std::accumulate(recent_episode_rewards_.begin(),
                                recent_episode_rewards_.end(), 0.0f);
    return sum / static_cast<float>(recent_episode_rewards_.size());
}

float MjEnvManager::GetMeanEpisodeLength() const {
    if (recent_episode_lengths_.empty()) return 0.0f;
    int sum = std::accumulate(recent_episode_lengths_.begin(),
                              recent_episode_lengths_.end(), 0);
    return static_cast<float>(sum) / static_cast<float>(recent_episode_lengths_.size());
}

// =============================================================================
// Default Functions
// =============================================================================

float MjEnvManager::ComputeDefaultReward(const std::vector<float>& action) const {
    // Generic reward: forward velocity (x-axis) minus control cost
    // This matches the structure used by HalfCheetah, Ant, etc.
    if (!model_ || !data_) return 0.0f;

    // Forward velocity = x velocity of the root body
    float forward_vel = 0.0f;
    if (model_->nv > 0) {
        forward_vel = static_cast<float>(data_->qvel[0]);
    }

    // Control cost = sum of squared actions
    float ctrl_cost = 0.0f;
    for (size_t i = 0; i < action.size(); ++i) {
        ctrl_cost += action[i] * action[i];
    }
    ctrl_cost *= 0.1f;

    return forward_vel - ctrl_cost;
}

bool MjEnvManager::CheckDefaultTermination() const {
    // Generic termination: check if the root body height is too low (fell over)
    if (!model_ || !data_) return false;

    // If the model has a free joint, check z-position of root body.
    // The 0.2m threshold is a generic approximation. Per-environment
    // termination functions should override this via SetTerminationFn().
    if (model_->nq >= 3 && model_->jnt_type[0] == mjJNT_FREE) {
        float z_pos = static_cast<float>(data_->qpos[2]);
        if (z_pos < 0.2f) return true;
    }

    return false;
}

std::vector<float> MjEnvManager::ComputeDefaultObservation() const {
    if (!model_ || !data_) return {};

    std::vector<float> obs;
    obs.reserve(model_->nq + model_->nv + model_->nsensordata);

    // Joint positions
    for (int i = 0; i < model_->nq; ++i) {
        obs.push_back(static_cast<float>(data_->qpos[i]));
    }

    // Joint velocities
    for (int i = 0; i < model_->nv; ++i) {
        obs.push_back(static_cast<float>(data_->qvel[i]));
    }

    // Sensor data (accelerometers, gyroscopes, etc.)
    for (int i = 0; i < model_->nsensordata; ++i) {
        obs.push_back(static_cast<float>(data_->sensordata[i]));
    }

    return obs;
}

} // namespace cyxwiz::plugin::mujoco
