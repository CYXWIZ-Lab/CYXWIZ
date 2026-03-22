#include "rl_script_generator.h"
#include <sstream>
#include <spdlog/spdlog.h>
#include <filesystem>

namespace cyxwiz {

static std::string GetParam(const std::map<std::string, std::string>& params,
                            const std::string& key, const std::string& default_val) {
    auto it = params.find(key);
    return (it != params.end() && !it->second.empty()) ? it->second : default_val;
}

static bool GetBoolParam(const std::map<std::string, std::string>& params,
                         const std::string& key, bool default_val) {
    auto val = GetParam(params, key, default_val ? "true" : "false");
    return val == "true" || val == "1";
}

std::string RLScriptGenerator::Generate(
    const RLTrainingConfig& config,
    const std::map<std::string, std::string>& reward_params,
    const std::map<std::string, std::string>& obs_filter_params,
    const std::string& save_path)
{
    // Resolve absolute MJCF path
    std::string mjcf_path = config.env_mjcf_path;
    if (!mjcf_path.empty() && !std::filesystem::path(mjcf_path).is_absolute()) {
        mjcf_path = std::filesystem::absolute(mjcf_path).string();
    }
    // Normalize to forward slashes for Python
    for (auto& c : mjcf_path) { if (c == '\\') c = '/'; }

    std::string abs_save = save_path;
    if (!abs_save.empty() && !std::filesystem::path(abs_save).is_absolute()) {
        abs_save = std::filesystem::absolute(abs_save).string();
    }
    for (auto& c : abs_save) { if (c == '\\') c = '/'; }

    // Reward params
    std::string alive_bonus = GetParam(reward_params, "alive_bonus", "1.0");
    std::string ctrl_cost_w = GetParam(reward_params, "ctrl_cost_weight", "0.1");
    bool velocity_reward = GetBoolParam(reward_params, "velocity_reward", true);
    std::string h_thresh = GetParam(reward_params, "height_penalty_threshold", "0.0");
    std::string h_val = GetParam(reward_params, "height_penalty_value", "10.0");

    // Obs filter params
    bool inc_qpos = GetBoolParam(obs_filter_params, "include_qpos", true);
    bool inc_qvel = GetBoolParam(obs_filter_params, "include_qvel", true);
    bool inc_sensors = GetBoolParam(obs_filter_params, "include_sensors", false);

    // Algorithm
    std::string algo = "PPO";  // Default
    // Could be extended to SAC based on config

    // Hidden sizes string
    std::string hidden_str;
    for (size_t i = 0; i < config.hidden_sizes.size(); i++) {
        if (i > 0) hidden_str += ", ";
        hidden_str += std::to_string(config.hidden_sizes[i]);
    }

    std::ostringstream ss;
    ss << R"(# =============================================================================
# CyxWiz RL Training Script (Auto-Generated)
# Environment: )" << mjcf_path << R"(
# Algorithm: )" << algo << R"(
# =============================================================================

import sys
import os

# Check dependencies
_missing = []
for _mod in ['gymnasium', 'mujoco', 'stable_baselines3']:
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_mod)
if _missing:
    print(f"ERROR: Missing Python packages: {', '.join(_missing)}")
    print("Install with: pip install mujoco gymnasium stable-baselines3")
    sys.exit(1)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import pycyxwiz
from stable_baselines3 import )" << algo << R"(
from stable_baselines3.common.callbacks import BaseCallback
import time

# =============================================================================
# Custom Gymnasium Environment
# =============================================================================
class CyxWizMuJoCoEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self):
        super().__init__()
        self.mj_model = mujoco.MjModel.from_xml_path(")" << mjcf_path << R"(")
        self.mj_data = mujoco.MjData(self.mj_model)
        self.max_steps = )" << config.max_episode_steps << R"(
        self.frame_skip = )" << (config.n_steps > 0 ? 1 : 1) << R"(
        self.step_count = 0

        # Action space: continuous, one per actuator
        n_act = self.mj_model.nu
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32
        )

        # Observation space
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

    def _get_obs(self):
        parts = []
)";

    // Generate observation construction
    if (inc_qpos) {
        ss << "        parts.append(self.mj_data.qpos.flatten().copy())\n";
    }
    if (inc_qvel) {
        ss << "        parts.append(self.mj_data.qvel.flatten().copy())\n";
    }
    if (inc_sensors) {
        ss << "        if self.mj_model.nsensor > 0:\n";
        ss << "            parts.append(self.mj_data.sensordata.flatten().copy())\n";
    }

    ss << R"(        if not parts:
            parts.append(np.concatenate([
                self.mj_data.qpos.flatten(),
                self.mj_data.qvel.flatten()
            ]))
        return np.concatenate(parts).astype(np.float32)

    def _compute_reward(self, action):
        reward = )" << alive_bonus << R"(  # alive bonus
        # Control cost
        ctrl_cost = )" << ctrl_cost_w << R"( * np.sum(np.square(action))
        reward -= ctrl_cost
)";

    if (velocity_reward) {
        ss << R"(        # Velocity reward (forward velocity)
        if self.mj_data.qvel.shape[0] > 0:
            reward += float(self.mj_data.qvel[0])
)";
    }

    // Height penalty
    ss << R"(        # Height penalty
        height_threshold = )" << h_thresh << R"(
        if height_threshold > 0.0:
            if self.mj_model.nq >= 3 and self.mj_model.jnt_type[0] == 0:  # free joint
                z = float(self.mj_data.qpos[2])
                if z < height_threshold:
                    reward -= )" << h_val << R"(
        return reward

    def _check_termination(self):
        # Default: check if root body fell below ground
        if self.mj_model.nq >= 3 and self.mj_model.jnt_type[0] == 0:  # free joint
            z = float(self.mj_data.qpos[2])
            return z < 0.2
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        # Apply action
        np.clip(action, self.action_space.low, self.action_space.high, out=action)
        self.mj_data.ctrl[:] = action

        # Step physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.mj_model, self.mj_data)

        self.step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}


# =============================================================================
# CyxWiz Metrics Callback
# =============================================================================
class CyxWizCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._last_ep_count = 0

    def _on_step(self) -> bool:
        # Check stop
        if pycyxwiz.rl_should_stop():
            return False

        # Check pause
        while pycyxwiz.rl_is_paused():
            time.sleep(0.1)
            if pycyxwiz.rl_should_stop():
                return False

        # Stream episode metrics when new episodes complete
        if len(self.model.ep_info_buffer) > self._last_ep_count:
            self._last_ep_count = len(self.model.ep_info_buffer)
            ep = self.model.ep_info_buffer[-1]
            pycyxwiz.rl_update_metric("episode_reward", float(ep["r"]))
            pycyxwiz.rl_update_metric("episode_length", float(ep["l"]))

        return True

    def _on_rollout_end(self):
        # Stream policy diagnostics after each PPO update
        try:
            logs = self.logger.name_to_value
            mapping = {
                "train/policy_gradient_loss": "policy_loss",
                "train/value_loss": "value_loss",
                "train/explained_variance": "explained_variance",
                "train/entropy_loss": "entropy",
            }
            for key, metric_name in mapping.items():
                if key in logs:
                    pycyxwiz.rl_update_metric(metric_name, float(logs[key]))
        except Exception:
            pass  # Logger may not be available yet


# =============================================================================
# Train
# =============================================================================
print("Creating environment...")
env = CyxWizMuJoCoEnv()
print(f"  Action space: {env.action_space}")
print(f"  Observation space: {env.observation_space}")

print("Creating )" << algo << R"( agent...")
model = )" << algo << R"((
    "MlpPolicy",
    env,
    learning_rate=)" << config.learning_rate << R"(,
    gamma=)" << config.gamma << R"(,
    gae_lambda=)" << config.gae_lambda << R"(,
    clip_range=)" << config.clip_range << R"(,
    n_steps=)" << config.n_steps << R"(,
    batch_size=)" << config.batch_size << R"(,
    n_epochs=)" << config.n_epochs << R"(,
    ent_coef=)" << config.entropy_coeff << R"(,
    vf_coef=)" << config.value_coeff << R"(,
    policy_kwargs=dict(net_arch=[)" << hidden_str << R"(]),
    verbose=0,
)

print(f"Training for )" << config.total_timesteps << R"( timesteps...")
pycyxwiz.rl_set_stop(False)
pycyxwiz.rl_set_paused(False)

try:
    model.learn(
        total_timesteps=)" << config.total_timesteps << R"(,
        callback=CyxWizCallback(),
    )
except KeyboardInterrupt:
    print("Training interrupted by user")

# Save model
save_dir = os.path.dirname(")" << abs_save << R"(")
if save_dir:
    os.makedirs(save_dir, exist_ok=True)
model.save(")" << abs_save << R"(")
print(f"Model saved to: )" << abs_save << R"(")
print("RL_TRAINING_COMPLETE")
)";

    spdlog::info("RLScriptGenerator: Generated training script ({} chars)", ss.str().size());
    return ss.str();
}

} // namespace cyxwiz
