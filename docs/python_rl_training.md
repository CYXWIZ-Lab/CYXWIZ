# Python Gymnasium + Stable-Baselines3 RL Training

This document describes the Python-based reinforcement learning training system integrated into CyxWiz Engine.

## Overview

CyxWiz Engine uses **Python Gymnasium** and **Stable-Baselines3 (SB3)** for real RL training on MuJoCo environments. When you click "Train RL" in the Node Editor, the engine:

1. Generates a complete Python training script from your node configuration
2. Executes it asynchronously via the embedded Python interpreter
3. Streams training metrics back to the Training Dashboard in real-time

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CyxWiz Engine (C++)                        │
├─────────────────────────────────────────────────────────────────┤
│  Node Editor                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ MuJoCo Plant │──│  RL Agent    │──│RewardFunction│          │
│  │  (MJCF path) │  │(hyperparams) │  │  (weights)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              RLScriptGenerator::Generate()               │   │
│  │         Builds complete Python training script           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           ScriptingEngine::ExecuteScriptAsync()          │   │
│  │              Runs Python in background thread            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Python Runtime                               │
├─────────────────────────────────────────────────────────────────┤
│  import gymnasium, mujoco, stable_baselines3                    │
│  import pycyxwiz  # C++ bindings                                │
│                                                                 │
│  class CyxWizMuJoCoEnv(gym.Env):                               │
│      # Custom environment using mujoco Python package           │
│      # Reward function from RewardFunction node                 │
│      # Observation filter from ObservationFilter node           │
│                                                                 │
│  class CyxWizCallback(BaseCallback):                           │
│      def _on_step(self):                                        │
│          if pycyxwiz.rl_should_stop(): return False            │
│          pycyxwiz.rl_update_metric("episode_reward", value)    │
│                                                                 │
│  model = PPO("MlpPolicy", env, learning_rate=..., ...)         │
│  model.learn(total_timesteps=..., callback=CyxWizCallback())   │
│  model.save("trained_model")                                    │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Training Dashboard (C++)                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Episode Reward  │  │  Policy Loss    │  │   Value Loss    │ │
│  │     [plot]      │  │     [plot]      │  │     [plot]      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

Install the required Python packages:

```bash
pip install mujoco gymnasium stable-baselines3
```

**Note**: Python 3.11 or 3.12 recommended. Python 3.14 does not have prebuilt `mujoco` wheels yet.

## Usage

### 1. Create RL Training Graph

In the Node Editor, add the following nodes:

| Node | Purpose | Key Parameters |
|------|---------|----------------|
| **MuJoCo Plant** | Physics environment | `mjcf_path` - path to MJCF XML file |
| **RL Agent** | Training algorithm | `learning_rate`, `gamma`, `total_timesteps`, etc. |
| **RewardFunction** (optional) | Custom rewards | `alive_bonus`, `ctrl_cost_weight`, `velocity_reward` |
| **ObservationFilter** (optional) | Observation space | `include_qpos`, `include_qvel`, `include_sensors` |

Connect MuJoCo Plant → RL Agent.

### 2. Configure Hyperparameters

Select the RL Agent node and configure in the Properties panel:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.0003 | Adam optimizer learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda for advantage estimation |
| `clip_range` | 0.2 | PPO clipping parameter |
| `n_steps` | 2048 | Steps per rollout |
| `batch_size` | 64 | Minibatch size |
| `n_epochs` | 10 | Epochs per update |
| `total_timesteps` | 100000 | Total training steps |
| `hidden_sizes` | [64, 64] | MLP hidden layer sizes |

### 3. Start Training

Click **"Train RL"** in the Node Editor toolbar.

The Training Dashboard will show:
- **Episode Reward** - Average reward per episode
- **Episode Length** - Average steps per episode
- **Policy Loss** - Policy gradient loss
- **Value Loss** - Value function loss
- **Explained Variance** - How well value function predicts returns

### 4. Control Training

| Button | Action |
|--------|--------|
| **Pause** | Temporarily pause training |
| **Resume** | Continue from pause |
| **Stop** | Cancel training, save current model |

## Generated Python Script

The `RLScriptGenerator` creates a complete Python script like this:

```python
# =============================================================================
# CyxWiz RL Training Script (Auto-Generated)
# Environment: D:/models/humanoid.xml
# Algorithm: PPO
# =============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import pycyxwiz
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import time

class CyxWizMuJoCoEnv(gym.Env):
    def __init__(self):
        self.mj_model = mujoco.MjModel.from_xml_path("D:/models/humanoid.xml")
        self.mj_data = mujoco.MjData(self.mj_model)
        self.max_steps = 1000

        n_act = self.mj_model.nu
        self.action_space = spaces.Box(-1.0, 1.0, (n_act,), np.float32)

        obs = self._get_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, obs.shape, np.float32)

    def _get_obs(self):
        return np.concatenate([
            self.mj_data.qpos.flatten(),
            self.mj_data.qvel.flatten()
        ]).astype(np.float32)

    def _compute_reward(self, action):
        reward = 1.0  # alive bonus
        reward -= 0.1 * np.sum(np.square(action))  # control cost
        reward += float(self.mj_data.qvel[0])  # forward velocity
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.mj_data.ctrl[:] = action
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}


class CyxWizCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._last_ep_count = 0

    def _on_step(self) -> bool:
        # Check stop signal from C++
        if pycyxwiz.rl_should_stop():
            return False

        # Check pause signal
        while pycyxwiz.rl_is_paused():
            time.sleep(0.1)
            if pycyxwiz.rl_should_stop():
                return False

        # Stream episode metrics
        if len(self.model.ep_info_buffer) > self._last_ep_count:
            self._last_ep_count = len(self.model.ep_info_buffer)
            ep = self.model.ep_info_buffer[-1]
            pycyxwiz.rl_update_metric("episode_reward", float(ep["r"]))
            pycyxwiz.rl_update_metric("episode_length", float(ep["l"]))

        return True

    def _on_rollout_end(self):
        # Stream policy diagnostics after each PPO update
        logs = self.logger.name_to_value
        for key, metric in [
            ("train/policy_gradient_loss", "policy_loss"),
            ("train/value_loss", "value_loss"),
            ("train/explained_variance", "explained_variance")
        ]:
            if key in logs:
                pycyxwiz.rl_update_metric(metric, float(logs[key]))


# Training
env = CyxWizMuJoCoEnv()
model = PPO(
    "MlpPolicy", env,
    learning_rate=0.0003,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    policy_kwargs=dict(net_arch=[64, 64]),
    verbose=0,
)

model.learn(total_timesteps=100000, callback=CyxWizCallback())
model.save("trained_policy")
print("RL_TRAINING_COMPLETE")
```

## pycyxwiz RL Bindings

The `pycyxwiz` module provides these functions for Python↔C++ communication:

| Function | Description |
|----------|-------------|
| `rl_update_metric(name, value)` | Send metric to Training Dashboard |
| `rl_should_stop()` | Check if user clicked Stop |
| `rl_is_paused()` | Check if user clicked Pause |
| `rl_set_stop(bool)` | Set stop flag (used by C++) |
| `rl_set_paused(bool)` | Set pause flag (used by C++) |
| `rl_set_metric_callback(fn)` | Register Python callback for metrics |

## Metrics Streamed to Dashboard

| Metric Name | Source | Description |
|-------------|--------|-------------|
| `episode_reward` | `ep_info["r"]` | Cumulative reward per episode |
| `episode_length` | `ep_info["l"]` | Steps per episode |
| `policy_loss` | `train/policy_gradient_loss` | PPO policy loss |
| `value_loss` | `train/value_loss` | Value function loss |
| `explained_variance` | `train/explained_variance` | Value prediction quality |

## Key Files

| File | Purpose |
|------|---------|
| `cyxwiz-engine/src/core/rl_script_generator.h` | Script generator header |
| `cyxwiz-engine/src/core/rl_script_generator.cpp` | Generates Python training scripts |
| `cyxwiz-engine/src/gui/node_editor.cpp` | `OnStartRLTraining()`, `OnStopRLTraining()` |
| `cyxwiz-backend/python/bindings.cpp` | pycyxwiz RL bridge functions |
| `cyxwiz-engine/src/gui/panels/training_dashboard_panel.cpp` | Real-time metrics display |

## Troubleshooting

### "Missing Python packages: mujoco"

Install the required packages:
```bash
pip install mujoco gymnasium stable-baselines3
```

### "DLL load failed while importing pycyxwiz"

Ensure the DLL search path includes:
- `build/lib/Release/` (cyxwiz-backend.dll, af.dll)
- `build/bin/Release/` (CUDA, OpenCL dependencies)

### Training doesn't start

1. Check the Python Console for error messages
2. Verify MJCF file path is correct and file exists
3. Ensure MuJoCo Plant node is connected to RL Agent node

### Metrics not updating

1. Verify `pycyxwiz` imports successfully in Python
2. Check that Training Dashboard panel is visible (View → Training Dashboard)

## Future Improvements

- [ ] Add SAC algorithm support
- [ ] Support custom reward functions via Python code node
- [ ] Add curriculum learning options
- [ ] Support multi-environment parallel training (SubprocVecEnv)
- [ ] Add model checkpointing during training
- [ ] Support loading pretrained models for fine-tuning
