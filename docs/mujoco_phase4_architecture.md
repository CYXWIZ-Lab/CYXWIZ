# MuJoCo Phase 4 Architecture: Graph Execution & RL Training Integration

**Document Version:** 1.0
**Date:** 2026-02-02
**Status:** Design Specification

---

## Executive Summary

This document defines the architectural integration strategy for completing MuJoCo simulation functionality in CyxWiz Engine. Phase 4 bridges the visual node editor with live simulation, enabling two critical workflows:

1. **Manual Control Mode**: Signal sources (sliders, sine waves) → MuJoCo Plant → Scopes/Visualizers
2. **RL Training Mode**: RL Agent ↔ MuJoCo Environment with live viewport updates and metrics

**Design Principles:**
- **Zero breaking changes** to existing code generation (PyTorch/TF/Keras)
- **DLL safety**: All cross-boundary communication via interfaces
- **Incremental implementation**: Each feature stands alone
- **Reuse existing systems**: TrainingDashboardPanel, AsyncTaskManager, NodeType::PluginCustom

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Feature 1: Graph Execution Engine](#3-feature-1-graph-execution-engine)
4. [Feature 2: Functional RL Nodes](#4-feature-2-functional-rl-nodes)
5. [Feature 3: Live RL Training with Viewport](#5-feature-3-live-rl-training-with-viewport)
6. [Feature 4: Training Dashboard Integration](#6-feature-4-training-dashboard-integration)
7. [Feature 5: ONNX Policy Export](#7-feature-5-onnx-policy-export)
8. [Feature 6: Slider → MuJoCo Live Control](#8-feature-6-slider--mujoco-live-control)
9. [Data Flow Diagrams](#9-data-flow-diagrams)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [Risk Analysis & Mitigations](#11-risk-analysis--mitigations)

---

## 1. Current State Analysis

### 1.1 What Works

| Component | Status | Details |
|-----------|--------|---------|
| **Node Editor** | ✅ Complete | Visual graph builder, NodeType enum, links, validation |
| **Code Generation** | ✅ Complete | PyTorch/TF/Keras/PyCyxWiz generation from graph |
| **MuJoCo Plugin** | ✅ Core Complete | DLL with MjEnvManager, MjRenderer, MjViewportPanel |
| **Plugin System** | ✅ Working | INodeProvider, IPanelProvider, NodeTypeCallback |
| **Signal Nodes** | ✅ Defined | Constant, SignalSlider, SineWave, StepSignal, Ramp, Scope |
| **Training System** | ✅ Local Only | TrainingExecutor for Sequential models |
| **Dynamic Pins** | ✅ Working | MuJoCoPlant parses MJCF for actuator/sensor pins |

### 1.2 What's Missing

| Feature | Current State | Gap |
|---------|---------------|-----|
| **Graph Execution** | Signal nodes exist but don't execute | No runtime evaluation loop |
| **RL Nodes Logic** | Visual placeholders only | No reward computation, no policy training |
| **Live Training** | Training runs in background | No viewport updates during RL training |
| **Dashboard RL Metrics** | Dashboard exists for supervised learning | No episode reward, length, success rate |
| **ONNX Export** | No model export | Cannot save trained policy |
| **Live Control** | Sliders render but don't control sim | No connection from UI to MjSimulationExecutor |

### 1.3 Critical Constraints

**DLL Boundary Safety:**
- Engine and Plugin are separate binaries (EXE + DLL)
- NO direct C++ object passing (MSVC/GCC ABI differences)
- Communication via interfaces only (INodeProvider, custom callbacks)
- Plugin cannot directly access NodeEditor state

**NodeType::PluginCustom Design:**
- Plugin nodes appear as NodeType::PluginCustom in MLNode::type
- Actual type resolved via string: MLNode::plugin_qualified_name = "mujoco:MuJoCoPlant"
- Plugin system uses NodeTypeCallback for type resolution

**Existing TrainingExecutor:**
- Designed for Sequential models (Dense, Conv2D, etc.)
- Cannot handle RL training loop (step/reset/rollout)
- Must create parallel RLTrainingExecutor for RL

---

## 2. System Architecture Overview

### 2.1 Two Parallel Execution Paths

```
CODE GENERATION PATH (Existing):
┌──────────────────────────────────────────────────────────────────┐
│  NodeEditor Graph → GraphCompiler → PyTorch/TF/Keras Code        │
│  - Sequential models only                                         │
│  - Static layer stack                                             │
│  - Used for export/deployment                                     │
└──────────────────────────────────────────────────────────────────┘

LIVE EXECUTION PATH (New):
┌──────────────────────────────────────────────────────────────────┐
│  NodeEditor Graph → GraphExecutor → Signal/MuJoCo Node Evaluation│
│  - Dynamic evaluation loop                                        │
│  - Signal flow: Sliders → Plant → Scopes                         │
│  - RL training: Agent ↔ Env with live updates                    │
└──────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Both paths coexist. Code generation remains unchanged. Graph execution is an **orthogonal feature**.

### 2.2 Component Interaction Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CyxWiz Engine (EXE)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────┐     ┌───────────────┐     ┌──────────────────┐     │
│  │ NodeEditor │────>│ GraphExecutor │<───>│ AsyncTaskManager │     │
│  │  (Visual)  │     │  (Evaluator)  │     │   (Background)   │     │
│  └────────────┘     └───────────────┘     └──────────────────┘     │
│         │                    │                       │              │
│         │                    v                       v              │
│         │         ┌─────────────────────┐  ┌──────────────────┐    │
│         └────────>│ RLTrainingExecutor  │  │TrainingDashboard │    │
│                   │ (Episode Loop)      │  │     Panel        │    │
│                   └─────────────────────┘  └──────────────────┘    │
│                            │                         ^              │
└────────────────────────────┼─────────────────────────┼──────────────┘
                             │  DLL Boundary           │
┌────────────────────────────┼─────────────────────────┼──────────────┐
│                            v                         │              │
│                    INodeProvider                     │              │
│                    ITrainingHook                     │              │
│                            │                         │              │
│  ┌─────────────────────────┼─────────────────────────┼──────────┐  │
│  │              MuJoCo Plugin (DLL)                   │          │  │
│  ├────────────────────────────────────────────────────┼──────────┤  │
│  │  ┌──────────────────┐  ┌─────────────────────┐    │          │  │
│  │  │ MjSimulation     │  │ MjEnvManager        │    │          │  │
│  │  │   Executor       │  │  (Physics)          │    │          │  │
│  │  └──────────────────┘  └─────────────────────┘    │          │  │
│  │         │                       │                  │          │  │
│  │         │                       v                  │          │  │
│  │         │               ┌───────────────┐          │          │  │
│  │         │               │ MjRenderer    │──────────┘          │  │
│  │         │               │ (OpenGL)      │                     │  │
│  │         │               └───────────────┘                     │  │
│  │         │                       │                             │  │
│  │         v                       v                             │  │
│  │  ┌──────────────────────────────────────┐                    │  │
│  │  │     MjViewportPanel (ImGui)          │                    │  │
│  │  │  - Play/Pause/Step controls          │                    │  │
│  │  │  - Renders via MjRenderer texture    │                    │  │
│  │  └──────────────────────────────────────┘                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                      MuJoCo Plugin                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Feature 1: Graph Execution Engine

### 3.1 Purpose

Evaluate node graph at runtime to enable:
- Signal sources outputting values
- MuJoCo Plant consuming inputs and producing outputs
- Scopes visualizing signals in real-time

### 3.2 Design: GraphExecutor Class

**Location:** `cyxwiz-engine/src/core/graph_executor.h/cpp`

**Responsibilities:**
1. Build execution order from NodeEditor graph (topological sort)
2. Allocate value storage for each pin
3. Execute nodes in dependency order
4. Handle plugin node evaluation via callbacks

```cpp
// cyxwiz-engine/src/core/graph_executor.h
#pragma once

#include "../gui/node_editor.h"
#include <vector>
#include <map>
#include <variant>
#include <functional>

namespace cyxwiz {

// Runtime value flowing through graph
using NodeValue = std::variant<
    float,                      // Scalar
    std::vector<float>,         // Vector (sensor outputs, etc.)
    Tensor,                     // Multi-dimensional array
    std::string                 // String (for env handles, etc.)
>;

// Result of evaluating a plugin node
struct PluginNodeEvalResult {
    std::map<int, NodeValue> output_pin_values;  // pin_id -> value
    bool success = true;
    std::string error_message;
};

// Callback for plugin nodes to evaluate themselves
// Provided by PluginManager, dispatches to correct plugin DLL
using PluginEvalCallback = std::function<PluginNodeEvalResult(
    const std::string& plugin_qualified_name,  // "mujoco:MuJoCoPlant"
    const std::map<std::string, std::string>& parameters,
    const std::map<int, NodeValue>& input_values  // pin_id -> value
)>;

/**
 * GraphExecutor - Evaluates node graph at runtime for live simulation
 *
 * Usage:
 *   GraphExecutor executor(nodes, links);
 *   executor.SetPluginEvalCallback(...);
 *   executor.Prepare();  // One-time setup
 *   while (simulating) {
 *       executor.EvaluateFrame();  // Each simulation step
 *       auto value = executor.GetPinValue(scope_input_pin_id);
 *   }
 */
class GraphExecutor {
public:
    GraphExecutor(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links
    );

    // Setup
    void SetPluginEvalCallback(PluginEvalCallback callback);
    bool Prepare();  // Build execution order, validate graph

    // Execution
    bool EvaluateFrame();  // Evaluate entire graph once
    void Reset();          // Clear all pin values

    // Value access
    NodeValue GetPinValue(int pin_id) const;
    void SetPinValue(int pin_id, NodeValue value);  // For external inputs

    // Errors
    bool HasError() const { return !error_message_.empty(); }
    std::string GetError() const { return error_message_; }

private:
    const std::vector<gui::MLNode>& nodes_;
    const std::vector<gui::NodeLink>& links_;

    std::vector<int> execution_order_;  // Node IDs in topological order
    std::map<int, NodeValue> pin_values_;  // pin_id -> current value
    PluginEvalCallback plugin_eval_callback_;
    std::string error_message_;

    // Helpers
    bool BuildExecutionOrder();
    bool EvaluateNode(int node_id);
    bool EvaluateBuiltInNode(const gui::MLNode& node);
    bool EvaluatePluginNode(const gui::MLNode& node);
    std::map<int, NodeValue> GatherInputValues(int node_id) const;
};

} // namespace cyxwiz
```

### 3.3 Built-In Node Evaluation

**Signal Nodes:**
- **Constant**: Output fixed value from parameters
- **SignalSlider**: Read current slider value from ImGui state
- **SineWave**: Compute `amplitude * sin(2πf*t + phase)` using current sim time
- **StepSignal**: Output 0 before threshold, value after
- **RampSignal**: Linear interpolation
- **Scope**: Store input value for plotting (no outputs)

**Example Implementation:**
```cpp
bool GraphExecutor::EvaluateBuiltInNode(const gui::MLNode& node) {
    switch (node.type) {
        case gui::NodeType::Constant: {
            float value = std::stof(node.parameters.at("value"));
            SetPinValue(node.outputs[0].id, value);
            return true;
        }
        case gui::NodeType::SignalSlider: {
            // Read from ImGui slider state (stored in node parameters)
            float value = std::stof(node.parameters.at("current_value"));
            SetPinValue(node.outputs[0].id, value);
            return true;
        }
        case gui::NodeType::SineWave: {
            float amplitude = std::stof(node.parameters.at("amplitude"));
            float frequency = std::stof(node.parameters.at("frequency"));
            float phase = std::stof(node.parameters.at("phase"));
            float t = GetSimulationTime();  // From plugin or global state
            float value = amplitude * std::sin(2.0f * M_PI * frequency * t + phase);
            SetPinValue(node.outputs[0].id, value);
            return true;
        }
        case gui::NodeType::SignalScope: {
            // Read input value and store for plotting
            auto input_value = GetPinValue(node.inputs[0].id);
            // Store in Scope's internal buffer for ImPlot rendering
            // (Implementation: store in node's parameter map or separate registry)
            return true;
        }
        default:
            return false;  // Not a signal node
    }
}
```

### 3.4 Plugin Node Evaluation

**Key Challenge:** Engine cannot directly call plugin functions (DLL boundary).

**Solution:** Callback indirection via PluginManager.

**Flow:**
1. GraphExecutor encounters NodeType::PluginCustom
2. Calls `plugin_eval_callback_` with plugin_qualified_name and input values
3. PluginManager routes to correct plugin DLL's INodeProvider
4. Plugin implements `INodeProvider::EvaluateNode()` (NEW interface method)
5. Plugin returns output values
6. GraphExecutor stores outputs and continues

**New Interface Method:**
```cpp
// cyxwiz-engine/include/plugin/interfaces/i_node_provider.h

struct NodeEvalContext {
    std::string node_type_name;  // e.g., "MuJoCoPlant"
    std::map<std::string, std::string> parameters;
    std::map<int, NodeValue> input_values;  // pin_id -> value
};

struct NodeEvalResult {
    std::map<int, NodeValue> output_values;
    bool success = true;
    std::string error_message;
};

class INodeProvider {
public:
    // ... existing methods ...

    /**
     * Evaluate a node at runtime (for live simulation)
     * @param ctx Node parameters and inputs
     * @return Output values for each output pin
     */
    virtual NodeEvalResult EvaluateNode(const NodeEvalContext& ctx) = 0;
};
```

**MuJoCo Plugin Implementation:**
```cpp
// plugins/simulation/mujoco/src/mujoco_plugin.cpp

NodeEvalResult MuJoCoPlugin::EvaluateNode(const NodeEvalContext& ctx) {
    NodeEvalResult result;

    if (ctx.node_type_name == "MuJoCoPlant") {
        // Extract actuator inputs from input pin values
        std::vector<float> ctrl_vector;
        for (const auto& [pin_id, value] : ctx.input_values) {
            if (std::holds_alternative<float>(value)) {
                ctrl_vector.push_back(std::get<float>(value));
            }
        }

        // Apply controls to MjSimulationExecutor
        std::vector<ActuatorInput> actuators;
        for (size_t i = 0; i < ctrl_vector.size(); ++i) {
            actuators.push_back({GetActuatorName(i), ctrl_vector[i]});
        }
        sim_executor_.SetActuatorInputs(actuators);

        // Get sensor outputs after step
        auto sensors = sim_executor_.GetSensorOutputs();

        // Map sensors to output pins
        for (size_t i = 0; i < sensors.size(); ++i) {
            int pin_id = GetSensorOutputPinId(i);  // Resolve from dynamic pins
            result.output_values[pin_id] = sensors[i].values;
        }

        return result;
    }

    result.success = false;
    result.error_message = "Unknown node type: " + ctx.node_type_name;
    return result;
}
```

### 3.5 Integration with NodeEditor

**Execution Trigger:** User clicks "Run Simulation" button in NodeEditor toolbar.

```cpp
// cyxwiz-engine/src/gui/node_editor.cpp

void NodeEditor::ShowToolbar() {
    // ... existing buttons ...

    if (ImGui::Button(ICON_FA_PLAY " Run Simulation")) {
        OnRunSimulation();
    }

    if (ImGui::Button(ICON_FA_STOP " Stop Simulation")) {
        OnStopSimulation();
    }
}

void NodeEditor::OnRunSimulation() {
    if (is_simulating_) return;

    // Create executor
    graph_executor_ = std::make_unique<GraphExecutor>(nodes_, links_);
    graph_executor_->SetPluginEvalCallback([this](const auto& name, const auto& params, const auto& inputs) {
        return plugin_manager_->EvaluateNode(name, params, inputs);
    });

    if (!graph_executor_->Prepare()) {
        ShowErrorDialog("Graph execution failed: " + graph_executor_->GetError());
        return;
    }

    // Start simulation loop in background thread
    sim_thread_ = std::thread([this]() {
        while (is_simulating_) {
            graph_executor_->EvaluateFrame();
            std::this_thread::sleep_for(std::chrono::milliseconds(16));  // ~60 Hz
        }
    });

    is_simulating_ = true;
}
```

### 3.6 Files to Modify/Create

| Action | File | Changes |
|--------|------|---------|
| **CREATE** | `cyxwiz-engine/src/core/graph_executor.h` | GraphExecutor class definition |
| **CREATE** | `cyxwiz-engine/src/core/graph_executor.cpp` | Implementation |
| **MODIFY** | `cyxwiz-engine/include/plugin/interfaces/i_node_provider.h` | Add EvaluateNode() method |
| **MODIFY** | `cyxwiz-engine/src/gui/node_editor.h` | Add simulation state flags |
| **MODIFY** | `cyxwiz-engine/src/gui/node_editor.cpp` | Add Run/Stop buttons, thread management |
| **MODIFY** | `plugins/simulation/mujoco/src/mujoco_plugin.cpp` | Implement EvaluateNode() |

---

## 4. Feature 2: Functional RL Nodes

### 4.1 Current State

RL nodes exist as **visual placeholders** returned by `MuJoCoPlugin::GetNodeTypes()`:
- MuJoCoEnv
- RewardFunction
- ObservationFilter
- RLAgent

They have no runtime logic—code generation returns placeholder Python strings.

### 4.2 Goal

Make these nodes functional for:
1. **Code Generation**: Generate valid Gymnasium/Stable-Baselines3 code
2. **Live Execution**: Provide runtime evaluation for in-engine RL training

### 4.3 Design Strategy

**Dual Implementation Approach:**
- **Code Generation Path**: Enhanced `MuJoCoPlugin::GenerateCode()` for Python export
- **Live Execution Path**: Implemented in `MuJoCoPlugin::EvaluateNode()` for C++ runtime

### 4.4 RewardFunction Node

**Purpose:** Compute shaped reward from raw environment state.

**Parameters:**
- `alive_bonus`: Fixed reward per timestep (encourage survival)
- `ctrl_cost_weight`: Penalty for large control inputs
- `velocity_reward`: Bonus for forward velocity (locomotion)
- `height_penalty`: Penalty if torso height drops below threshold

**Code Generation (Python):**
```python
def compute_reward(env_state):
    reward = 1.0  # alive_bonus
    reward -= 0.1 * np.sum(env_state['ctrl'] ** 2)  # ctrl_cost_weight
    reward += env_state['qvel'][0]  # velocity_reward (forward axis)
    if env_state['qpos'][2] < 0.5:  # height_penalty
        reward -= 10.0
    return reward
```

**Live Execution (C++):**
```cpp
// In MuJoCoPlugin::EvaluateNode()
if (ctx.node_type_name == "RewardFunction") {
    // Extract env state from input pin
    auto env_state = std::get<EnvState>(ctx.input_values.at(env_input_pin_id));

    float reward = 0.0f;
    reward += std::stof(ctx.parameters.at("alive_bonus"));

    // Control cost
    float ctrl_cost = 0.0f;
    for (float u : env_state.ctrl) {
        ctrl_cost += u * u;
    }
    reward -= std::stof(ctx.parameters.at("ctrl_cost_weight")) * ctrl_cost;

    // Velocity reward
    if (ctx.parameters.at("velocity_reward") == "true") {
        reward += env_state.qvel[0];  // Forward velocity
    }

    result.output_values[reward_output_pin_id] = reward;
    return result;
}
```

### 4.5 ObservationFilter Node

**Purpose:** Select/normalize observations for policy input.

**Parameters:**
- `include_qpos`: Include joint positions
- `include_qvel`: Include joint velocities
- `include_sensors`: Include sensor readings
- `normalize`: Apply running mean/std normalization

**Implementation:** Similar dual approach—Python code for export, C++ for live.

### 4.6 RLAgent Node

**Purpose:** RL policy network (PPO/SAC) that learns to maximize reward.

**Complexity:** This is the most complex node—requires full RL training loop.

**Design Decision:**
- **Code Generation**: Generate Stable-Baselines3 code (let SB3 handle training)
- **Live Execution**: Use lightweight C++ PPO implementation (simplified for real-time)

**Code Generation (Python):**
```python
from stable_baselines3 import PPO

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    policy_kwargs=dict(net_arch=[64, 64])
)
model.learn(total_timesteps=1000000)
```

**Live Execution (C++):**
- Implement lightweight PPO in plugin (see Section 5 for details)
- Or: Embed Python interpreter and use SB3 directly (heavier but full-featured)

### 4.7 Files to Modify

| File | Changes |
|------|---------|
| `plugins/simulation/mujoco/src/mujoco_plugin.cpp` | Enhance GenerateCode() for RL nodes |
| `plugins/simulation/mujoco/src/mujoco_plugin.cpp` | Implement EvaluateNode() for RL nodes |
| `plugins/simulation/mujoco/src/rl/reward_shaper.h/cpp` | NEW: RewardFunction logic |
| `plugins/simulation/mujoco/src/rl/observation_filter.h/cpp` | NEW: ObservationFilter logic |
| `plugins/simulation/mujoco/src/rl/ppo_agent.h/cpp` | NEW: Lightweight PPO (see Section 5) |

---

## 5. Feature 3: Live RL Training with Viewport

### 5.1 Goal

Run RL training in CyxWiz Engine with:
- Real-time viewport updates every N episodes
- Metrics reported to TrainingDashboardPanel
- Pause/Resume/Stop controls
- No blocking of UI thread

### 5.2 Architecture: RLTrainingExecutor

**Location:** `cyxwiz-engine/src/core/rl_training_executor.h/cpp`

**Parallel to TrainingExecutor:**
- TrainingExecutor: Supervised learning (Sequential models)
- RLTrainingExecutor: Reinforcement learning (episode loop)

```cpp
// cyxwiz-engine/src/core/rl_training_executor.h
#pragma once

#include <functional>
#include <atomic>
#include <thread>
#include <mutex>

namespace cyxwiz {

struct RLTrainingConfig {
    std::string env_mjcf_path;
    int total_timesteps = 1000000;
    int max_episode_steps = 1000;

    // PPO hyperparameters
    float learning_rate = 3e-4f;
    float gamma = 0.99f;
    float gae_lambda = 0.95f;
    float clip_range = 0.2f;
    int n_steps = 2048;
    int batch_size = 64;
    int n_epochs = 10;
    std::vector<int> hidden_sizes = {64, 64};

    // Reward shaping
    float alive_bonus = 1.0f;
    float ctrl_cost_weight = 0.1f;
    bool velocity_reward = true;
};

struct RLTrainingMetrics {
    int episode_count = 0;
    int total_timesteps = 0;
    float mean_episode_reward = 0.0f;
    float mean_episode_length = 0.0f;
    float policy_loss = 0.0f;
    float value_loss = 0.0f;
    float explained_variance = 0.0f;

    // State
    bool is_training = false;
    bool is_paused = false;
    std::string status_message;

    // History for plotting
    std::vector<float> reward_history;
    std::vector<float> length_history;
};

using RLEpisodeCallback = std::function<void(int episode, float reward, float length)>;
using RLUpdateCallback = std::function<void(int timesteps, const RLTrainingMetrics& metrics)>;
using RLCompleteCallback = std::function<void(const RLTrainingMetrics& final_metrics)>;

/**
 * RLTrainingExecutor - Runs RL training loop with viewport integration
 *
 * Collaborates with MuJoCo plugin for simulation and policy updates.
 * Updates TrainingDashboardPanel with RL-specific metrics.
 */
class RLTrainingExecutor {
public:
    RLTrainingExecutor(const RLTrainingConfig& config);
    ~RLTrainingExecutor();

    // Training control
    void Start(
        RLEpisodeCallback episode_cb = nullptr,
        RLUpdateCallback update_cb = nullptr,
        RLCompleteCallback complete_cb = nullptr
    );
    void Stop();
    void Pause();
    void Resume();

    // State queries
    bool IsTraining() const { return is_training_.load(); }
    bool IsPaused() const { return is_paused_.load(); }
    RLTrainingMetrics GetMetrics() const;

    // Viewport update frequency
    void SetViewportUpdateInterval(int episodes) { viewport_update_interval_ = episodes; }

private:
    void TrainingLoop();
    void RunEpisode();
    void UpdatePolicy();  // PPO update step
    void NotifyViewportUpdate();

    RLTrainingConfig config_;
    std::atomic<bool> is_training_{false};
    std::atomic<bool> is_paused_{false};
    std::atomic<bool> stop_requested_{false};

    std::thread training_thread_;
    mutable std::mutex metrics_mutex_;
    RLTrainingMetrics metrics_;

    RLEpisodeCallback episode_callback_;
    RLUpdateCallback update_callback_;
    RLCompleteCallback complete_callback_;

    int viewport_update_interval_ = 10;  // Update viewport every 10 episodes
};

} // namespace cyxwiz
```

### 5.3 Training Loop Pseudocode

```cpp
void RLTrainingExecutor::TrainingLoop() {
    // Initialize environment via plugin
    auto env = plugin_manager_->GetMuJoCoEnv(config_.env_mjcf_path);

    // Initialize policy network (PPO)
    PPOAgent agent(env->GetObservationDim(), env->GetActionDim(), config_);

    // Rollout buffer
    std::vector<Transition> buffer;

    int episode = 0;
    int timestep = 0;

    while (timestep < config_.total_timesteps && !stop_requested_) {
        if (is_paused_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Run one episode
        auto obs = env->Reset();
        float episode_reward = 0.0f;
        int episode_length = 0;

        for (int step = 0; step < config_.max_episode_steps; ++step) {
            // Policy forward pass
            auto [action, log_prob, value] = agent.SelectAction(obs);

            // Environment step
            auto [next_obs, reward, done, info] = env->Step(action);

            // Store transition
            buffer.push_back({obs, action, reward, value, log_prob});

            episode_reward += reward;
            episode_length++;
            timestep++;

            obs = next_obs;
            if (done) break;
        }

        // Update viewport every N episodes
        if (episode % viewport_update_interval_ == 0) {
            NotifyViewportUpdate();  // Tells plugin to render current state
        }

        // Policy update (PPO)
        if (buffer.size() >= config_.n_steps) {
            UpdatePolicy(agent, buffer);
            buffer.clear();
        }

        // Report metrics
        if (episode_callback_) {
            episode_callback_(episode, episode_reward, episode_length);
        }

        episode++;
    }

    // Final callback
    if (complete_callback_) {
        complete_callback_(GetMetrics());
    }
}
```

### 5.4 Viewport Update Mechanism

**Challenge:** Viewport rendering happens on main thread (ImGui), training on background thread.

**Solution:** Deferred rendering via flag.

```cpp
// In RLTrainingExecutor
void RLTrainingExecutor::NotifyViewportUpdate() {
    // Set flag that tells plugin to render next frame
    plugin_manager_->SetViewportNeedsUpdate("mujoco", true);
}

// In MuJoCoPlugin::RenderPanel()
void MuJoCoPlugin::RenderPanel(const std::string& panel_id, bool* visible) {
    if (panel_id == "mujoco_viewport") {
        if (viewport_needs_update_) {
            // Render current MuJoCo state to texture
            renderer_.Render(env_manager_.GetData());
            viewport_needs_update_ = false;
        }
        viewport_panel_.Render(env_manager_, renderer_, visible);
    }
}
```

### 5.5 Integration with NodeEditor

**User Workflow:**
1. Build RL graph: MuJoCoEnv → ObservationFilter → RLAgent → RewardFunction
2. Click "Train RL" button in NodeEditor toolbar
3. RLTrainingExecutor launches background thread
4. TrainingDashboardPanel shows episode reward, length
5. MuJoCoViewportPanel updates every N episodes
6. User can pause/resume/stop

**Toolbar Button:**
```cpp
void NodeEditor::ShowToolbar() {
    // Detect if graph contains RL nodes
    bool has_rl_nodes = HasNodeOfType(NodeType::PluginCustom, "mujoco:RLAgent");

    if (has_rl_nodes) {
        if (ImGui::Button(ICON_FA_ROBOT " Train RL")) {
            OnStartRLTraining();
        }
    } else {
        if (ImGui::Button(ICON_FA_PLAY " Train Model")) {
            OnStartSupervisedTraining();  // Existing path
        }
    }
}

void NodeEditor::OnStartRLTraining() {
    // Build RLTrainingConfig from graph
    RLTrainingConfig config = ExtractRLConfigFromGraph();

    // Create executor
    rl_executor_ = std::make_unique<RLTrainingExecutor>(config);

    // Setup callbacks
    rl_executor_->Start(
        [this](int ep, float reward, float length) {
            // Update dashboard
            training_dashboard_->UpdateCustomMetric("episode_reward", reward);
            training_dashboard_->UpdateCustomMetric("episode_length", length);
        },
        [this](int ts, const RLTrainingMetrics& m) {
            // Update policy loss, value loss, etc.
            training_dashboard_->UpdateLoss(m.policy_loss);
        },
        [this](const RLTrainingMetrics& m) {
            // Training complete
            ShowNotification("RL training complete! Mean reward: " + std::to_string(m.mean_episode_reward));
        }
    );
}
```

### 5.6 Files to Create/Modify

| Action | File | Purpose |
|--------|------|---------|
| **CREATE** | `cyxwiz-engine/src/core/rl_training_executor.h` | RLTrainingExecutor class |
| **CREATE** | `cyxwiz-engine/src/core/rl_training_executor.cpp` | Training loop implementation |
| **CREATE** | `plugins/simulation/mujoco/src/rl/ppo_agent.h` | PPO policy network |
| **CREATE** | `plugins/simulation/mujoco/src/rl/ppo_agent.cpp` | PPO training logic |
| **MODIFY** | `cyxwiz-engine/src/gui/node_editor.cpp` | Add "Train RL" button |
| **MODIFY** | `plugins/simulation/mujoco/src/mujoco_plugin.h` | Add viewport update flag |

---

## 6. Feature 4: Training Dashboard Integration

### 6.1 Current Dashboard

**Location:** `cyxwiz-engine/src/gui/panels/training_dashboard.h/cpp`

**Existing Metrics:**
- Loss (train/val)
- Accuracy (train/val)
- Throughput (samples/sec)
- Learning rate

**Gap:** No RL-specific metrics (episode reward, length, success rate).

### 6.2 Design: Extensible Metric System

**Add Custom Metrics Support:**

```cpp
// In training_dashboard.h
class TrainingDashboardPanel : public Panel {
public:
    // ... existing methods ...

    // NEW: Custom metric support for RL
    void UpdateCustomMetric(const std::string& name, float value);
    void RegisterCustomPlot(const std::string& name, const std::string& display_name, ImVec4 color);

private:
    std::map<std::string, std::vector<float>> custom_metric_history_;
    std::map<std::string, std::string> custom_plot_ids_;  // name -> PlotManager ID
};
```

**Implementation:**
```cpp
void TrainingDashboardPanel::UpdateCustomMetric(const std::string& name, float value) {
    auto& history = custom_metric_history_[name];
    history.push_back(value);

    if (history.size() > MAX_HISTORY) {
        history.erase(history.begin());
    }

    // Update PlotManager
    if (custom_plot_ids_.count(name)) {
        PlotManager::Instance().UpdateData(custom_plot_ids_[name], history);
    }
}

void TrainingDashboardPanel::RegisterCustomPlot(const std::string& name,
                                                 const std::string& display_name,
                                                 ImVec4 color) {
    std::string plot_id = "custom_" + name;
    PlotManager::Instance().CreateLinePlot(plot_id, display_name);
    PlotManager::Instance().SetLineColor(plot_id, 0, color);
    custom_plot_ids_[name] = plot_id;
}
```

### 6.3 RL Metrics Integration

**Setup in RLTrainingExecutor:**
```cpp
void RLTrainingExecutor::Start(...) {
    // Register RL plots
    training_dashboard_->RegisterCustomPlot("episode_reward", "Episode Reward", ImVec4(0.2f, 0.8f, 0.2f, 1.0f));
    training_dashboard_->RegisterCustomPlot("episode_length", "Episode Length", ImVec4(0.8f, 0.6f, 0.2f, 1.0f));
    training_dashboard_->RegisterCustomPlot("explained_variance", "Explained Variance", ImVec4(0.4f, 0.4f, 1.0f, 1.0f));

    // Start training loop
    training_thread_ = std::thread([this]() { TrainingLoop(); });
}
```

### 6.4 Dashboard Layout

**Three Panels:**
1. **Supervised Learning** (existing): Loss, Accuracy, Throughput
2. **RL Metrics** (new): Episode Reward, Episode Length, Success Rate
3. **Policy Diagnostics** (new): Policy Loss, Value Loss, Explained Variance

**Render Tabs:**
```cpp
void TrainingDashboardPanel::Render() {
    if (ImGui::BeginTabBar("MetricsTabs")) {
        if (ImGui::BeginTabItem("Supervised")) {
            RenderLossChart();
            RenderAccuracyChart();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Reinforcement Learning")) {
            RenderCustomPlot("episode_reward");
            RenderCustomPlot("episode_length");
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Policy Diagnostics")) {
            RenderCustomPlot("policy_loss");
            RenderCustomPlot("value_loss");
            RenderCustomPlot("explained_variance");
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}
```

### 6.5 Files to Modify

| File | Changes |
|------|---------|
| `cyxwiz-engine/src/gui/panels/training_dashboard.h` | Add custom metric API |
| `cyxwiz-engine/src/gui/panels/training_dashboard.cpp` | Implement custom metrics + tabbed layout |

---

## 7. Feature 5: ONNX Policy Export

### 7.1 Purpose

Save trained RL policy as ONNX file for:
- Deployment to embedded systems
- Integration with robotics frameworks (ROS, Isaac Sim)
- Inference in other languages (C++, Rust, JavaScript)

### 7.2 Design

**Export Button in NodeEditor:**
```cpp
if (ImGui::Button(ICON_FA_FILE_EXPORT " Export Policy (ONNX)")) {
    ExportPolicyONNX();
}
```

**Export Implementation:**
```cpp
void NodeEditor::ExportPolicyONNX() {
    if (!rl_executor_ || !rl_executor_->IsTraining()) {
        ShowErrorDialog("No trained policy available. Train an RL agent first.");
        return;
    }

    // Get trained policy network from executor
    auto policy = rl_executor_->GetPolicyNetwork();

    // Convert to ONNX via plugin
    std::string onnx_path = ShowSaveFileDialog("Save ONNX Model", "*.onnx");
    if (onnx_path.empty()) return;

    if (plugin_manager_->ExportPolicyONNX("mujoco", policy, onnx_path)) {
        ShowNotification("Policy exported to " + onnx_path);
    } else {
        ShowErrorDialog("ONNX export failed.");
    }
}
```

### 7.3 Plugin-Side Export

**New Interface Method:**
```cpp
// In INodeProvider or new IModelExporter interface
virtual bool ExportPolicyONNX(
    const std::string& policy_id,
    const std::string& output_path
) = 0;
```

**MuJoCo Plugin Implementation:**
```cpp
bool MuJoCoPlugin::ExportPolicyONNX(const std::string& policy_id, const std::string& output_path) {
    // Get policy network from RLAgent
    auto policy = GetPolicyNetwork(policy_id);
    if (!policy) return false;

    // Use onnxruntime or PyTorch's torch.onnx.export via embedded Python
    // Simplified: Assume we have ONNX exporter utility
    ONNXExporter exporter;
    exporter.SetInputShape({1, policy->GetObservationDim()});  // Batch size 1
    exporter.SetOutputShape({1, policy->GetActionDim()});
    exporter.AddLayer("policy_input", LayerType::Input);

    for (const auto& layer : policy->GetLayers()) {
        exporter.AddLayer(layer.name, layer.type, layer.weights);
    }

    exporter.AddLayer("policy_output", LayerType::Output);

    return exporter.Export(output_path);
}
```

### 7.4 Alternative: Python-Based Export

**If plugin has embedded Python:**
```cpp
bool MuJoCoPlugin::ExportPolicyONNX(const std::string& policy_id, const std::string& output_path) {
    // Execute Python script to export via torch.onnx
    std::string script = R"(
import torch
import torch.onnx

policy = load_policy(')" + policy_id + R"(')
dummy_input = torch.randn(1, policy.observation_dim)

torch.onnx.export(
    policy,
    dummy_input,
    ')" + output_path + R"(',
    export_params=True,
    opset_version=11,
    input_names=['observation'],
    output_names=['action'],
    dynamic_axes={'observation': {0: 'batch_size'}, 'action': {0: 'batch_size'}}
)
    )";

    return python_executor_->ExecuteScript(script);
}
```

### 7.5 Files to Create/Modify

| Action | File | Purpose |
|--------|------|---------|
| **CREATE** | `plugins/simulation/mujoco/src/export/onnx_exporter.h` | ONNX export utility |
| **CREATE** | `plugins/simulation/mujoco/src/export/onnx_exporter.cpp` | ONNX serialization |
| **MODIFY** | `cyxwiz-engine/src/gui/node_editor.cpp` | Add export button/dialog |
| **MODIFY** | `plugins/simulation/mujoco/src/mujoco_plugin.cpp` | Implement ExportPolicyONNX() |

---

## 8. Feature 6: Slider → MuJoCo Live Control

### 8.1 Goal

When in Manual Control mode:
1. User adjusts SignalSlider in Node Editor
2. Slider value immediately updates MuJoCo actuator
3. Viewport re-renders to show updated pose

### 8.2 Current Behavior

- Slider renders in NodeEditor
- Slider value stored in `MLNode::parameters["current_value"]`
- **No connection to MjSimulationExecutor**

### 8.3 Design: Real-Time Parameter Binding

**Architecture:**
```
┌─────────────────┐
│  SignalSlider   │  (NodeEditor)
│  [ImGui Slider] │
└────────┬────────┘
         │ OnValueChanged
         v
┌─────────────────────┐
│  GraphExecutor      │
│  - Updates pin value│
└────────┬────────────┘
         │ Next EvaluateFrame()
         v
┌──────────────────────────┐
│  MuJoCoPlant Node        │  (Plugin via EvaluateNode)
│  - Reads input pin       │
│  - Sets actuator control │
└────────┬─────────────────┘
         │ SetActuatorInputs()
         v
┌──────────────────────────┐
│  MjSimulationExecutor    │
│  - Applies control       │
│  - Calls mj_step()       │
└────────┬─────────────────┘
         │ GetSensorOutputs()
         v
┌──────────────────────────┐
│  MjRenderer              │
│  - Updates texture       │
└──────────────────────────┘
```

### 8.4 Implementation

**Step 1: SignalSlider ImGui Callback**
```cpp
// In NodeEditor::RenderNodes()
if (node.type == NodeType::SignalSlider) {
    float value = std::stof(node.parameters["current_value"]);
    if (ImGui::SliderFloat(("##slider_" + std::to_string(node.id)).c_str(), &value,
                           std::stof(node.parameters["min"]),
                           std::stof(node.parameters["max"]))) {
        // Value changed
        node.parameters["current_value"] = std::to_string(value);

        // Notify GraphExecutor to update pin value immediately
        if (graph_executor_ && is_simulating_) {
            graph_executor_->SetPinValue(node.outputs[0].id, value);
        }
    }
}
```

**Step 2: GraphExecutor Immediate Update**
```cpp
void GraphExecutor::SetPinValue(int pin_id, NodeValue value) {
    std::lock_guard<std::mutex> lock(pin_values_mutex_);
    pin_values_[pin_id] = value;

    // Mark downstream nodes as needing re-evaluation
    MarkDownstreamDirty(pin_id);
}
```

**Step 3: Continuous Evaluation Loop**
```cpp
// In NodeEditor::OnRunSimulation() thread
while (is_simulating_) {
    // Evaluate graph at 60 Hz
    graph_executor_->EvaluateFrame();

    // Throttle to real-time or faster
    std::this_thread::sleep_for(std::chrono::milliseconds(16));  // 60 FPS
}
```

**Step 4: MuJoCo Plant Updates**
- GraphExecutor calls plugin's EvaluateNode("MuJoCoPlant", ...)
- Plugin reads slider value from input pin
- Plugin calls `sim_executor_.SetActuatorInputs()`
- `MjSimulationExecutor::ExecuteOneStep()` applies control and steps physics

**Step 5: Viewport Renders**
- MjSimulationExecutor updates env_manager_->GetData()
- MjRenderer renders on next frame (main thread)
- Viewport shows updated robot pose

### 8.5 Performance Optimization

**Challenge:** Evaluating entire graph 60 times per second may be slow for large graphs.

**Solution:** Dirty flag propagation.

```cpp
void GraphExecutor::MarkDownstreamDirty(int pin_id) {
    // Find all nodes downstream of this pin
    std::set<int> dirty_nodes;
    PropagateDirtyFlag(pin_id, dirty_nodes);

    // Only re-evaluate dirty nodes on next frame
    dirty_node_ids_ = dirty_nodes;
}

bool GraphExecutor::EvaluateFrame() {
    if (dirty_node_ids_.empty()) {
        // Full evaluation
        for (int node_id : execution_order_) {
            EvaluateNode(node_id);
        }
    } else {
        // Partial evaluation (only dirty nodes)
        for (int node_id : dirty_node_ids_) {
            EvaluateNode(node_id);
        }
        dirty_node_ids_.clear();
    }
    return true;
}
```

### 8.6 Files to Modify

| File | Changes |
|------|---------|
| `cyxwiz-engine/src/core/graph_executor.h` | Add SetPinValue(), dirty tracking |
| `cyxwiz-engine/src/core/graph_executor.cpp` | Implement partial evaluation |
| `cyxwiz-engine/src/gui/node_editor.cpp` | Add ImGui slider callback |

---

## 9. Data Flow Diagrams

### 9.1 Manual Control Mode Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                         MANUAL CONTROL MODE                           │
│                  (Simulink-style control engineering)                 │
└──────────────────────────────────────────────────────────────────────┘

USER INTERACTION (Main Thread):
┌───────────────┐
│  NodeEditor   │
│  ┌─────────┐  │     ImGui::SliderFloat(value)
│  │ Slider  │  │ ─────────────────────────┐
│  │ [0.5]   │  │                          │
│  └─────────┘  │                          v
└───────────────┘                  ┌──────────────────┐
                                   │ Slider Value     │
                                   │ Updated in       │
                                   │ parameters map   │
                                   └────────┬─────────┘
                                            │
EVALUATION LOOP (Background Thread @ 60Hz): │
                                            v
                              ┌─────────────────────────┐
                              │    GraphExecutor        │
                              │  EvaluateFrame()        │
                              │  - Read Slider value    │
                              │  - Evaluate SineWave    │
                              │  - Propagate to Plant   │
                              └────────┬────────────────┘
                                       │ EvaluateNode("MuJoCoPlant")
                                       v
                              ┌─────────────────────────┐
                              │  Plugin: MuJoCo DLL     │
                              │  EvaluateNode():        │
                              │  - Gather actuator vals │
                              │  - SetActuatorInputs()  │
                              └────────┬────────────────┘
                                       │
                                       v
                              ┌─────────────────────────┐
                              │  MjSimulationExecutor   │
                              │  ExecuteOneStep():      │
                              │  - Apply ctrl           │
                              │  - mj_step(model, data) │
                              │  - Read sensors         │
                              └────────┬────────────────┘
                                       │
                                       v
                              ┌─────────────────────────┐
                              │  Return sensor outputs  │
                              │  to GraphExecutor       │
                              └────────┬────────────────┘
                                       │
                                       v
                              ┌─────────────────────────┐
                              │  GraphExecutor          │
                              │  - Store sensor values  │
                              │  - Update Scope nodes   │
                              └─────────────────────────┘

RENDERING (Main Thread, on next frame):
                              ┌─────────────────────────┐
                              │  MjViewportPanel        │
                              │  Render():              │
                              │  - Call MjRenderer      │
                              │  - Update texture       │
                              │  - Display ImGui image  │
                              └─────────────────────────┘
```

### 9.2 RL Training Mode Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                          RL TRAINING MODE                             │
│                   (Reinforcement Learning with PPO)                   │
└──────────────────────────────────────────────────────────────────────┘

USER INTERACTION (Main Thread):
┌───────────────┐
│  NodeEditor   │     Click "Train RL" button
│  [Train RL]   │ ─────────────────────────┐
└───────────────┘                          │
                                           v
                              ┌─────────────────────────┐
                              │  RLTrainingExecutor     │
                              │  Start():               │
                              │  - Parse RL config      │
                              │  - Init PPO agent       │
                              │  - Launch thread        │
                              └────────┬────────────────┘
                                       │
TRAINING LOOP (Background Thread):     │
                                       v
                      ┌────────────────────────────────┐
                      │  RLTrainingExecutor::          │
                      │  TrainingLoop():               │
                      │                                │
                      │  for episode in episodes:      │
                      │      obs = env.reset()         │
                      │      for step in steps:        │
                      │          action = agent(obs)   │
                      │          next_obs, reward, ... │
                      │          store_transition()    │
                      │      update_policy()           │
                      │      report_metrics()          │
                      │      if episode % N:           │
                      │          update_viewport()     │
                      └────────┬───────────────────────┘
                               │
                               v (env.step())
                      ┌────────────────────────────────┐
                      │  Plugin: MuJoCo DLL            │
                      │  MjEnvManager::Step():         │
                      │  - Apply action                │
                      │  - mj_step()                   │
                      │  - Compute reward              │
                      │  - Return (obs, reward, done)  │
                      └────────┬───────────────────────┘
                               │
                               v (every N episodes)
                      ┌────────────────────────────────┐
                      │  Plugin: MuJoCo DLL            │
                      │  SetViewportNeedsUpdate(true)  │
                      └────────┬───────────────────────┘
                               │
RENDERING (Main Thread):       │
                               v
                      ┌────────────────────────────────┐
                      │  MjViewportPanel::Render()     │
                      │  if viewport_needs_update_:    │
                      │      renderer_.Render()        │
                      │      viewport_needs_update_=0  │
                      └────────┬───────────────────────┘
                               │
                               v
                      ┌────────────────────────────────┐
                      │  TrainingDashboardPanel        │
                      │  UpdateCustomMetric():         │
                      │  - episode_reward              │
                      │  - episode_length              │
                      │  - policy_loss                 │
                      │  - value_loss                  │
                      └────────────────────────────────┘
```

### 9.3 Component Communication Summary

| From Component | To Component | Data | Mechanism |
|----------------|--------------|------|-----------|
| NodeEditor | GraphExecutor | Graph nodes/links | Direct C++ call |
| GraphExecutor | Plugin (EvaluateNode) | Node params + inputs | INodeProvider callback |
| Plugin | MjSimulationExecutor | Actuator values | Direct call (same DLL) |
| MjSimulationExecutor | Plugin | Sensor outputs | Return value |
| Plugin | GraphExecutor | Output pin values | Return struct |
| RLTrainingExecutor | Plugin (MjEnvManager) | Action vector | DLL function call |
| Plugin | RLTrainingExecutor | (obs, reward, done) | Return struct |
| RLTrainingExecutor | TrainingDashboard | Metrics | Callback function |
| Plugin | MjViewportPanel | Render flag | Atomic bool flag |
| MjViewportPanel | MjRenderer | Render command | Direct call (same DLL) |

---

## 10. Implementation Roadmap

### Phase 4.1: Graph Execution Foundation (Week 1-2)

**Priority:** CRITICAL
**Blockers:** None

- [ ] Create `GraphExecutor` class (`graph_executor.h/cpp`)
- [ ] Implement topological sort for execution order
- [ ] Add pin value storage (std::map<int, NodeValue>)
- [ ] Implement `EvaluateFrame()` loop
- [ ] Add built-in node evaluators (Constant, Slider, SineWave, Scope)
- [ ] Add `INodeProvider::EvaluateNode()` interface method
- [ ] Implement `MuJoCoPlugin::EvaluateNode()` for MuJoCoPlant
- [ ] Wire up "Run Simulation" button in NodeEditor toolbar
- [ ] Test: Slider → MuJoCo Plant → Scope (live control)

**Deliverable:** Manual Control mode fully functional.

### Phase 4.2: RL Node Logic (Week 3-4)

**Priority:** HIGH
**Blockers:** Phase 4.1 complete

- [ ] Enhance `MuJoCoPlugin::GenerateCode()` for RL nodes (Python export)
- [ ] Create `RewardShaper` class for RewardFunction node logic
- [ ] Create `ObservationFilter` class for ObservationFilter node logic
- [ ] Implement `MuJoCoPlugin::EvaluateNode()` for RewardFunction
- [ ] Implement `MuJoCoPlugin::EvaluateNode()` for ObservationFilter
- [ ] Test: Graph with RewardFunction computes correct reward
- [ ] Test: Code generation produces valid SB3 Python code

**Deliverable:** RewardFunction and ObservationFilter nodes work in both paths.

### Phase 4.3: Lightweight PPO Agent (Week 5-7)

**Priority:** HIGH
**Blockers:** Phase 4.2 complete

- [ ] Create `PPOAgent` class (`ppo_agent.h/cpp` in plugin)
- [ ] Implement policy network (MLP with tanh activation)
- [ ] Implement value network
- [ ] Implement rollout buffer
- [ ] Implement advantage estimation (GAE)
- [ ] Implement policy update (PPO clip objective)
- [ ] Implement value update (MSE loss)
- [ ] Test: PPO learns on CartPole-like task

**Deliverable:** Functional C++ PPO implementation.

### Phase 4.4: RL Training Integration (Week 8-9)

**Priority:** HIGH
**Blockers:** Phase 4.3 complete

- [ ] Create `RLTrainingExecutor` class (`rl_training_executor.h/cpp`)
- [ ] Implement training loop (episode rollout + policy update)
- [ ] Add viewport update mechanism (deferred rendering flag)
- [ ] Wire up "Train RL" button in NodeEditor
- [ ] Add pause/resume/stop controls
- [ ] Test: RL training runs in background, viewport updates
- [ ] Test: Training can be paused/resumed without crash

**Deliverable:** Live RL training with viewport feedback.

### Phase 4.5: Training Dashboard RL Metrics (Week 10)

**Priority:** MEDIUM
**Blockers:** Phase 4.4 complete

- [ ] Add custom metric API to TrainingDashboardPanel
- [ ] Implement `UpdateCustomMetric()` and `RegisterCustomPlot()`
- [ ] Add RL metrics tab (Episode Reward, Episode Length, Success Rate)
- [ ] Add policy diagnostics tab (Policy Loss, Value Loss, Explained Variance)
- [ ] Wire up RLTrainingExecutor callbacks to dashboard
- [ ] Test: Dashboard shows live RL metrics during training

**Deliverable:** TrainingDashboard displays RL metrics.

### Phase 4.6: ONNX Policy Export (Week 11)

**Priority:** LOW
**Blockers:** Phase 4.4 complete

- [ ] Create `ONNXExporter` utility class
- [ ] Implement policy serialization to ONNX format
- [ ] Add "Export Policy (ONNX)" button in NodeEditor
- [ ] Wire up export dialog and file save
- [ ] Test: Exported ONNX file loads in onnxruntime
- [ ] Test: Exported policy produces same actions as trained policy

**Deliverable:** Users can export trained policies as ONNX.

### Phase 4.7: Slider Live Control Optimization (Week 12)

**Priority:** MEDIUM
**Blockers:** Phase 4.1 complete

- [ ] Implement dirty flag propagation in GraphExecutor
- [ ] Add partial graph evaluation (only dirty nodes)
- [ ] Optimize slider callback to minimize latency
- [ ] Add performance metrics (FPS, evaluation time)
- [ ] Test: Slider updates feel responsive (<16ms latency)

**Deliverable:** Optimized live control with minimal lag.

---

## 11. Risk Analysis & Mitigations

### Risk 1: DLL Boundary Complexity

**Risk Level:** HIGH

**Description:** Passing C++ objects across DLL boundary is unsafe (ABI differences, heap allocation issues).

**Mitigation:**
- Use only POD types in interface methods (no std::vector, std::string directly)
- Wrap complex types in opaque handles or use serialization (JSON/protobuf)
- Use NodeValue as std::variant (all alternatives are POD or simple types)
- Test on both MSVC and GCC/Clang to catch ABI issues early

**Status:** Partially mitigated by existing plugin system design.

### Risk 2: Performance of Graph Evaluation

**Risk Level:** MEDIUM

**Description:** Evaluating large graphs at 60 Hz may be too slow, causing viewport stuttering.

**Mitigation:**
- Implement dirty flag propagation (only re-evaluate changed nodes)
- Profile with large graphs (100+ nodes) and optimize hot paths
- Consider GPU acceleration for tensor operations in nodes
- Add frame skip option (evaluate graph every N frames)

**Status:** Not yet implemented—requires benchmarking.

### Risk 3: PPO Implementation Correctness

**Risk Level:** MEDIUM

**Description:** RL algorithms are notoriously tricky—bugs can cause training to fail silently.

**Mitigation:**
- Unit test each PPO component (GAE, policy loss, value loss)
- Compare against Stable-Baselines3 on reference tasks (CartPole, Pendulum)
- Add extensive logging of intermediate values (advantages, returns, etc.)
- Use known-good hyperparameters from SB3 zoo

**Status:** Requires careful testing and validation.

### Risk 4: Viewport Rendering Synchronization

**Risk Level:** MEDIUM

**Description:** Rendering from background thread while UI updates can cause race conditions.

**Mitigation:**
- Use atomic bool flag for viewport update signaling (no locks)
- Render to texture on background thread, only swap texture on main thread
- Use double buffering for MjRenderer textures
- Test with TSan (Thread Sanitizer) to catch data races

**Status:** Design uses deferred rendering flag—should be safe.

### Risk 5: Code Generation vs Live Execution Divergence

**Risk Level:** LOW

**Description:** Generated Python code and live C++ execution may behave differently.

**Mitigation:**
- Share logic between code generation and live execution (e.g., RewardShaper class used by both)
- Add integration tests that compare outputs (Python vs C++)
- Document any known differences clearly
- Consider generating Python from live execution state (inverse direction)

**Status:** Low risk if we keep logic in plugin and call from both paths.

### Risk 6: Memory Management with Long-Running Training

**Risk Level:** LOW

**Description:** RL training can run for hours—memory leaks or unbounded growth could crash.

**Mitigation:**
- Use RAII everywhere (smart pointers, scoped locks)
- Circular buffers for metric history (fixed size)
- Periodically call TrimMemory() on DataRegistry
- Test with Valgrind or AddressSanitizer

**Status:** Existing systems already use RAII—low risk.

### Risk 7: Breaking Existing Code Generation

**Risk Level:** VERY LOW

**Description:** Adding graph execution might break existing PyTorch/TF/Keras generation.

**Mitigation:**
- Graph execution is completely orthogonal (separate code path)
- Existing GraphCompiler unchanged
- Add regression tests for code generation before Phase 4 changes
- CI/CD runs all code generation tests on every commit

**Status:** Design guarantees zero impact on existing code generation.

---

## Appendix A: File Change Summary

### New Files to Create

| File | Purpose | Lines Est. |
|------|---------|-----------|
| `cyxwiz-engine/src/core/graph_executor.h` | GraphExecutor class definition | 150 |
| `cyxwiz-engine/src/core/graph_executor.cpp` | Graph evaluation implementation | 500 |
| `cyxwiz-engine/src/core/rl_training_executor.h` | RL training loop | 150 |
| `cyxwiz-engine/src/core/rl_training_executor.cpp` | RL training implementation | 600 |
| `plugins/simulation/mujoco/src/rl/reward_shaper.h` | Reward function logic | 80 |
| `plugins/simulation/mujoco/src/rl/reward_shaper.cpp` | Reward computation | 200 |
| `plugins/simulation/mujoco/src/rl/observation_filter.h` | Observation processing | 80 |
| `plugins/simulation/mujoco/src/rl/observation_filter.cpp` | Observation filtering | 150 |
| `plugins/simulation/mujoco/src/rl/ppo_agent.h` | PPO policy network | 120 |
| `plugins/simulation/mujoco/src/rl/ppo_agent.cpp` | PPO training logic | 800 |
| `plugins/simulation/mujoco/src/export/onnx_exporter.h` | ONNX export utility | 60 |
| `plugins/simulation/mujoco/src/export/onnx_exporter.cpp` | ONNX serialization | 300 |

**Total New Code:** ~3,190 lines

### Files to Modify

| File | Changes | Est. Lines Added |
|------|---------|-----------------|
| `cyxwiz-engine/include/plugin/interfaces/i_node_provider.h` | Add EvaluateNode() method | +30 |
| `cyxwiz-engine/src/gui/node_editor.h` | Add simulation state, executor pointers | +20 |
| `cyxwiz-engine/src/gui/node_editor.cpp` | Add Run/Train buttons, thread management | +200 |
| `cyxwiz-engine/src/gui/panels/training_dashboard.h` | Add custom metric API | +15 |
| `cyxwiz-engine/src/gui/panels/training_dashboard.cpp` | Implement custom metrics, tabs | +150 |
| `plugins/simulation/mujoco/src/mujoco_plugin.h` | Add viewport update flag, new methods | +10 |
| `plugins/simulation/mujoco/src/mujoco_plugin.cpp` | Implement EvaluateNode(), enhance GenerateCode() | +300 |
| `plugins/simulation/mujoco/src/mj_simulation_executor.h` | Add real-time control APIs | +15 |
| `plugins/simulation/mujoco/src/mj_simulation_executor.cpp` | Implement continuous stepping | +100 |

**Total Modified Code:** ~840 lines

**Grand Total:** ~4,030 lines of new/modified code

---

## Appendix B: Testing Strategy

### Unit Tests

- [ ] GraphExecutor topological sort with cycles
- [ ] GraphExecutor pin value propagation
- [ ] Signal node evaluation (Constant, Slider, SineWave)
- [ ] Plugin node evaluation (MuJoCoPlant actuator mapping)
- [ ] RewardShaper with different configurations
- [ ] ObservationFilter normalization
- [ ] PPOAgent policy forward pass
- [ ] PPOAgent advantage estimation (GAE)
- [ ] PPOAgent policy update (clip loss)
- [ ] RLTrainingExecutor episode loop
- [ ] RLTrainingExecutor pause/resume

### Integration Tests

- [ ] Full graph: Slider → MuJoCo Plant → Scope (manual control)
- [ ] Full RL graph: MuJoCoEnv → RLAgent → RewardFunction
- [ ] RL training completes 100 episodes without crash
- [ ] Viewport updates during training
- [ ] Training dashboard shows correct metrics
- [ ] Exported ONNX file matches trained policy
- [ ] Code generation produces valid Python for RL nodes

### Performance Tests

- [ ] Graph evaluation FPS with 50 nodes
- [ ] Graph evaluation FPS with 100 nodes
- [ ] Slider latency (<16ms from UI to viewport)
- [ ] RL training throughput (episodes/sec)
- [ ] Memory usage during 1-hour training run

### Stress Tests

- [ ] 1000 episode training run (memory stability)
- [ ] Rapid pause/resume/stop during training
- [ ] Multiple simultaneous training runs
- [ ] Large RL graph (10+ RL nodes)

---

## Appendix C: Future Enhancements (Post-Phase 4)

**Not in scope for Phase 4, but documented for future planning:**

1. **Multi-Agent RL:** Support for MARL algorithms (MAPPO, QMIX)
2. **Distributed Training:** RL training across multiple nodes (PPO with multiple workers)
3. **Model Zoo Integration:** Pre-trained policies downloadable from cloud
4. **Sim-to-Real Transfer:** Domain randomization tools for robotics
5. **Curriculum Learning:** Automatic task difficulty progression
6. **Hyperparameter Tuning:** Optuna integration for automatic RL tuning
7. **3D Trajectory Recording:** Record and replay learned behaviors
8. **VR/AR Visualization:** Immersive visualization of RL training

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-02 | Claude Code | Initial document creation |

---

**END OF DOCUMENT**
