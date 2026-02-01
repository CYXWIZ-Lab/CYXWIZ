# MuJoCo Simulink-Style Training in CyxWiz Engine

Replicate the MATLAB/Simulink MuJoCo workflow: visual node graph → real-time physics simulation + RL training, with 3D viewport showing the robot moving.

---

## Two Workflows

### Workflow 1: Manual Control (like the MATLAB video)
```
[Constant: 0] ──► shoulder_pan  ┐
[Constant: π] ──► shoulder_lift │  MuJoCo Plant    ──► [Scope: sensors]
[Slider: 0-2π] ► elbow         │  (scene.xml)     ──► [Viewport: rgb]
[Constant: 0] ──► wrist_1      │
[Constant: 0] ──► wrist_2      │
[Constant: 0] ──► wrist_3      ┘
```
User wires values → actuators, hits Play, sees robot move in 3D viewport.

### Workflow 2: RL Training
```
[MuJoCo Plant] ──► [Reward Function] ──► [RL Agent (PPO/SAC)]
   (scene.xml)        (alive+ctrl)         (trains policy)
       │                                        │
       └────────── [Viewport: live training] ◄──┘
```
Agent trains via episodes, 3D viewport shows the robot learning.

---

## What Exists vs What's Missing

| Component | Status | Notes |
|-----------|--------|-------|
| MuJoCo plugin (physics, renderer, viewport) | ✅ Done | `MjEnvManager`, `MjRenderer`, `MjViewportPanel` |
| Environment library (7 built-in) | ✅ Done | `MjEnvLibrary` with cards UI |
| RL node types (4 nodes) | ✅ Done | MuJoCoEnv, RewardFunction, ObservationFilter, RLAgent |
| Code generation from nodes | ✅ Done | Generates Python/SB3 code |
| Training executor (ML training) | ✅ Done | `TrainingExecutor` runs SequentialModel in background thread |
| **Dynamic pins from MJCF** | ❌ Missing | Pins are static; need per-actuator/sensor pins |
| **Simulation executor** | ❌ Missing | No real-time node graph stepping (only code gen) |
| **Signal source nodes** | ❌ Missing | No Constant, Slider, Sine, Step nodes |
| **RL simulation loop** | ❌ Missing | MuJoCo step/reset not wired to training loop |
| **Menagerie models** | ❌ Missing | Only 7 basic envs, not 70+ from Menagerie |

---

## Implementation Plan

### Part 1: MJCF Parser (`mj_mjcf_parser.h/cpp`) — NEW

Standalone utility: load MJCF XML via MuJoCo API, extract actuator/sensor/camera names.

```cpp
// plugins/simulation/mujoco/src/mj_mjcf_parser.h
struct MjcfModelInfo {
    std::string model_name;
    struct Actuator { std::string name; std::string joint; float range_low, range_high; };
    struct Sensor { std::string name; std::string type; int dim; };
    struct Camera { std::string name; };
    std::vector<Actuator> actuators;
    std::vector<Sensor> sensors;
    std::vector<Camera> cameras;
    int nq, nv;  // generalized coords/velocities
};

MjcfModelInfo ParseMjcfFile(const std::string& path);
```

Uses `mj_loadXML()` → reads `m->nu`, `mj_id2name(m, mjOBJ_ACTUATOR, i)`, `m->actuator_ctrlrange`, etc. → `mj_deleteModel()`.

**Files:** `plugins/simulation/mujoco/src/mj_mjcf_parser.h`, `mj_mjcf_parser.cpp`

---

### Part 2: Dynamic Pin Support in Node Editor

Add ability for plugin nodes to update their pins at runtime when parameters change.

**`cyxwiz-engine/src/plugin/plugin_types.h`** — extend `PluginNodeTypeInfo`:
```cpp
bool supports_dynamic_pins = false;
```

**`cyxwiz-engine/src/plugin/interfaces/i_node_provider.h`** — new method:
```cpp
struct DynamicPinResult {
    std::vector<PinInfo> pins;
    std::map<std::string, std::string> metadata;
};
virtual DynamicPinResult ResolveDynamicPins(
    const std::string& node_type,
    const std::map<std::string, std::string>& params) { return {}; }
```

**`cyxwiz-engine/src/gui/node_editor.h`** — `MLNode` extension:
```cpp
bool has_dynamic_pins = false;
std::string resolved_config;  // tracks which config was resolved
```

**`cyxwiz-engine/src/gui/node_editor.cpp`** — pin rebuild:
- When Properties panel changes `mjcf_path` on a `supports_dynamic_pins` node
- Call plugin's `ResolveDynamicPins()` → get new pin list
- Save connections by name → clear pins → create new pins → restore matching connections

**`cyxwiz-engine/src/gui/node_editor_nodes.cpp`** — handle dynamic pins in `CreateNode()`

---

### Part 3: MuJoCo Plant Node (replaces static MuJoCoEnv)

**`plugins/simulation/mujoco/src/mujoco_plugin.h/cpp`**:

Replace `MuJoCoEnv` with `MuJoCoPlant` node type:
- Default pins: `u` (input vector), `sensor` (output), `rgb` (output), `depth` (output)
- `supports_dynamic_pins = true`
- When `mjcf_path` changes → `ResolveDynamicPins()` parses MJCF → returns per-actuator input pins + per-sensor output pins
- Properties dialog shows: Model Info, Control Vector Type, Sensor Bus Type, Sample Time (like MATLAB)

**Interface mode** (parameter `interface`):
- `"bus"` — Individual named pins per actuator (shoulder_pan, shoulder_lift, etc.)
- `"vector"` — Single `u` input pin (float array)

---

### Part 4: Simulation Executor — NEW

The core missing piece: **execute the node graph in real-time**, stepping MuJoCo physics.

**`plugins/simulation/mujoco/src/mj_simulation_executor.h/cpp`** — NEW

```cpp
class MjSimulationExecutor {
public:
    void SetNodeGraph(const std::vector<NodeInfo>& nodes,
                      const std::vector<LinkInfo>& links);

    // Workflow 1: Manual control loop
    void StartSimulation();   // Background thread: read inputs → mj_step → update outputs → render
    void StopSimulation();
    void PauseSimulation();
    void StepOnce();          // Single physics step

    // Workflow 2: RL training loop
    void StartRLTraining();   // Background thread: episode loop with agent
    void StopRLTraining();

    // State
    bool IsRunning() const;
    float GetSimTime() const;
    int GetEpisodeCount() const;
    float GetMeanReward() const;

private:
    // Manual control loop (runs in background thread)
    void SimulationLoop() {
        while (running_) {
            // 1. Read actuator values from connected source nodes
            for (int i = 0; i < model_->nu; i++) {
                data_->ctrl[i] = GetInputValue(actuator_pin_ids_[i]);
            }
            // 2. Step physics
            mj_step(model_, data_);
            // 3. Push sensor outputs to connected nodes
            PushSensorOutputs();
            // 4. Render frame to viewport
            renderer_->RenderFrame(model_, data_);
            // 5. Sleep to match sample_time
            SleepUntilNextStep();
        }
    }

    // RL training loop
    void RLTrainingLoop() {
        while (running_ && episode < max_episodes) {
            env_.Reset();
            while (!done) {
                // 1. Get observation
                auto obs = env_.GetObservation();
                // 2. Agent selects action
                auto action = agent_->SelectAction(obs);
                // 3. Step environment
                auto result = env_.Step(action);
                // 4. Agent learns
                agent_->Update(obs, action, result.reward, result.observation);
                // 5. Render current state
                renderer_->RenderFrame(model_, data_);
                // 6. Report metrics
                ReportMetrics();
            }
        }
    }

    // Read value from a connected source node (Constant, Slider, Sine, etc.)
    float GetInputValue(int pin_id);
};
```

**Integration with MuJoCo Plugin:**
- `MuJoCoPlugin` owns the executor
- Viewport panel's Play/Pause/Step buttons call executor methods
- Node editor's "Train" button triggers `StartRLTraining()` when MuJoCo nodes are present

---

### Part 5: Signal Source Nodes

New plugin node types for wiring values to actuators:

| Node | Output | Parameters | Runtime Behavior |
|------|--------|------------|-----------------|
| **Constant** | float | `value` | Returns fixed value |
| **Slider** | float | `min`, `max`, `value` | Returns current slider position (interactive) |
| **Sine Wave** | float | `amplitude`, `frequency`, `phase` | Returns `A * sin(2πft + φ)` at sim time |
| **Step** | float | `step_time`, `initial`, `final` | Returns `initial` before t, `final` after |
| **Ramp** | float | `slope`, `start_time` | Returns `slope * (t - start_time)` |
| **Scope** | — | — | Plots input signal over time (ImPlot) |

These are registered via `GetNodeTypes()` in the MuJoCo plugin and executed by `MjSimulationExecutor`.

**`plugins/simulation/mujoco/src/mujoco_plugin.cpp`** — add to `GetNodeTypes()` and `GenerateCode()`.

---

### Part 6: Viewport Integration

**`plugins/simulation/mujoco/src/mj_viewport_panel.cpp`** — enhance:

- **Play** button → `executor.StartSimulation()` or `executor.StartRLTraining()`
- **Pause** button → `executor.PauseSimulation()`
- **Step** button → `executor.StepOnce()`
- **Stop** button → `executor.StopSimulation()`
- **Status bar**: sim time, episode count, reward, FPS
- **Mode toggle**: "Manual Control" vs "RL Training"

---

### Part 7: Menagerie Model Library

Extend `MjEnvLibrary` with 70+ models from MuJoCo Menagerie:

**Bundled** (10 models shipped in `assets/menagerie/`):
- `franka_emika_panda`, `universal_robots_ur5e`, `unitree_go2`, `unitree_g1`, `agility_cassie`, `anymal_c`, `shadow_hand`, `robotiq_2f85`, `google_robot`, `bitcraze_crazyflie_2`

**Downloadable** (~60 more): Full catalog, downloaded on demand via `cpp-httplib` from GitHub raw content.

**Files:** `mj_env_library.h/cpp` (extend), `mj_menagerie_downloader.h/cpp` (new), `mj_env_browser_panel.h/cpp` (download UI)

---

## Files Summary

| File | Action |
|------|--------|
| `plugins/simulation/mujoco/src/mj_mjcf_parser.h` | **NEW** — MJCF model introspection |
| `plugins/simulation/mujoco/src/mj_mjcf_parser.cpp` | **NEW** |
| `plugins/simulation/mujoco/src/mj_simulation_executor.h` | **NEW** — Real-time sim + RL training loops |
| `plugins/simulation/mujoco/src/mj_simulation_executor.cpp` | **NEW** |
| `plugins/simulation/mujoco/src/mj_menagerie_downloader.h` | **NEW** — GitHub model downloader |
| `plugins/simulation/mujoco/src/mj_menagerie_downloader.cpp` | **NEW** |
| `plugins/simulation/mujoco/src/mujoco_plugin.h` | Add executor, dynamic pins, new nodes |
| `plugins/simulation/mujoco/src/mujoco_plugin.cpp` | Plant node, signal nodes, code gen, executor wiring |
| `plugins/simulation/mujoco/src/mj_env_library.h` | Extend EnvInfo, add Menagerie catalog |
| `plugins/simulation/mujoco/src/mj_env_library.cpp` | 70+ model entries |
| `plugins/simulation/mujoco/src/mj_viewport_panel.h` | Play/Pause/Step/Stop + mode toggle |
| `plugins/simulation/mujoco/src/mj_viewport_panel.cpp` | Executor integration |
| `plugins/simulation/mujoco/src/mj_env_browser_panel.h` | Download state UI |
| `plugins/simulation/mujoco/src/mj_env_browser_panel.cpp` | Download buttons, progress |
| `plugins/simulation/mujoco/CMakeLists.txt` | Add new sources, link httplib |
| `cyxwiz-engine/src/plugin/plugin_types.h` | `supports_dynamic_pins` field |
| `cyxwiz-engine/src/plugin/interfaces/i_node_provider.h` | `ResolveDynamicPins()` method |
| `cyxwiz-engine/src/gui/node_editor.h` | Dynamic pin fields in MLNode |
| `cyxwiz-engine/src/gui/node_editor.cpp` | Pin rebuild logic |
| `cyxwiz-engine/src/gui/node_editor_nodes.cpp` | Dynamic pin creation |

---

## Implementation Order

1. **Part 1**: MJCF Parser (standalone, testable)
2. **Part 2**: Dynamic pin support in node editor
3. **Part 3**: MuJoCo Plant node with dynamic actuator/sensor pins
4. **Part 5**: Signal source nodes (Constant, Slider, Sine, etc.)
5. **Part 4**: Simulation executor (manual control loop + RL training loop)
6. **Part 6**: Viewport Play/Pause/Step integration
7. **Part 7**: Menagerie model library + downloader
8. Build & verify
