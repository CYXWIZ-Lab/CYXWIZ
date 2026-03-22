# CyxWiz MuJoCo Simulation Plugin — Architecture Design

> Design date: 2026-01-31

## Overview

The MuJoCo plugin embeds Google DeepMind's MuJoCo physics engine directly into CyxWiz as a native C++ plugin. It provides:

1. **3D Viewport** — Real-time physics visualization in an ImGui panel
2. **Environment Nodes** — Visual node editor nodes for RL environment configuration
3. **Training Integration** — Step-based RL training with observation/action/reward flow
4. **Model Browser** — Load MJCF/URDF robot models with preview
5. **Built-in Environments** — Standard Gymnasium-compatible envs (Ant, HalfCheetah, Humanoid, etc.)

## How It Works — User Perspective

### Workflow 1: Train an RL Agent on a Robot

```
1. Open Plugin Manager → Enable "MuJoCo Simulation"
2. Node Editor: Add "MuJoCo Env" node → select "Ant-v5" from dropdown
3. Node Editor: Connect Env → RL Agent (PPO/SAC) → Env feedback loop
4. Open "MuJoCo Viewport" panel → see 3D ant robot
5. Click "Train" → agent trains, viewport shows live simulation
6. Metrics panel shows reward curve, episode length
7. Export trained policy as PyTorch/ONNX model
```

### Workflow 2: Custom Robot with MJCF

```
1. Asset Browser: Import custom_robot.xml (MJCF format)
2. Node Editor: Add "MuJoCo Env" node → point to custom_robot.xml
3. Configure observation space, action space, reward function
4. Train with any RL algorithm
```

### Workflow 3: Headless Batch Training

```
1. Same node graph, but uncheck "Enable Viewport"
2. Physics runs without rendering → 10-100x faster
3. Used for hyperparameter sweeps, distributed training
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CyxWiz Engine                               │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │   Node Editor     │  │  Training System  │  │  Data Registry │ │
│  │                    │  │                    │  │                │ │
│  │  [MuJoCo Env]─────│──│──[RL Agent]────────│──│──[Replay Buf] │ │
│  │  [Reward Fn ]     │  │  [PPO/SAC/DQN]     │  │                │ │
│  │  [Obs Filter]     │  │                    │  │                │ │
│  └────────┬───────────┘  └────────┬───────────┘  └────────────────┘ │
│           │                       │                                  │
│  ┌────────▼───────────────────────▼──────────────────────────────┐ │
│  │              MuJoCo Plugin (DLL)                               │ │
│  │                                                                 │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐ │ │
│  │  │ MjEnvManager│  │ MjRenderer   │  │ MjEnvRegistry        │ │ │
│  │  │             │  │              │  │                      │ │ │
│  │  │ • mjModel*  │  │ • mjvScene   │  │ • Ant-v5             │ │ │
│  │  │ • mjData*   │  │ • mjvCamera  │  │ • HalfCheetah-v5     │ │ │
│  │  │ • Step()    │  │ • mjrContext │  │ • Humanoid-v5        │ │ │
│  │  │ • Reset()   │  │ • FBO→Tex   │  │ • Custom MJCF...     │ │ │
│  │  │ • GetObs()  │  │ • Render()   │  │                      │ │ │
│  │  └──────┬──────┘  └──────┬───────┘  └──────────────────────┘ │ │
│  │         │                │                                     │ │
│  │  ┌──────▼────────────────▼────────────────────────────────┐  │ │
│  │  │              libmujoco (C library)                      │  │ │
│  │  │  mj_step() | mj_forward() | mjv_updateScene()          │  │ │
│  │  │  mjr_render() | mjr_readPixels() | mj_loadXML()        │  │ │
│  │  └────────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Plugin Interfaces Used

| Interface | Usage |
|-----------|-------|
| `IPlugin` | Lifecycle: load MuJoCo library, initialize renderer, cleanup |
| `INodeProvider` | 3 node types: MuJoCo Env, Reward Function, Observation Filter |
| `IPanelProvider` | 2 panels: MuJoCo Viewport (3D), Environment Browser |
| `ITrainingHook` | OnEpochEnd: log episode rewards, render preview frame |
| `IDataProvider` | Load MJCF/URDF models as "datasets" for the environment |

---

## Key Components

### 1. MjEnvManager — Physics Engine Wrapper

Core class managing MuJoCo model, data, and simulation stepping.

```cpp
class MjEnvManager {
public:
    // Lifecycle
    bool LoadModel(const std::string& mjcf_path);   // mj_loadXML
    bool LoadModelFromString(const std::string& xml); // mj_loadXML from string
    void Reset();                                      // mj_resetData
    void Close();                                      // mj_deleteData, mj_deleteModel

    // Simulation
    StepResult Step(const std::vector<float>& action); // mj_step, returns obs/reward/done
    std::vector<float> GetObservation() const;          // Read qpos, qvel, sensor data
    float GetReward() const;                            // Compute reward from state
    bool IsDone() const;                                // Terminal condition check

    // State access
    int GetObservationDim() const;
    int GetActionDim() const;
    const mjModel* GetModel() const { return model_; }
    const mjData* GetData() const { return data_; }

private:
    mjModel* model_ = nullptr;
    mjData* data_ = nullptr;

    // Environment config
    std::string env_id_;
    int max_episode_steps_ = 1000;
    int current_step_ = 0;
};

struct StepResult {
    std::vector<float> observation;
    float reward;
    bool terminated;
    bool truncated;
    std::map<std::string, float> info;
};
```

### 2. MjRenderer — 3D Viewport Rendering

Renders MuJoCo scene to an OpenGL texture for display in ImGui.

```cpp
class MjRenderer {
public:
    bool Initialize(mjModel* model, int width, int height);
    void Shutdown();

    // Render one frame to internal FBO, return OpenGL texture ID
    GLuint RenderFrame(const mjModel* model, const mjData* data);

    // Camera control (mouse interaction in ImGui panel)
    void RotateCamera(float dx, float dy);
    void ZoomCamera(float dz);
    void PanCamera(float dx, float dy);
    void ResetCamera();
    void TrackBody(int body_id);

    // Settings
    void SetResolution(int width, int height);
    void SetRenderFlags(int flags);  // wireframe, contact points, etc.
    bool IsEnabled() const { return enabled_; }
    void SetEnabled(bool enabled) { enabled_ = enabled; }

private:
    // MuJoCo visualization structs
    mjvCamera cam_;
    mjvOption opt_;
    mjvScene scn_;
    mjrContext con_;

    // OpenGL resources
    GLuint fbo_ = 0;           // Framebuffer object
    GLuint color_tex_ = 0;     // Color attachment (displayed in ImGui)
    GLuint depth_rbo_ = 0;     // Depth renderbuffer
    int width_ = 640;
    int height_ = 480;
    bool enabled_ = true;
    bool initialized_ = false;
};
```

**Rendering Pipeline:**
```
1. mjv_updateScene(model, data, &opt_, NULL, &cam_, mjCAT_ALL, &scn_)
2. glBindFramebuffer(GL_FRAMEBUFFER, fbo_)
3. mjr_render(viewport, &scn_, &con_)
4. glBindFramebuffer(GL_FRAMEBUFFER, 0)
5. Return color_tex_ → ImGui::Image((ImTextureID)color_tex_, size)
```

### 3. MjEnvRegistry — Built-in Environments

Pre-configured environments matching Gymnasium MuJoCo standard.

```cpp
struct MjEnvConfig {
    std::string env_id;           // e.g. "Ant-v5"
    std::string display_name;     // e.g. "Ant"
    std::string category;         // "Locomotion", "Manipulation", "Balancing"
    std::string description;
    std::string mjcf_path;        // Path to bundled MJCF XML
    int obs_dim;
    int act_dim;
    int max_episode_steps;
    float reward_threshold;       // "Solved" threshold
};

class MjEnvRegistry {
public:
    static MjEnvRegistry& Instance();

    void RegisterBuiltinEnvs();   // Register all standard envs
    void RegisterCustomEnv(const MjEnvConfig& config);

    const MjEnvConfig* GetEnv(const std::string& env_id) const;
    std::vector<const MjEnvConfig*> GetEnvsByCategory(const std::string& cat) const;
    std::vector<const MjEnvConfig*> GetAllEnvs() const;

private:
    std::unordered_map<std::string, MjEnvConfig> envs_;
};
```

**Built-in Environments (bundled MJCF files):**

| Category | Environment | obs_dim | act_dim | Description |
|----------|------------|---------|---------|-------------|
| Locomotion | Ant-v5 | 27 | 8 | 4-legged creature |
| Locomotion | HalfCheetah-v5 | 17 | 6 | 2D running |
| Locomotion | Hopper-v5 | 11 | 3 | 1-legged hopping |
| Locomotion | Humanoid-v5 | 376 | 17 | Bipedal walking |
| Locomotion | Walker2d-v5 | 17 | 6 | 2D bipedal walking |
| Locomotion | Swimmer-v5 | 8 | 2 | 2D swimming |
| Balancing | InvertedPendulum-v5 | 4 | 1 | Balance a pole |
| Balancing | InvertedDoublePendulum-v5 | 11 | 1 | Double pendulum |
| Control | Reacher-v5 | 11 | 2 | Reach a target |
| Manipulation | Pusher-v5 | 23 | 7 | Push object to goal |

MJCF XML files are bundled from the [Gymnasium MuJoCo assets](https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/mujoco/assets).

### 4. Node Types (INodeProvider)

#### MuJoCo Env Node

Central node representing a simulation environment.

```
┌──────────────────────────┐
│    MuJoCo Environment    │
│  ┌────────────────────┐  │
│  │ Env: [Ant-v5    ▼] │  │
│  │ Steps: [1000     ] │  │
│  │ Render: [✓]        │  │
│  │ Frame Skip: [5   ] │  │
│  └────────────────────┘  │
│                          │
│  ○ action (input)        │
│                          │
│         observation ○    │
│              reward ○    │
│                done ○    │
│                info ○    │
└──────────────────────────┘
```

**Parameters:**
- `env_id` — dropdown of registered environments
- `mjcf_path` — custom MJCF file (overrides env_id)
- `max_episode_steps` — auto-truncation
- `frame_skip` — physics steps per action
- `render_enabled` — toggle viewport rendering

#### Reward Function Node

Custom reward shaping (optional, overrides default reward).

```
┌──────────────────────────┐
│    Reward Function       │
│  ┌────────────────────┐  │
│  │ Type: [Custom   ▼] │  │
│  │ Formula:            │  │
│  │ [forward_vel - 0.1 │  │
│  │  * ctrl_cost      ]│  │
│  └────────────────────┘  │
│                          │
│  ○ observation (input)   │
│  ○ action (input)        │
│         reward ○ (out)   │
└──────────────────────────┘
```

#### Observation Filter Node

Normalize, clip, or select observation features.

```
┌──────────────────────────┐
│   Observation Filter     │
│  ┌────────────────────┐  │
│  │ Normalize: [✓]     │  │
│  │ Clip: [-10, 10]    │  │
│  │ Features: [all  ▼] │  │
│  └────────────────────┘  │
│                          │
│  ○ raw_obs (input)       │
│      filtered_obs ○ (out)│
└──────────────────────────┘
```

### 5. Panels (IPanelProvider)

#### MuJoCo Viewport Panel

3D visualization of the running simulation.

```
┌─────────────────────────────────────────────────┐
│ MuJoCo Viewport                          [─][□] │
├─────────────────────────────────────────────────┤
│ [▶ Play] [⏸ Pause] [⟲ Reset] [📷 Screenshot]  │
│ Env: Ant-v5 | Step: 1,234 | Reward: 45.2       │
├─────────────────────────────────────────────────┤
│                                                  │
│            ┌──────────────────┐                  │
│            │                  │                  │
│            │   3D Rendered    │                  │
│            │   MuJoCo Scene   │                  │
│            │   (OpenGL tex)   │                  │
│            │                  │                  │
│            └──────────────────┘                  │
│                                                  │
│  Mouse: LMB=Rotate  RMB=Pan  Scroll=Zoom        │
├─────────────────────────────────────────────────┤
│ Render: [✓] Contacts [✓] Forces [ ] Wireframe   │
│ Camera: [Free ▼]  Track Body: [torso ▼]         │
│ Speed: [1.0x ▼]  Resolution: [640x480 ▼]        │
└─────────────────────────────────────────────────┘
```

**Implementation:**
- `ImGui::Image((ImTextureID)(intptr_t)renderer_.RenderFrame(...), size)`
- Mouse interaction via `ImGui::IsItemHovered()` + `ImGui::GetIO().MouseDelta`
- Render controls as ImGui widgets below the viewport

#### Environment Browser Panel

Browse, preview, and configure environments.

```
┌─────────────────────────────────────────────────┐
│ Environment Browser                      [─][□] │
├──────────────┬──────────────────────────────────┤
│ Categories   │ Ant-v5                            │
│              │ ────────────────────               │
│ ▶ Locomotion │ 4-legged creature that walks      │
│   HalfCheetah│ forward using 8 actuators.        │
│   Ant ←──────│                                    │
│   Hopper     │ Observation: 27 dims              │
│   Humanoid   │ Action: 8 dims (continuous)       │
│   Walker2d   │ Reward: forward velocity - costs  │
│   Swimmer    │ Max steps: 1000                   │
│              │                                    │
│ ▶ Balancing  │ [Add to Node Editor]              │
│ ▶ Manipulation│ [Load Custom MJCF...]            │
│ ▶ Custom     │                                    │
└──────────────┴──────────────────────────────────┘
```

---

## Training Integration

### RL Training Loop

The MuJoCo plugin integrates with CyxWiz's training system:

```
┌────────────┐    action     ┌────────────┐
│  RL Agent  │──────────────▶│  MuJoCo    │
│  (Policy)  │               │  Env       │
│            │◀──────────────│            │
│  PPO/SAC/  │  obs, reward  │  Step()    │
│  DQN/A2C   │  done, info   │  Reset()   │
└────────────┘               └────────────┘
       │                           │
       ▼                           ▼
 ┌──────────┐              ┌──────────────┐
 │ Replay   │              │ MjRenderer   │
 │ Buffer   │              │ (viewport)   │
 └──────────┘              └──────────────┘
```

**Training flow per step:**
1. Agent selects action from policy network: `action = policy(observation)`
2. Environment steps: `result = env.Step(action)`
3. Store transition: `buffer.Add(obs, action, reward, next_obs, done)`
4. If `done`: `env.Reset()`
5. Every N steps: update policy from buffer
6. Optionally: render frame for viewport

**Headless mode:** Skip step 6 for maximum throughput. Physics runs at ~1M steps/sec on CPU.

### ITrainingHook Integration

```cpp
void MuJoCoPlugin::OnEpochEnd(TrainingContext& ctx) {
    // Log RL-specific metrics
    ctx.custom_metrics["mean_episode_reward"] = env_manager_.GetMeanEpisodeReward();
    ctx.custom_metrics["mean_episode_length"] = env_manager_.GetMeanEpisodeLength();
    ctx.custom_metrics["success_rate"] = env_manager_.GetSuccessRate();

    // Render one frame for viewport preview (if enabled)
    if (renderer_.IsEnabled()) {
        renderer_.RenderFrame(env_manager_.GetModel(), env_manager_.GetData());
    }
}

bool MuJoCoPlugin::ShouldStopEarly(const TrainingContext& ctx) {
    // Stop if reward exceeds solved threshold
    auto it = ctx.custom_metrics.find("mean_episode_reward");
    if (it != ctx.custom_metrics.end()) {
        auto* env_config = MjEnvRegistry::Instance().GetEnv(current_env_id_);
        if (env_config && it->second >= env_config->reward_threshold) {
            return true;  // Environment solved!
        }
    }
    return false;
}
```

---

## Code Generation

The MuJoCo Env node generates PyTorch RL training code:

```python
# Generated by CyxWiz - MuJoCo Environment
import gymnasium as gym
import torch

# Create environment
env = gym.make("Ant-v5", render_mode="human")
obs, info = env.reset()

# Training loop
for episode in range(1000):
    done = False
    total_reward = 0
    obs, info = env.reset()

    while not done:
        action = policy(torch.tensor(obs, dtype=torch.float32))
        obs, reward, terminated, truncated, info = env.step(action.numpy())
        done = terminated or truncated
        total_reward += reward

    print(f"Episode {episode}: reward = {total_reward:.2f}")

env.close()
```

---

## File Structure

```
plugins/simulation/mujoco/
├── plugin.json                    # Manifest
├── CMakeLists.txt                 # Build config
├── src/
│   ├── mujoco_plugin.h/cpp        # IPlugin + INodeProvider + IPanelProvider + ITrainingHook
│   ├── mj_env_manager.h/cpp       # Physics wrapper (mjModel, mjData, Step/Reset)
│   ├── mj_renderer.h/cpp          # OpenGL FBO rendering for ImGui viewport
│   ├── mj_env_registry.h/cpp      # Built-in environment definitions
│   ├── mj_viewport_panel.h/cpp    # 3D viewport ImGui panel
│   └── mj_browser_panel.h/cpp     # Environment browser ImGui panel
├── assets/
│   ├── ant.xml                    # Bundled MJCF models
│   ├── half_cheetah.xml
│   ├── hopper.xml
│   ├── humanoid.xml
│   ├── walker2d.xml
│   ├── swimmer.xml
│   ├── inverted_pendulum.xml
│   ├── inverted_double_pendulum.xml
│   ├── reacher.xml
│   └── pusher.xml
└── bin/
    └── mujoco_plugin.dll          # Built plugin
```

---

## Dependencies

| Dependency | Version | How Acquired |
|-----------|---------|-------------|
| MuJoCo | 3.x | FetchContent or pre-built binary |
| cyxwiz-plugin-sdk | (internal) | Link from engine build |
| OpenGL | 3.3+ | System (already available) |
| GLFW | 3.x | vcpkg (already available) |

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.20)
project(mujoco_plugin LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

# Fetch MuJoCo
include(FetchContent)
FetchContent_Declare(
    mujoco
    GIT_REPOSITORY https://github.com/google-deepmind/mujoco.git
    GIT_TAG        3.2.7
)
FetchContent_MakeAvailable(mujoco)

add_library(mujoco_plugin SHARED
    src/mujoco_plugin.cpp
    src/mj_env_manager.cpp
    src/mj_renderer.cpp
    src/mj_env_registry.cpp
    src/mj_viewport_panel.cpp
    src/mj_browser_panel.cpp
)

target_link_libraries(mujoco_plugin PRIVATE
    cyxwiz-plugin-sdk
    mujoco::mujoco
)

# Copy MJCF assets
add_custom_command(TARGET mujoco_plugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/assets
    $<TARGET_FILE_DIR:mujoco_plugin>/../assets
)
```

---

## plugin.json

```json
{
    "id": "com.cyxwiz.simulation.mujoco",
    "name": "MuJoCo Simulation",
    "version": "1.0.0",
    "api_version": "1.0.0",
    "description": "Physics simulation for robotics and RL training (powered by Google DeepMind MuJoCo)",
    "author": "CyxWiz Lab",
    "license": "Apache-2.0",
    "capabilities": ["ProvidesNodes", "ProvidesPanels", "ProvidesTraining", "RequiresGPU"],
    "permissions": ["UIModify", "Training", "DataRegistry", "GPU", "FileSystem"],
    "platforms": {
        "windows": { "library": "bin/mujoco_plugin.dll" },
        "linux": { "library": "bin/libmujoco_plugin.so" },
        "macos": { "library": "bin/libmujoco_plugin.dylib" }
    }
}
```

---

## Implementation Phases

### Phase 1: Core Physics
- MjEnvManager (load MJCF, step, reset, get obs/reward)
- Plugin skeleton (IPlugin lifecycle)
- Single hardcoded env (InvertedPendulum — simplest)
- Headless training integration via ITrainingHook

### Phase 2: Viewport Rendering
- MjRenderer (FBO → OpenGL texture → ImGui::Image)
- MjViewportPanel (3D viewport with mouse camera control)
- Play/Pause/Reset controls
- Render flags (contacts, forces, wireframe)

### Phase 3: Node Editor Integration
- MuJoCo Env node (INodeProvider)
- Observation Filter node
- Reward Function node
- Code generation (PyTorch Gymnasium)

### Phase 4: Environment Library
- Bundle all 10 standard MJCF environments
- MjEnvRegistry with categories
- Environment Browser panel
- Custom MJCF loading from Asset Browser

### Phase 5: Advanced Features
- Multiple simultaneous environments (vectorized)
- Domain randomization (randomize physics params)
- Sim-to-real config export
- Video recording (save rendered frames)

---

## Sources

- [MuJoCo Documentation](https://mujoco.readthedocs.io/en/stable/)
- [MuJoCo Visualization API](https://mujoco.readthedocs.io/en/stable/programming/visualization.html)
- [MuJoCo C API Functions](https://mujoco.readthedocs.io/en/latest/APIreference/APIfunctions.html)
- [Gymnasium MuJoCo Environments](https://gymnasium.farama.org/environments/mujoco/)
- [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)
- [MuJoCo GitHub](https://github.com/google-deepmind/mujoco)
