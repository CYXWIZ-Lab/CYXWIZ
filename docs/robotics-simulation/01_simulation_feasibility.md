# CyxWiz Simulation Plugin Feasibility Analysis

> Analysis date: 2026-01-31

## Engine Constraints

CyxWiz Engine is an ML pipeline builder + trainer, not a game engine. Key constraints:

| Aspect | Current State |
|--------|--------------|
| Graphics | OpenGL 3.3 core + compatibility profile |
| UI | Dear ImGui (no 3D scene editor) |
| Compute | ArrayFire (tensor operations, not mesh rendering) |
| Plugin SDK | DLL loading, INodeProvider, IPanelProvider, ITrainingHook, IDataProvider |
| Python | Embedded interpreter (pybind11) + GymConnector |
| GPU | GTX 1050 Ti (4GB VRAM) as development target |

## MATLAB Analogy

MATLAB does not embed Unreal Engine. It provides **toolboxes** that:
- Connect to external simulators (Gazebo, CARLA via ROS Toolbox)
- Bring sensor data/controls back into the MATLAB workspace
- Provide lightweight built-in visualizers for physics (Simscape)
- Let users design control systems and train RL agents against sim environments

CyxWiz should follow the same model: **lightweight embedded sims + connectors to heavy external sims**.

---

## Feasibility Per Simulator

### 1. MuJoCo — FEASIBLE (Embedded)

| Criterion | Assessment |
|-----------|-----------|
| License | Apache 2.0 |
| Language | C library with C++ headers |
| Size | ~10 MB shared library |
| OpenGL | Uses OpenGL 1.5 compatibility profile (compatible with our 3.3 context) |
| Offscreen rendering | Native `mjFB_OFFSCREEN` + `mjr_readPixels()` |
| GPU requirement | Minimal — CPU physics, OpenGL for rendering only |
| CMake integration | `find_package(mujoco)` or FetchContent |
| Gym compatibility | Gymnasium has 11 MuJoCo envs + Gymnasium-Robotics (Fetch, Adroit, Kitchen) |

**Why it works:**
- Pure C library links directly into a plugin DLL
- Sub-millisecond physics steps = real-time training
- Offscreen render to FBO → `glTexImage2D` → `ImGui::Image()` for viewport
- Shares our GLFW OpenGL context (MuJoCo needs compatibility profile, we have it)
- Covers robotics, RL, locomotion, manipulation in one package
- Can also use headless (no rendering) for fast training

**Integration approach:** Direct C++ embedding as a CyxWiz plugin.

---

### 2. PyBullet — FEASIBLE (Python Bridge)

| Criterion | Assessment |
|-----------|-----------|
| License | Zlib |
| Language | Python (C++ Bullet underneath) |
| Gym compatibility | Native OpenAI Gym / Gymnasium |
| Integration | Via existing GymConnector + ScriptingEngine |

**Why it works:**
- Already have GymConnector (Python bridge to Gym envs)
- `import pybullet` in embedded Python interpreter
- Built-in robot models (Kuka, Panda, UR5)
- Good for quick prototyping and education

**Integration approach:** Python plugin using existing ScriptingEngine. No C++ DLL needed — just a Gym env node that wraps PyBullet envs.

---

### 3. CARLA — PARTIALLY FEASIBLE (External Connector)

| Criterion | Assessment |
|-----------|-----------|
| License | MIT |
| Language | C++/Python (Unreal Engine 4 server) |
| Size | 20+ GB install |
| GPU requirement | 6GB+ VRAM minimum |
| Architecture | Client-server (Python client connects to UE4 server) |

**Why it works (as connector):**
- Python client library connects to external CARLA server
- Stream camera/LIDAR/GPS sensor data into CyxWiz panels
- Send vehicle controls from trained RL agent
- Use as data source for training (replay buffers, imitation learning)

**Why NOT embedded:**
- Requires Unreal Engine 4 (cannot embed in ImGui app)
- 20+ GB footprint
- Heavy GPU requirements compete with training workload

**Integration approach:** Python connector plugin. User runs CARLA server externally, CyxWiz provides dashboard/controller panels and training pipeline.

---

### 4. Gazebo — PARTIALLY FEASIBLE (External Connector)

| Criterion | Assessment |
|-----------|-----------|
| License | Apache 2.0 |
| Language | C++/Python |
| Architecture | Standalone process with plugin system |
| ROS dependency | Native ROS/ROS2 integration |

**Why it works (as connector):**
- gRPC or ROS2 bridge to external Gazebo instance
- Multi-robot simulation
- Sensor simulation (camera, LIDAR, IMU)

**Why NOT embedded:**
- Large dependency tree (SDF, OGRE3D, Bullet/ODE/DART)
- ROS ecosystem dependency
- Designed as standalone application, not a library

**Integration approach:** Future connector plugin via ROS2 bridge or gRPC transport layer. Lower priority than MuJoCo.

---

### 5. NVIDIA Isaac Sim — NOT FEASIBLE

| Criterion | Assessment |
|-----------|-----------|
| License | Proprietary (free to use) |
| Platform | NVIDIA Omniverse |
| GPU requirement | RTX 2070+ minimum, RTX 3080+ recommended |
| Architecture | Omniverse platform application |

**Why NOT:**
- Requires NVIDIA Omniverse runtime (cannot embed)
- RTX GPU mandatory (our dev target is GTX 1050 Ti)
- Proprietary — cannot redistribute
- 32GB+ RAM requirement

**Verdict:** Skip entirely. Users with Isaac Sim don't need CyxWiz for simulation.

---

### 6. Microsoft AirSim — NOT FEASIBLE

| Criterion | Assessment |
|-----------|-----------|
| License | MIT |
| Status | **Maintenance mode / deprecated** |
| Architecture | UE4/Unity plugin |

**Why NOT:**
- Deprecated by Microsoft
- Requires Unreal Engine or Unity
- No active development

**Verdict:** Skip. Use CARLA for autonomous driving instead.

---

## Priority Roadmap

| Priority | Plugin | Type | Effort |
|----------|--------|------|--------|
| **P0** | MuJoCo Simulation | Embedded (C++ DLL) | Major feature |
| **P1** | Gym/PyBullet Environments | Python bridge (extend GymConnector) | Medium |
| **P2** | CARLA Connector | Python connector | Medium |
| **P3** | Gazebo Connector | ROS2/gRPC bridge | Future |

---

## Sources

- [MuJoCo Visualization API](https://mujoco.readthedocs.io/en/stable/programming/visualization.html)
- [MuJoCo Programming Guide](https://mujoco.readthedocs.io/en/stable/programming/index.html)
- [Gymnasium MuJoCo Environments](https://gymnasium.farama.org/environments/mujoco/)
- [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)
- [MuJoCo Playground](https://playground.mujoco.org/assets/playground_technical_report.pdf)
- [MuJoCo CMake Integration](https://github.com/google-deepmind/mujoco/issues/401)
- [LocoMuJoCo Benchmark](https://github.com/robfiras/loco-mujoco)
