# 3D Simulation Environments for Robotics & AI Agents

> Research compiled: 2026-01-20

## Overview

This document surveys open-source 3D simulation environments for training AI agents in robotics, autonomous vehicles, and real-world applications.

---

## Top Simulators

### 1. CARLA - Autonomous Driving

| Attribute | Value |
|-----------|-------|
| Purpose | Autonomous driving research |
| Engine | Unreal Engine 4 |
| License | MIT |
| Language | C++, Python |
| ROS Support | Yes (ROS bridge) |
| URL | https://carla.org |

**Features:**
- LIDAR, cameras, GPS, depth sensors
- OpenDRIVE map standard
- Dynamic weather, day/night cycles
- Traffic simulation
- Scenario runner for benchmarks

**Hardware Requirements:**
- GPU: 6GB+ VRAM (8GB+ recommended)
- RAM: 16GB+
- Storage: 20GB+

---

### 2. MuJoCo - Physics Simulation

| Attribute | Value |
|-----------|-------|
| Purpose | Fast, accurate physics for articulated bodies |
| Owner | Google DeepMind |
| License | Apache 2.0 |
| Language | C, Python |
| Citations | 3800+ |
| URL | https://mujoco.org |

**Features:**
- Best-in-class contact physics
- Sub-millisecond simulation steps
- XML model format (MJCF)
- Unity plugin available
- GPU-accelerated via MuJoCo-XLA

**Best For:**
- Robot manipulation
- Locomotion (walking, running)
- Reinforcement learning research

---

### 3. Gazebo (Ignition)

| Attribute | Value |
|-----------|-------|
| Purpose | General robotics simulation |
| License | Apache 2.0 |
| Language | C++, Python |
| ROS Support | Native |
| URL | https://gazebosim.org |

**Features:**
- Native ROS/ROS2 integration
- SDF model format
- Multi-robot simulation
- Sensor plugins (camera, LIDAR, IMU)
- Physics engine options (ODE, Bullet, DART)

**Best For:**
- ROS-based robot development
- Multi-robot coordination
- Sensor simulation

---

### 4. NVIDIA Isaac Sim / Isaac Lab

| Attribute | Value |
|-----------|-------|
| Purpose | GPU-accelerated robot learning |
| License | Free (proprietary) |
| Platform | NVIDIA Omniverse |
| URL | https://developer.nvidia.com/isaac-sim |

**Features:**
- Parallel simulation (1000s of environments)
- Photorealistic rendering (RTX)
- Domain randomization
- ROS2 bridge
- USD format support

**Hardware Requirements:**
- GPU: RTX 2070+ (RTX 3080+ recommended)
- RAM: 32GB+
- VRAM: 8GB+ (12GB+ recommended)

**Best For:**
- Large-scale RL training
- Sim-to-real transfer
- Industrial robotics

---

### 5. PyBullet

| Attribute | Value |
|-----------|-------|
| Purpose | Prototyping and learning |
| License | Zlib |
| Language | Python |
| URL | https://pybullet.org |

**Features:**
- Pure Python API
- OpenAI Gym compatible
- Built-in robot models (Kuka, Panda, etc.)
- Inverse kinematics
- VR support

**Best For:**
- Learning RL
- Quick prototyping
- Education

---

### 6. Microsoft AirSim

| Attribute | Value |
|-----------|-------|
| Purpose | Drones and autonomous vehicles |
| Engine | Unreal Engine / Unity |
| License | MIT |
| Status | Maintenance mode |
| URL | https://github.com/microsoft/AirSim |

**Features:**
- High-fidelity visuals
- Multi-vehicle support
- PX4/ArduPilot integration
- Weather simulation

---

## Comparison Matrix

| Simulator | Physics Quality | Visual Quality | Speed | Ease of Use | ROS Support |
|-----------|----------------|----------------|-------|-------------|-------------|
| CARLA | Medium | High | Slow | Medium | Yes |
| MuJoCo | Excellent | Low | Fast | Medium | No |
| Gazebo | Good | Medium | Medium | Easy | Native |
| Isaac Sim | Excellent | Excellent | Fast* | Hard | Yes |
| PyBullet | Good | Low | Fast | Easy | No |
| AirSim | Medium | High | Slow | Medium | Yes |

*Fast with GPU parallelization

---

## Use Case Recommendations

| Application | Primary | Secondary |
|-------------|---------|-----------|
| Self-driving cars | CARLA | AirSim |
| Robot manipulation | MuJoCo | Isaac Lab |
| Walking robots | MuJoCo | PyBullet |
| Drones/UAVs | AirSim | Gazebo |
| Multi-robot systems | Gazebo | Isaac Sim |
| RL research | MuJoCo | PyBullet |
| Industrial robots | Isaac Sim | Gazebo |
| Education | PyBullet | Gazebo |

---

## Sources

- https://carla.org/
- https://github.com/microsoft/AirSim
- https://news.mit.edu/2022/researchers-release-open-source-photorealistic-simulator-autonomous-driving-0621
- https://techcrunch.com/2026/01/05/nvidia-launches-alpamayo-open-ai-models-that-allow-autonomous-vehicles-to-think-like-a-human/
- https://roboticsknowledgebase.com/wiki/robotics-project-guide/choose-a-sim/
- https://github.com/knmcguire/best-of-robot-simulators
- https://cybernachos.github.io/robotics-overview/simulation-platforms-guide/
