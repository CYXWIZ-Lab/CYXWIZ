# MuJoCo in CyxWiz Engine — UX Walkthrough

How the MuJoCo integration looks and feels in practice, from the user's perspective.

---

## Screen Layout

When working with MuJoCo, the engine shows 4 panels:

```
┌──────────────────────────────────────────────────────────────────────┐
│ File  Edit  View  Tools  Simulation  Help                           │
├──────────────┬───────────────────────────────┬───────────────────────┤
│              │                               │                       │
│  Environment │       NODE EDITOR             │   MuJoCo Viewport     │
│  Library     │                               │                       │
│              │  ┌──────────┐  ┌──────────┐   │   ┌─────────────────┐ │
│  🔍 Search   │  │ Constant │  │          │   │   │                 │ │
│              │  │  value:0 ├──► MuJoCo   │   │   │   3D Robot      │ │
│  ▼ Arms      │  └──────────┘  │ Plant    │   │   │   Rendering     │ │
│    Panda     │  ┌──────────┐  │          ├───│───│   (live)        │ │
│    UR5e      │  │ Constant │  │scene.xml │   │   │                 │ │
│    UR10e     │  │ value:π  ├──►          │   │   │                 │ │
│  ▼ Quadruped │  └──────────┘  └────┬─────┘   │   └─────────────────┘ │
│    Go2       │                     │         │   ▶ Play ⏸ ⏹ ⏭ Step  │
│    ANYmal    │              ┌──────┴──────┐  │   Time: 2.34s  60 FPS │
│    Cassie    │              │   Scope     │  │                       │
│  ▼ Humanoids │              │  (sensors)  │  │  ┌───────────────────┐│
│    G1        │              └─────────────┘  │  │  Properties       ││
│  ▼ Drones    │                               │  │  Model: UR5e      ││
│    Crazyflie │                               │  │  Actuators: 6     ││
│              │                               │  │  Sensors: 0       ││
│  [Download]  │                               │  │  Sample: 0.002s   ││
│  [Bundled]   │                               │  │  Interface: Bus   ││
└──────────────┴───────────────────────────────┴──┴───────────────────┘
```

---

## Workflow 1: Manual Robot Control

### Step 1: Browse & Load Model

User opens **Environment Library** panel (left sidebar). Models organized by category:

```
Environment Library
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 Search models...

▼ Arms
  ┌─────────────────────┐
  │ 🦾 Franka Panda     │
  │ 7-DOF arm           │
  │ [Bundled] [Load]    │
  ├─────────────────────┤
  │ 🦾 UR5e             │
  │ 6-DOF arm           │
  │ [Bundled] [Load]    │
  ├─────────────────────┤
  │ 🦾 UR10e            │
  │ 6-DOF arm           │
  │ [☁ Download]        │
  └─────────────────────┘

▼ Quadrupeds
  ┌─────────────────────┐
  │ 🐕 Unitree Go2      │
  │ 12 actuators        │
  │ [Bundled] [Load]    │
  └─────────────────────┘

▼ Humanoids
  ┌─────────────────────┐
  │ 🚶 Unitree G1       │
  │ 23 actuators        │
  │ [Bundled] [Load]    │
  └─────────────────────┘
```

Clicking **[Load]** on UR5e:
- Loads `scene.xml` via MuJoCo
- 3D viewport shows the robot arm on a table
- Console: `[MuJoCo] Loaded UR5e — 6 actuators, 0 sensors, nq=6, nv=6`

### Step 2: Add MuJoCo Plant Node

User right-clicks node editor → **RL / Simulation** → **MuJoCo Plant**

The node appears with the loaded model's info:

```
┌─────────────────────────────────┐
│         MuJoCo Plant            │
│         UR5e (scene.xml)        │
│─────────────────────────────────│
│                                 │
│  ► shoulder_pan    sensor ►     │
│  ► shoulder_lift   qpos   ►     │
│  ► elbow           qvel   ►     │
│  ► wrist_1         rgb    ►     │
│  ► wrist_2         depth  ►     │
│  ► wrist_3                      │
│                                 │
└─────────────────────────────────┘
```

Left side: **input pins** — one per actuator (auto-detected from MJCF)
Right side: **output pins** — sensor data, joint positions/velocities, camera

### Step 3: Wire Control Signals

User adds **Constant** nodes and connects them:

```
┌──────────┐
│ Constant │
│ value: 0 ├───────────────────► shoulder_pan
└──────────┘
┌──────────┐                    ┌─────────────────────────────────┐
│ Constant │                    │         MuJoCo Plant            │
│ value: π ├───────────────────►│         UR5e                    │
└──────────┘  shoulder_lift     │                                 │
┌──────────┐                    │  ► shoulder_pan    sensor ►──┐  │
│ Slider   │                    │  ► shoulder_lift   qpos   ►  │  │
│ 0 — 2π   ├───────────────────►│  ► elbow           qvel   ►  │  │
│ ═══●════ │  elbow             │  ► wrist_1         rgb    ►  │  │
└──────────┘                    │  ► wrist_2         depth  ►  │  │
┌──────────┐                    │  ► wrist_3                   │  │
│ Sine Wave│                    └─────────────────────────────────┘
│ A:1 f:2  ├───────────────────► wrist_1           │
└──────────┘                                       │
                                            ┌──────┴──────┐
                                            │    Scope    │
                                            │  ~~~~~~~~~~  │
                                            │  Joint Pos   │
                                            └─────────────┘
```

### Step 4: Run Simulation

User clicks **▶ Play** in the viewport toolbar:

```
MuJoCo Viewport
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─────────────────────────────────┐
│                                 │
│      UR5e Robot Arm             │
│      (3D rendered, moving)      │
│                                 │
│      Arm responds to wired      │
│      actuator values in         │
│      real-time                  │
│                                 │
│      Slider changes → arm       │
│      elbow moves instantly      │
│                                 │
│      Sine wave → wrist_1        │
│      oscillates back & forth    │
│                                 │
└─────────────────────────────────┘
 ▶ Play  ⏸ Pause  ⏹ Stop  ⏭ Step
 Time: 3.42s | 500 Hz | FPS: 60
 Mode: Manual Control
```

The **Scope** node shows a real-time plot of joint positions:

```
Scope — Joint Positions
━━━━━━━━━━━━━━━━━━━━━━━━
  qpos
  1.5 │    ╱╲    ╱╲    ╱╲
      │   ╱  ╲  ╱  ╲  ╱  ╲
  0.0 │──╱────╲╱────╲╱────╲──
      │
 -1.5 │
      └───────────────────────
       0    1    2    3    4 s
```

### Step 5: Adjust Live

User drags the **Slider** node's handle → elbow angle changes instantly in viewport.
User changes **Sine Wave** frequency in Properties → oscillation speed updates live.

---

## Workflow 2: RL Training

### Step 1: Same model loading (Environment Library)

### Step 2: Build RL Pipeline in Node Editor

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│ MuJoCo Plant │     │   Reward     │     │ Observation  │     │ RL Agent │
│ Ant-v5       │     │   Function   │     │   Filter     │     │          │
│              ├────►│              ├────►│              ├────►│  PPO     │
│  scene.xml   │ env │ alive: 1.0   │ env │ qpos: ✓     │ obs │  lr:3e-4 │
│              │     │ ctrl: 0.1    │     │ qvel: ✓     │     │  γ:0.99  │
│  sensor ►────│     │ velocity: ✓  │     │ norm: ✓     │     │          │
│  qpos   ►    │     │              │     │              │     │ policy ► │
│  qvel   ►    │     │              │     │              │     │          │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────┘
```

### Step 3: Click "Train" in Node Editor Toolbar

The system:
1. Compiles the node graph into RL configuration
2. Spawns background training thread
3. MuJoCo viewport shows the ant learning to walk

```
MuJoCo Viewport                      Training Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━             ━━━━━━━━━━━━━━━━━━━━
┌──────────────────────┐             Episode: 847
│                      │             Steps: 1,234,567
│   Ant learning to    │
│   walk (live 3D)     │             Mean Reward
│                      │             250│      ╱────
│   Falls, gets up,    │                │    ╱╱
│   gradually walks    │                │  ╱╱
│   further            │             0  │╱╱
│                      │                └──────────
│                      │                0  200  400  600
└──────────────────────┘
 ⏸ Pause Training  ⏹ Stop          Episode Length
 Episode: 847 | Reward: 234.5       200│    ╱──────
 Mode: RL Training (PPO)               │  ╱╱
                                     0  │╱╱
                                        └──────────
```

### Step 4: Training Controls

- **⏸ Pause** — Freezes training, viewport shows last state
- **⏹ Stop** — Ends training, saves best policy
- **Speed slider** — Render every Nth episode (faster training)
- **Export** — Save trained policy as ONNX/PyTorch

---

## Workflow 3: Code Generation (existing feature, enhanced)

After building either pipeline, user clicks **Generate Code** → gets runnable Python:

```python
# Generated by CyxWiz Engine — MuJoCo RL Training Pipeline
import mujoco
import gymnasium as gym
from stable_baselines3 import PPO

# Environment
env = gym.make('Ant-v5', max_episode_steps=1000)

# Reward shaping
from gymnasium.wrappers import TransformReward
def shaped_reward(obs, action, next_obs, info):
    reward = 1.0  # alive bonus
    reward -= 0.1 * (action ** 2).sum()  # control cost
    reward += next_obs[0]  # forward velocity
    return reward

# Agent
model = PPO(
    'MlpPolicy', env,
    learning_rate=3e-4, gamma=0.99,
    gae_lambda=0.95, clip_range=0.2,
    n_steps=2048, batch_size=64,
    n_epochs=10,
    policy_kwargs=dict(net_arch=[64, 64]),
    verbose=1,
)

# Train
model.learn(total_timesteps=1_000_000)
model.save("ant_policy")
```

---

## Properties Panel — MuJoCo Plant Details

When MuJoCo Plant node is selected:

```
Properties — MuJoCo Plant
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▼ Model Info
  Name:          UR5e
  File:          .../universal_robots_ur5e/scene.xml
  Actuators:     6
  Sensors:       0
  Cameras:       1
  DOF (nq/nv):   6 / 6
  Sample Time:   0.002s

▼ Control Inputs
  Interface:     [Bus ▼]     ← Bus = per-actuator pins
                              Vector = single array pin

  Actuator Map:
  ┌────────────────┬──────────┬───────────┐
  │ Name           │ Joint    │ Range     │
  ├────────────────┼──────────┼───────────┤
  │ shoulder_pan   │ joint1   │ -2π..2π   │
  │ shoulder_lift  │ joint2   │ -2π..2π   │
  │ elbow          │ joint3   │ -π..π     │
  │ wrist_1        │ joint4   │ -2π..2π   │
  │ wrist_2        │ joint5   │ -2π..2π   │
  │ wrist_3        │ joint6   │ -2π..2π   │
  └────────────────┴──────────┴───────────┘

▼ Rendering
  Camera:        [default ▼]
  Resolution:    640 × 480
  Show contacts: ☐
  Show forces:   ☐
  Wireframe:     ☐

▼ Simulation
  Timestep:      0.002
  Frame skip:    1
  Max steps:     1000
```

---

## Menu Integration

```
Simulation (new top-level menu)
├── ▶ Run Simulation        Ctrl+F5
├── ⏸ Pause                 Ctrl+F6
├── ⏹ Stop                  Ctrl+F7
├── ⏭ Step Once             F10
├── ─────────────────
├── Mode
│   ├── ● Manual Control
│   └── ○ RL Training
├── ─────────────────
├── Open Environment Library
├── Load MJCF Model...      Ctrl+Shift+M
└── Export Policy...
```

---

## End-to-End User Journey

1. **Open CyxWiz Engine**
2. **View → Environment Library** — browse robots
3. **Click [Load] on "Unitree Go2"** — quadruped appears in viewport
4. **Right-click node editor → RL/Simulation → MuJoCo Plant** — node created with 12 actuator pins
5. **Add Reward Function, Observation Filter, RL Agent nodes** — wire them together
6. **Click ▶ Train** — background RL training starts
7. **Watch the Go2 quadruped learn to walk** in the 3D viewport
8. **Pause to inspect** — check reward curve, adjust hyperparameters
9. **Resume training** — agent continues learning
10. **Training converges** — export policy as ONNX
11. **Switch to Manual Control mode** — wire Slider to each leg
12. **Drag sliders** — control each joint manually, test edge cases
13. **Generate Python code** — get reproducible training script
