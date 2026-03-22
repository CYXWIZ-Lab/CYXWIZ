# MuJoCo Simulation Plugin — User Guide

## Overview

The MuJoCo plugin integrates the [MuJoCo](https://mujoco.org/) physics engine into CyxWiz Engine, providing:

- **Environment Library** — Browse and load 40+ robot environments (builtin + MuJoCo Menagerie)
- **3D Viewport** — Real-time physics rendering with camera controls
- **Node Editor Integration** — Simulink-style control via MuJoCo Plant node
- **RL Training** — Gymnasium-compatible simulation loop (placeholder agent, Python integration planned)

---

## Getting Started

### 1. Open the Environment Library

Go to **View > Plugins > Environment Library** (or find it in the sidebar).

### 2. Load an Environment

- Find **Inverted Pendulum** under "Classic Control"
- Click the **Load** button on its card
- The environment loads into the MuJoCo engine

### 3. Open the Viewport

Go to **View > Plugins > MuJoCo Viewport**.

You should see the inverted pendulum rendered in 3D. Click **Play** to watch it fall under gravity, or **Step** to advance one physics tick at a time.

### 4. Control via Node Editor

- In the **Node Editor**, right-click and add a **MuJoCo Plant** node
- Click **Sync from Environment Library** in the Properties panel (or it syncs automatically)
- The node discovers actuators and sensors from the loaded model and creates dynamic pins
- Connect a **Slider** node to an actuator pin
- Click **Run Sim** in the Node Editor toolbar
- Move the slider to control the pendulum

---

## Environment Library

The Environment Library panel lets you browse, search, download, and load MuJoCo environments.

### Search and Filter

- **Search bar** — Type to filter environments by name or description (case-insensitive)
- **Category tabs** — Click a category to filter: All, Classic Control, Locomotion, Manipulation, Arms, Quadrupeds, Humanoids, etc.

### Environment Cards

Each card shows:

| Field | Description |
|-------|-------------|
| **Name** | Environment name (e.g. "Inverted Pendulum") |
| **Source badge** | `[Builtin]` (gray), `[Downloaded]` (green), or `[Cloud]` (blue) |
| **Description** | Short description of the task |
| **Act / Obs** | Action and observation dimensions |
| **Category** | Environment category |

### Loading Environments

- **Builtin environments** — Click **Load** to load immediately
- **Menagerie environments** (Cloud badge) — Click **Download** first, then **Load** after download completes
- **Currently loaded** — Shows green **Loaded** text

### Import from URL

Expand the "Import from URL" section at the bottom to load custom MJCF models:

1. Paste a URL to an MJCF XML file or GitHub directory
2. Click **Import**
3. The model downloads to `~/.cyxwiz/imported/` and appears in the library under "Imported" category

---

## MuJoCo Viewport

The viewport renders the physics scene in real-time and provides simulation controls.

### Toolbar Buttons

| Button | Action |
|--------|--------|
| **Play** | Start auto-stepping physics (zero action — free fall) |
| **Pause** | Stop auto-stepping |
| **Reset** | Reset environment to initial state |
| **Step** | Advance one physics timestep |
| **Render** | Toggle rendering on/off (checkbox) |
| **Settings** | Expand/collapse visualization settings |

**Status line** (right side): Shows current step count, episode reward, and episode number.

> **Note:** When RL Training or Node Editor graph simulation is active, Play does not step physics — it only controls whether the viewport renders. The active simulation drives physics instead.

### Camera Controls

| Mouse Input | Action |
|-------------|--------|
| **Left drag** | Rotate camera (azimuth/elevation) |
| **Right drag** | Pan camera (move lookat point) |
| **Scroll wheel** | Zoom in/out |

Camera controls work when hovering over the viewport image.

### Visualization Settings

Click **Settings** to expand the options panel:

| Button | Toggles |
|--------|---------|
| **Contacts** | Contact point markers |
| **Forces** | Contact force vectors |
| **Wireframe** | Wireframe rendering mode |
| **Joints** | Joint axis visualization |
| **Actuators** | Actuator visualization |

**Speed slider** — Controls playback speed (0.1x to 10.0x). Affects how many physics steps run per frame when Play is active.

---

## Simulation Modes

Below the toolbar, the viewport shows simulation controls when a sim executor is attached.

### Manual Control

Click **Manual Control** to start a simulation loop that:
- Reads actuator values from the sim executor (set by external sources)
- Steps physics at real-time pace
- Reports sensor outputs

Controls available: **Pause**, **Resume**, **Step**, **Stop**

**Realtime Factor slider** (0.0x to 5.0x):
- `1.0x` = real-time physics
- `0.0x` = run as fast as possible
- `5.0x` = 5x speed

**Metrics shown**: Simulated time, step count, real-time factor achieved.

### RL Training

Click **RL Training** to start a training loop that:
- Generates random actions (current placeholder implementation)
- Steps the environment for up to 1,000,000 timesteps
- Tracks episode rewards and lengths
- Resets when episodes end (terminated or truncated)

Controls available: **Pause**, **Resume**, **Step**, **Stop**

**Metrics shown**: Step count, episode count, current reward, mean reward.

> **Note:** RL Training currently uses random actions — there is no learning happening. This is a test harness. Real RL training will be implemented via Python scripting (Stable-Baselines3 / Gymnasium) in a future update.

### Graph Simulation

When the Node Editor's **Run Sim** is active, the viewport shows:

> "Physics driven by Node Editor graph simulation"

Manual Control and RL Training buttons are hidden. Stop the graph simulation to regain viewport sim controls.

---

## Node Editor Integration

### MuJoCo Plant Node

The **MuJoCo Plant** node is a Simulink-style plant block that bridges the node editor to MuJoCo physics.

**Adding the node:**
1. Right-click in the Node Editor canvas
2. Navigate to **Simulation / Control > MuJoCo Plant**

**Configuring the model:**
- Set the `mjcf_path` parameter in Properties to point to an MJCF file
- Or load a model from the Environment Library — the Plant node auto-syncs via **Sync from Environment Library** button in Properties

**Dynamic pins:**
Once a model is loaded, the node auto-discovers actuators and sensors from the MJCF file and creates:
- **Input pins** — One per actuator (in "bus" mode) or single `u` tensor (in "vector" mode)
- **Output pins** — One per sensor, plus `qpos`, `qvel`, and `sensor` (concatenated)

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mjcf_path` | (empty) | Path to MJCF XML file |
| `timestep` | 0.002 | Physics timestep (seconds) |
| `frame_skip` | 1 | Physics steps per graph tick |
| `interface` | bus | `bus` (per-actuator pins) or `vector` (single tensor) |
| `camera` | 0 | Camera index for image output |

**Running the simulation:**
1. Connect input nodes (Sliders, constants, etc.) to actuator pins
2. Connect output pins to Scope/display nodes
3. Click **Run Sim** in the Node Editor toolbar
4. The graph executor ticks each node — MuJoCo Plant steps physics and outputs sensor values
5. Click **Stop Sim** to end

### Mutual Exclusion

- **Run Sim** is disabled while RL Training is active in the viewport
- **RL Training / Manual Control** buttons are hidden while graph simulation is active
- Starting one automatically stops the other

---

## Node Types Reference

### MuJoCoEnv

| | |
|---|---|
| **Category** | RL / Simulation |
| **Purpose** | Define a Gymnasium-compatible MuJoCo environment for code generation |
| **Output pins** | `env` (Environment) |

**Parameters**: `mjcf_path`, `max_steps`, `frame_skip`, `reward_threshold`

Generates Python code for Gymnasium environment initialization.

### RewardFunction

| | |
|---|---|
| **Category** | RL / Simulation |
| **Purpose** | Define reward shaping for RL training |
| **Input pins** | `qpos`, `qvel`, `ctrl`, `sensor` (all Tensor) |
| **Output pins** | `reward` (Float) |

**Parameters**: `alive_bonus` (1.0), `ctrl_cost_weight` (0.1), `velocity_reward` (true), `height_penalty_threshold` (0.0), `height_penalty_value` (10.0)

**Formula**: `reward = alive_bonus - ctrl_cost * sum(action^2) + velocity_bonus - height_penalty`

### ObservationFilter

| | |
|---|---|
| **Category** | RL / Simulation |
| **Purpose** | Filter and normalize observations |
| **Input pins** | `qpos`, `qvel`, `sensor` (all Tensor) |
| **Output pins** | `obs` (Tensor) |

**Parameters**: `include_qpos` (true), `include_qvel` (true), `include_sensors` (false), `normalize` (true)

### RLAgent

| | |
|---|---|
| **Category** | RL / Simulation |
| **Purpose** | RL agent for code generation (PPO or SAC) |
| **Input pins** | `env` (Environment) |
| **Output pins** | `policy` (Model) |

**Parameters**: `algorithm` (PPO), `learning_rate` (3e-4), `gamma` (0.99), `gae_lambda` (0.95), `clip_range` (0.2), `n_steps` (2048), `batch_size` (64), `n_epochs` (10), `hidden_sizes` (64,64)

Generates Stable-Baselines3 training code.

### MuJoCoPlant

See [Node Editor Integration](#node-editor-integration) above for full details.

---

## Environment Reference

### Builtin Environments

These ship with the plugin and load instantly.

#### Classic Control

| Name | ID | Act | Obs | Max Steps | Description |
|------|----|-----|-----|-----------|-------------|
| Inverted Pendulum | `inverted_pendulum` | 1 | 4 | 1000 | Balance a pole on a sliding cart |
| Cart-Pole Swing-Up | `cartpole` | 1 | 4 | 500 | Swing up and balance a pole |
| Reacher 2D | `reacher` | 2 | 11 | 200 | Move 2-joint arm to target |

#### Locomotion

| Name | ID | Act | Obs | Max Steps | Description |
|------|----|-----|-----|-----------|-------------|
| Hopper | `hopper` | 3 | 11 | 1000 | Single-legged robot hopping |
| Walker 2D | `walker2d` | 6 | 17 | 1000 | Planar bipedal walking |
| Half Cheetah | `half_cheetah` | 6 | 17 | 1000 | 2D cheetah running |

#### Manipulation

| Name | ID | Act | Obs | Max Steps | Description |
|------|----|-----|-----|-----------|-------------|
| Pusher | `pusher` | 7 | 23 | 200 | 3-joint arm pushing object |

### Menagerie Environments (Cloud — Require Download)

These download from MuJoCo Menagerie on first use. Stored in `~/.cyxwiz/menagerie/`.

#### Arms

| Name | ID | Act | Obs |
|------|----|-----|-----|
| Franka Panda | `franka_emika_panda` | 7 | 14 |
| UR5e | `universal_robots_ur5e` | 6 | 12 |
| UR10e | `universal_robots_ur10e` | 6 | 12 |
| KUKA iiwa 14 | `kuka_iiwa_14` | 7 | 14 |
| Kinova Gen3 | `kinova_gen3` | 7 | 14 |
| Sawyer | `sawyer` | 7 | 14 |
| ALOHA (dual-arm) | `aloha` | 14 | 28 |

#### Quadrupeds

| Name | ID | Act | Obs |
|------|----|-----|-----|
| Unitree Go2 | `unitree_go2` | 12 | 36 |
| Unitree Go1 | `unitree_go1` | 12 | 36 |
| Unitree A1 | `unitree_a1` | 12 | 36 |
| ANYmal C | `anymal_c` | 12 | 36 |
| ANYmal B | `anymal_b` | 12 | 36 |
| Boston Dynamics Spot | `spot` | 12 | 36 |

#### Humanoids

| Name | ID | Act | Obs |
|------|----|-----|-----|
| Unitree G1 | `unitree_g1` | 23 | 69 |
| Unitree H1 | `unitree_h1` | 19 | 57 |
| Robotis OP3 | `robotis_op3` | 20 | 60 |

#### Bipeds

| Name | ID | Act | Obs |
|------|----|-----|-----|
| Agility Cassie | `agility_cassie` | 10 | 32 |
| Berkeley Humanoid | `berkeley_humanoid` | 12 | 36 |

#### Hands

| Name | ID | Act | Obs |
|------|----|-----|-----|
| Shadow Hand | `shadow_hand` | 24 | 48 |
| Allegro Hand | `wonik_allegro` | 16 | 32 |
| LEAP Hand | `leap_hand` | 16 | 32 |

#### Grippers

| Name | ID | Act | Obs |
|------|----|-----|-----|
| Robotiq 2F-85 | `robotiq_2f85` | 1 | 2 |
| Robotis RH-P12-RN | `robotis_rh_p12_rn` | 1 | 2 |

#### Drones

| Name | ID | Act | Obs |
|------|----|-----|-----|
| Crazyflie 2 | `bitcraze_crazyflie_2` | 4 | 12 |
| Skydio X2 | `skydio_x2` | 4 | 12 |

#### Mobile Manipulators

| Name | ID | Act | Obs |
|------|----|-----|-----|
| Google Robot | `google_robot` | 8 | 20 |
| Google Barkour vB | `google_barkour_vb` | 12 | 36 |

#### Biomechanical

| Name | ID | Act | Obs |
|------|----|-----|-----|
| MyoHand | `myo_hand` | 39 | 63 |
| MyoLeg | `myo_leg` | 80 | 120 |

---

## Troubleshooting

### Environment doesn't load
- Check the console for MJCF parse errors
- Ensure the MJCF file exists at the expected path
- For menagerie environments, verify the download completed (check `~/.cyxwiz/menagerie/`)

### Viewport shows black or "Rendering disabled"
- Click the **Render** checkbox to ensure rendering is enabled
- Try clicking **Reset** to reset the camera
- If the viewport shows nothing after loading, the renderer may have failed to initialize — check the console for OpenGL errors

### MuJoCo Plant node has no pins
- Ensure a model is loaded in the Environment Library
- Click **Sync from Environment Library** in the Properties panel
- Or set the `mjcf_path` parameter manually

### App crashes during simulation
- Avoid clicking multiple simulation controls rapidly
- If running RL Training, the Play button in the viewport is safe — it does not step physics during training
- Check console for error messages before the crash

### Menagerie download fails
- Verify internet connectivity
- Check that `~/.cyxwiz/menagerie/` directory is writable
- Try the URL import feature as an alternative
