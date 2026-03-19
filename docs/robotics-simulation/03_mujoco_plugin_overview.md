# MuJoCo Simulation Plugin for CyxWiz Engine

## Why This Plugin Exists

CyxWiz Engine trains ML models on **static datasets** — images, tabular data, text. You load data, build a model, train, evaluate. The data never changes in response to the model's predictions.

Reinforcement learning is different. An **agent** takes actions in an **environment**, the environment responds with new observations and rewards, and the agent learns from this interaction loop. There is no fixed dataset — the agent generates its own training data by exploring.

The MuJoCo plugin adds physics simulation to CyxWiz, turning it from a **data-to-model tool** into an **environment-to-agent platform**.

```
Before (Static ML):     Load Dataset → Build Model → Train → Evaluate
After  (RL + MuJoCo):   Pick Environment → Build Agent → Train → Watch it Learn → Export Policy
```

---

## What Is MuJoCo

MuJoCo (Multi-Joint dynamics with Contact) is Google DeepMind's open-source physics engine. It simulates articulated bodies — robots, creatures, vehicles — with accurate contact physics at sub-millisecond speed.

- **Used by**: Google DeepMind, OpenAI, UC Berkeley, Meta AI, Stanford
- **Standard for**: Gymnasium (OpenAI Gym successor), robotics RL benchmarks
- **License**: Apache 2.0 (free, open-source)
- **Speed**: ~1,000,000 physics steps per second on CPU
- **Size**: ~10 MB library

---

## Use Cases

### 1. Robot Manipulation — Train a Robot Arm to Pick Up Objects

**Scenario**: You're building a controller for a 7-DOF robot arm (like the Franka Panda) that needs to reach, grasp, and move objects.

**In CyxWiz**:
1. Open Environment Browser → select "Fetch Reach" or import your custom robot MJCF
2. Node Editor: connect **MuJoCo Env** → **PPO Agent** → feedback loop
3. The 3D viewport shows the robot arm moving in real-time
4. Training metrics show reward improving as the arm learns to reach the target
5. Export the trained policy as a PyTorch model → deploy on physical robot

**What MuJoCo provides**: Accurate joint dynamics, contact forces, gripper physics. The simulated arm behaves like a real one, so policies transfer to hardware.

```
Example environments:
  • Fetch Reach    — move end-effector to target position
  • Fetch Push     — push object to goal location
  • Fetch Slide    — slide puck across table to target
  • Fetch Pick&Place — pick up object and place at goal
```

---

### 2. Locomotion — Teach a Creature to Walk

**Scenario**: You want to train a simulated humanoid, quadruped, or custom creature to walk, run, or balance.

**In CyxWiz**:
1. Environment Browser → select "Ant-v5" (4-legged creature)
2. Node Editor: **MuJoCo Env** → **SAC Agent** (good for continuous control)
3. Watch the ant flop around initially, then gradually learn to walk
4. Reward curve shows improvement over ~500K steps (~5 minutes training)
5. Switch to "Humanoid-v5" for bipedal walking (harder, longer training)

**What MuJoCo provides**: Multi-body dynamics, ground contact, joint torque limits. Creatures learn physically plausible gaits, not animation tricks.

```
Example environments:
  • Ant-v5          — 4-legged walker (8 actuators)
  • HalfCheetah-v5  — 2D runner (6 actuators)
  • Humanoid-v5     — bipedal walker (17 actuators)
  • Hopper-v5       — 1-legged hopper (3 actuators)
  • Walker2d-v5     — 2D bipedal (6 actuators)
  • Swimmer-v5      — 2D swimmer (2 actuators)
```

---

### 3. Control Systems — Balance and Stabilize

**Scenario**: Classic control problems — balance an inverted pendulum, stabilize a double pendulum, reach a target with minimal energy.

**In CyxWiz**:
1. Environment Browser → select "InvertedPendulum-v5"
2. Node Editor: **MuJoCo Env** → **DQN** or **PPO Agent**
3. Agent learns to balance the pole in under 1 minute of training
4. Upgrade to "InvertedDoublePendulum-v5" for a harder challenge
5. Great for learning RL fundamentals before tackling complex tasks

**What MuJoCo provides**: Precise torque application, gravity, friction. Simple enough to train in seconds, physics-accurate enough to be meaningful.

```
Example environments:
  • InvertedPendulum-v5        — balance a pole (1 actuator)
  • InvertedDoublePendulum-v5  — balance a double pole (1 actuator)
  • Reacher-v5                 — move arm to target (2 actuators)
```

---

### 4. Vehicle Dynamics — Simplified Autonomous Navigation

**Scenario**: Train a vehicle controller for steering, braking, and acceleration using simplified physics.

**In CyxWiz**:
1. Import a custom vehicle MJCF model (car, drone, or boat)
2. Define observation space (position, velocity, heading, obstacles)
3. Define reward (distance to goal, staying on track, avoiding collisions)
4. Train controller using PPO or SAC
5. Export for deployment or further testing in CARLA (full driving sim)

**What MuJoCo provides**: Rigid body dynamics, wheel contact, terrain interaction. Not photorealistic driving (that's CARLA), but accurate physics for control learning.

---

### 5. Education and Research — RL Benchmarking

**Scenario**: A student or researcher wants to learn RL, compare algorithms, or reproduce paper results.

**In CyxWiz**:
1. Pick any standard Gymnasium environment from the browser
2. Build different agent architectures in the node editor (PPO vs SAC vs TD3)
3. Train all three side-by-side, compare reward curves
4. Watch the 3D viewport to understand what the agent is actually doing
5. All environments match Gymnasium standards — results are directly comparable to published benchmarks

**What MuJoCo provides**: The same physics used in papers by DeepMind, OpenAI, and top RL labs. Your results are reproducible and comparable.

```
Standard benchmarks available:
  • HalfCheetah  — baseline for continuous control papers
  • Ant          — multi-joint locomotion benchmark
  • Humanoid     — high-dimensional control challenge
  • Reacher      — simple manipulation baseline
```

---

### 6. Sim-to-Real Transfer — Train in Simulation, Deploy on Hardware

**Scenario**: You have a physical robot and want to train its controller in simulation before deploying.

**In CyxWiz**:
1. Model your robot in MJCF XML (define joints, actuators, sensors)
2. Import into CyxWiz → configure observation/action spaces
3. Enable domain randomization (randomize friction, mass, damping)
4. Train with randomized physics so the policy is robust
5. Export trained model → deploy on real robot via ROS or direct control

**What MuJoCo provides**: Accurate enough physics that policies trained in MuJoCo transfer to real robots. This is proven — Boston Dynamics, Agility Robotics, and DeepMind all use MuJoCo for sim-to-real.

---

## The MATLAB Comparison

CyxWiz with MuJoCo mirrors MATLAB's simulation ecosystem:

| MATLAB | CyxWiz |
|--------|--------|
| Simulink (visual blocks) | Node Editor (visual pipeline) |
| Simscape (physics engine) | MuJoCo (physics engine) |
| RL Toolbox (PPO, SAC, DQN) | Training System (PPO, SAC, DQN) |
| Mechanics Explorer (3D viewer) | MuJoCo Viewport (3D viewer) |
| App Designer (custom tools) | Plugin System (custom plugins) |
| MATLAB Drive (cloud compute) | CyxCloud (distributed training) |

The key difference: MATLAB costs $10,000+/year for the full simulation stack. CyxWiz + MuJoCo is entirely open-source and free.

---

## What the User Sees

### Node Editor
```
┌─────────────────┐      ┌──────────────┐      ┌─────────────────┐
│  MuJoCo Env     │      │  PPO Agent   │      │  Replay Buffer  │
│  [Ant-v5]       │─obs──│  (256, 256)  │──────│  (size: 100K)   │
│                 │◀─act─│              │      │                 │
│  reward ────────│──────│──reward      │      │                 │
│  done ──────────│──────│──done        │      │                 │
└─────────────────┘      └──────────────┘      └─────────────────┘
```

### 3D Viewport
```
┌─────────────────────────────────────┐
│ MuJoCo Viewport           [─][□][×] │
├─────────────────────────────────────┤
│ [▶ Play] [⏸] [⟲ Reset] [📷 Snap]  │
│ Ant-v5 | Step: 12,345 | R: 234.5   │
├─────────────────────────────────────┤
│                                     │
│       (3D rendered ant robot        │
│        walking on ground plane)     │
│                                     │
├─────────────────────────────────────┤
│ LMB: Rotate  RMB: Pan  Scroll: Zoom│
└─────────────────────────────────────┘
```

### Training Dashboard
```
┌─────────────────────────────────────┐
│ Episode Reward       [last 100 eps] │
│  300 ┤                          ╱   │
│  200 ┤                      ╱╱╱     │
│  100 ┤                ╱╱╱╱╱         │
│    0 ┤ ╱╱╱╱╱╱╱╱╱╱╱╱╱               │
│      └──────────────────────────────│
│       0    100K   200K   300K  Steps│
│                                     │
│ Mean Reward: 234.5  (solved: 500.0) │
│ Episode Length: 987 steps           │
│ Episodes: 1,234                     │
└─────────────────────────────────────┘
```

---

## Summary

| Without MuJoCo | With MuJoCo |
|----------------|-------------|
| Train on static datasets only | Train on interactive simulated environments |
| Image classification, regression | Robot control, locomotion, manipulation |
| Data is fixed | Agent generates its own data through exploration |
| No physics | Accurate multi-body contact physics |
| No 3D visualization of training | Live 3D viewport of agent learning |
| Compare to: scikit-learn, PyTorch | Compare to: MATLAB Simscape + RL Toolbox |

MuJoCo turns CyxWiz from a tool that trains models on data into a platform that trains agents in worlds.
