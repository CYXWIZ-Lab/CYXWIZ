# Train, play and export a policy: CartPole, InvertedPendulum, HalfCheetah

A step-by-step guide to the three reinforcement-learning examples that ship
with the Engine:

- **CartPole**: the Engine's own environment, no plugin needed.
- **InvertedPendulum** and **HalfCheetah**: MuJoCo physics, from the MuJoCo
  Simulation plugin.

It covers training each one from Studio, watching the trained policy,
saving it, and what it takes to use a policy outside the Engine (export, and
real robots). For every setting and panel, see
[Training a policy (PPO)](rl_training.md); for the environments themselves,
[Environments for reinforcement learning](rl_environments.md).

## Before you start

- **A device.** Training runs on the device chosen in **Preferences >
  Devices**. On an NVIDIA GPU with the **CUDA GPU pack**, PPO runs its small
  networks as fused GPU kernels and trains several times faster than on the
  CPU (see [What runs where on a GPU](gpu_providers.md)). On the CPU every
  example still trains; it just takes longer.
- **The MuJoCo Simulation plugin** (InvertedPendulum and HalfCheetah only).
  Open **Apps > Plugin Manager**:
  - if the plugin is already in the Engine's `plugins` folder, click
    **Refresh**;
  - otherwise click **Install...** and choose the plugin's folder (the one
    holding `plugin.json`).

  The Plugin Manager then lists *MuJoCo Simulation* as initialised, and the
  Console says `Registered 2 environments for 'com.cyxwiz.simulation.mujoco'`.
  A plugin loaded this way stays loaded for the session.
- **The examples** are in `examples/cyxgraph/rl/`. Open one with **File >
  Open** (or start the Engine with the file as its argument).

Times below were measured on an NVIDIA GeForce GTX 1050 Ti with the CUDA GPU
pack.

## 1. CartPole (built in)

Keep a pole upright on a moving cart by pushing it left or right. An episode
ends when the pole falls or after 500 steps, so the best score is 500.

1. Open `ppo_cartpole.cyxgraph`. The graph is the PPO (MLP) pattern: an
   **Environment** node, a policy network and a value network (two Dense
   layers of 64 with Tanh each), a **Policy Head**, a **Value Head** and the
   **PPO** node with settings tuned for CartPole (8 environment copies).
2. **Train > Start Training** (F5). The **RL Training Dashboard** opens:
   the episode reward climbs along the Stable-Baselines3 reference curve
   drawn under it.
3. The run stops by itself when the last 100 episodes average 475.
   Measured: 475.9 after 74,496 steps, about 8 seconds on the GPU.
4. **Play policy** on the dashboard. The **Policy Playback** panel draws the
   cart and pole and plays the trained policy; each episode now lasts the
   full 500 steps.

## 2. InvertedPendulum (MuJoCo)

The same task in 3D physics: a hinged pole on a cart pushed by a force.
Episodes last up to 1,000 steps (best score 1,000).

1. Load the MuJoCo plugin (see *Before you start*).
2. Open `ppo_inverted_pendulum.cyxgraph` (4 environment copies, 64-64 Tanh,
   tuned settings).
3. **Train** (F5). The run stops when the last 100 episodes average 950.
   Measured: 965.4 after about 120,000 steps, about 30 seconds on the GPU.
4. **Play policy**. The **MuJoCo Viewport** opens by itself and shows the
   pendulum balancing. While it plays, the Viewport's own Run / Step / Reset
   are off; **Stop** in Policy Playback gives the Viewport back its scene.

## 3. HalfCheetah (MuJoCo)

A two-dimensional cheetah with six motorised joints learns to run forward.
There is no "solved" score: the reward is the forward speed minus a small
cost for the effort, summed over 1,000 steps.

1. Load the MuJoCo plugin.
2. Open `ppo_half_cheetah.cyxgraph`: one copy, two layers of 256 with ReLU,
   observation and reward normalisation, and the RL Zoo settings for
   HalfCheetah. It runs all 1,000,000 steps (no target stop).
3. **Train** (F5). Measured: about 8-13 minutes on the GPU (over an hour on
   the CPU), ending at a mean return of about 3,800-5,100 depending on the
   seed (4,675.9 with seed 1). Stable-Baselines3 with the same settings ends
   at 2,970-4,660.
4. **Play policy**. The MuJoCo Viewport shows the cheetah running, seen
   through the model's **track** camera, which runs beside it; the top left
   of the scene shows the episode, step, return and action. **Record >
   Record video...** saves a GIF of the run in the project's exports folder
   ([The MuJoCo Viewport](mujoco_viewport.md)).

## 4. Training for rough ground, pushes and obstacles (MuJoCo)

A policy trained on Gymnasium's flat floor has never met a bump, a step or a
push: the Viewport's **Stress** panel shows it stalling at the first step
([The MuJoCo Viewport](mujoco_viewport.md)). The Environment node's
**Robustness (training)** section trains it for them. Everything there is off
by default, so the plain task stays Gymnasium's.

| Setting | What it does |
| --- | --- |
| **Ground** | Chips: Flat, Rough, Slopes, Steps, Obstacles. Each episode of each copy picks one of the chosen grounds; none chosen keeps the model's own floor |
| **Difficulty from / to** | The range of the grounds' difficulty (0 gentle, 1 hard) |
| **Curriculum mark** | A mean return. Training starts at the easiest ground; each time the mean return of recent episodes passes the mark, the hardest ground drawn rises by 0.1, up to **to**. 0: the whole range from the start |
| **Push every / up to** | A push in a random direction every so many simulated seconds (varied by half), up to the speed given |
| **Friction / Mass / Motor strength ±** | Each episode scales them by a factor drawn within the percentage |
| **Look ahead** | The ground's height 0.2 to 2.0 m ahead (10 values) joins what the robot observes |

The draws come from the run's seed, so a run repeats exactly. While it
trains, the RL Training Dashboard shows **Difficulty** (the curriculum's
level) beside the return; every MuJoCo task also shows **Falls** (of the last
100 episodes, those that ended early or upside down).

**Look ahead or not.** HalfCheetah observes only its own joints: without
Look ahead it is blind to a step or a block until it hits it, and can learn
to cope with rough ground, slopes, small steps and pushes but rarely with
tall steps or blocks. Look ahead lets it see them coming. It changes what the
policy observes, so it trains a new policy from the start, and that policy
plays only with Look ahead on (its saved settings bring it along).

A saved policy keeps all of these settings: **Play policy** plays it in the
conditions it trained in (a curriculum starts again from its easiest ground),
and the Stress panel can still change the ground.

## While it trains: watch the agent learn

The RL Training Dashboard's **Watch progress** (for an environment with a
view, such as the MuJoCo tasks) shows how the policy moves while it trains:

- **Live preview** plays the policy as it is now: every *N* updates (20 by
  default; **Now** takes the next update's) the run copies its networks, and
  one copy of the environment plays them in the MuJoCo Viewport at real time
  with the best actions, on the run's own ground and settings. Each new
  episode plays the newest networks. The Viewport says **Preview of update
  1,580** and **Training preview**; the dashboard's card says which update
  and steps the networks are from, how many episodes the preview played, and
  the last return with the update that played it. The preview has its own
  copy of the networks and of the environment: training is not slowed or
  changed (a little GPU time at each copy), and the Stress panel can disturb
  the preview's copy.
- **Raw training copy** shows copy 0 of the training itself, as it learns:
  at training pace (much faster than real time) and with the exploring,
  sampled actions. The Viewport says **Training copy 0**. It is watch only:
  the Stress panel is off for it, since a push or a new ground would change
  what training learns.
- **Off** shows nothing.

**Show the MuJoCo Viewport** opens the view. Preferences > Simulation >
Policy playback keeps the default (**Watch training**: Off, every 20
updates). Playing a policy (Policy Playback) takes the view: Watch progress
goes Off. The preview stops when the run ends; **Play policy** then plays the
trained policy.

## After training: watch, save, reopen

The trained policy stays in the Engine until the next Train.

- **Play policy** again at any time, from the dashboard or from **Play**
  under *Trained policy* in the PPO node's Properties. In Policy Playback:
  - **Best** takes the policy's most likely action; **Sampled** draws
    actions as training did.
  - **Real time**, **x4** or **As fast as possible**.
  - **Pause**, **New episode**, **Stop**.
- **Save policy...** writes one `.cyxmodel` file (by default
  `<project>/exports/<graph name>.cyxmodel`). It holds the RL graph, both
  networks, the optimizer states and `rl/policy.json`: the environment, its
  settings and the observation-normalisation statistics the policy expects.
- **Later:** **File > Open Model File...** on that file and **Play policy**
  plays it without touching the open graph; **Open its graph** opens the
  graph it was trained with. Or open the graph and use **Load policy...** in
  the PPO node's Properties. A MuJoCo policy needs the MuJoCo plugin loaded.

## Stress test: how much does a policy take?

Policy Playback's **Stress test** tab plays a policy on harder and harder
conditions and measures it, alone or against a second saved policy (a MuJoCo
task; the built-in CartPole has no grounds or pushes).

1. **Compare with**: type the path of a saved `.cyxmodel` and **Load** it
   (optional). It must play the same environment.
2. **Disturbance**: Rough, Slopes, Steps, Obstacles (that ground at each
   level's difficulty) or Kicks (a push every 1.5 s of up to 4 m/s times the
   difficulty).
3. **Levels** (difficulty 0 to 1) and **episodes a level**, then **Run test**.

Each level's episodes run side by side with the policy's best actions. A
level has its own seed, so both policies meet the same grounds and the same
pushes. The policies' own random physics and curriculum are off for the test;
each keeps its own Look ahead. The chart shows the mean return by difficulty,
the table each level's mean return and falls (an episode that ended early or
upside down). **Watch** plays that level's first episode again in the Play
tab, and the MuJoCo Viewport shows it. **Export CSV** saves the table in the
project's exports folder. **Keep these as the default** saves the disturbance,
levels and episodes; Preferences > Simulation > Policy playback holds them too.

The example `examples/cyxgraph/rl/ppo_half_cheetah_robust.cyxgraph` trains
HalfCheetah with the four grounds, a curriculum, pushes, varied friction and
mass, and Look ahead: compare its saved policy with the flat-ground one.

## Export: using a policy outside the Engine

**Today:**

- The `.cyxmodel` file is the Engine's format: it reopens and plays in any
  CyxWiz Engine with the same plugin.
- **Code Gen** writes the same training as a Python script: in the Studio,
  pick **Code Gen** mode, then **Generate** (an RL graph always gives the
  Stable-Baselines3 script, whichever framework is selected). The script
  trains Stable-Baselines3's PPO on the Engine's own environments
  (`pycyxwiz.rl`, the same physics), with the graph's layers and settings,
  and saves the model as `ppo_<environment>` (and, when it normalises, as
  HalfCheetah does, the observation statistics as
  `ppo_<environment>_vecnormalize.pkl`). Stable-Baselines3 documents how to
  export a saved policy to ONNX or TorchScript (its "Exporting models"
  guide); apply the saved normalisation to every observation before the
  exported policy sees it.

**Planned:** exporting an Engine-trained policy straight to ONNX, and
continuing training from a saved policy.

## Using a policy on a real robot

The three examples are simulator benchmarks: there is no physical
HalfCheetah, and these policies are not meant to drive hardware as they are.
What a policy learns is a mapping from *this simulator's observations* to
*this simulator's actuator commands*. To run a policy on a real machine:

1. **Train on a model of that machine.** The robot's own MuJoCo model (MJCF;
   the MuJoCo Menagerie has models of many real robots), with its real
   joints, masses and actuator limits, and a reward for the behaviour you
   want.
2. **Match the observation.** The robot's controller has to build exactly
   the vector the policy was trained on: the same sensor readings, in the same
   order and units, normalised with the saved statistics.
3. **Match the action.** The robot's actuators must accept the same
   commands (for example joint torques in [-1, 1] scaled to the motors), sent
   at the same control rate as the simulator's step (0.05 s for HalfCheetah).
4. **Close the reality gap.** A real robot differs from its model (friction,
   delays, sensor noise). The usual remedies are randomising those quantities
   during training, adding latency to the simulated observations, and
   testing first with the robot held or tethered, under a hardware safety
   stop.
5. **Run the policy in the robot's control loop**: read the sensors, build
   and normalise the observation, run the exported policy (ONNX Runtime and
   TorchScript run on most robot computers), clamp the action to safe limits,
   send it, repeat at the control rate.

What the Engine covers today: step 1 for its own tasks only (the plugin's
environments are fixed; training on your own robot model is planned), the
observation statistics for step 2 (in the saved policy), and steps 4-5 through
the Stable-Baselines3 script and its export. Training on a custom MuJoCo
robot model and direct ONNX export are on the roadmap.
