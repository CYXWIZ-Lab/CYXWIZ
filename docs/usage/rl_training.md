# Training a policy (PPO, SAC, TD3 and DQN)

New to it? [Train, play and export a policy](rl_walkthrough.md) walks through the three examples step by step.

The Engine trains reinforcement learning policies itself, from a graph on
the canvas: no Python, no Stable-Baselines3. Four algorithms, each written to
match Stable-Baselines3's (losses, gradients, updates, normalisation and
defaults checked against PyTorch and SB3; see
`cyxwiz-engine/tests/computation_truth/README.md`):

- **PPO** (on-policy): learns from fresh rollouts, discrete or continuous
  actions, quick per step. The graph and settings are below.
- **SAC** (Soft Actor-Critic, off-policy): learns continuous actions from a
  replay buffer of past steps, so it needs far fewer environment steps for
  the same return (on HalfCheetah about twice PPO's return in the same
  1,000,000 steps), but each step costs more. See [SAC](#sac).
- **TD3** (Twin Delayed DDPG, off-policy): learns continuous actions from a
  replay buffer too, with a deterministic policy that explores with added
  noise. Same graph shape as SAC. See [TD3](#td3).
- **DQN** (Deep Q-Network, off-policy): learns a choice of actions from a
  replay buffer, with one Q network and no Policy Head; it explores by
  choosing at random a share of the time. See [DQN](#dqn).

## The graph

```
Environment ─Observation─▶ Dense ▶ Tanh ▶ Dense ▶ Tanh ▶ Policy Head ─▶┐
            ─Observation─▶ Dense ▶ Tanh ▶ Dense ▶ Tanh ▶ Value Head  ─▶ PPO
            ─Environment─────────────────────────────────────────────▶┘
```

The quickest start: the **Pattern Browser** (View > Patterns) has **PPO (MLP
64-64)** under Reinforcement Learning; **Insert Pattern** drops the whole
graph (choose the hidden width and the activation when you insert it).

| Node | What it does |
|---|---|
| **Environment** | Picks an environment (the built-in CartPole-v1, or one a loaded plugin offers, such as the MuJoCo Simulation plugin's InvertedPendulum-v5 and HalfCheetah-v5; see [Environments](rl_environments.md)) and how many **Copies** step side by side. Properties lists the environment's facts (observation, actions, episode limit, step time) and its own settings, with the plugin's labels and checks. **Observation** feeds both networks; **Environment** goes to PPO. |
| Layers | Ordinary layer nodes, the ones supervised graphs use: Dense, the activations, Layer Norm, Dropout. The policy and the value network are separate chains, as in SB3's MlpPolicy. |
| **Policy Head** | Ends the policy layers. Adds the output layer sized from the environment: one logit per choice (categorical) for discrete actions, one mean per action (Gaussian, with one learned log standard deviation per action, **Log std at start**) for continuous ones. Properties shows what it adds. Do not put a Softmax before it. |
| **Value Head** | Ends the value layers: adds the one-output layer (the expected return). |
| **PPO** | The training settings (below). **Tuned for <environment>** loads tuned values for CartPole-v1, InvertedPendulum-v5 and HalfCheetah-v5 (RL Zoo's, or the Engine's tested ones; hover for the source). |

The networks start as SB3's do: orthogonal weights with gain sqrt 2, the
policy's output layer 0.01, the value's 1, biases zero.

## PPO's settings

Hover a label in Properties for its Stable-Baselines3 name. The defaults are
SB3's.

| Group | Setting (SB3 name) | Default |
|---|---|---|
| Run | Total steps (`total_timesteps`) | 100000 |
| | Stop at a mean return / Target return: stop once the last 100 episodes average it | off / 475 |
| | Seed (`seed`): the copies, the weights, the actions and the minibatches | 1 |
| Rollout | Steps per copy (`n_steps`); one rollout is Copies x Steps per copy | 2048 |
| | Discount (`gamma`), GAE lambda (`gae_lambda`) | 0.99, 0.95 |
| Update | Epochs (`n_epochs`), Minibatch size (`batch_size`) | 10, 64 |
| | Learning rate (`learning_rate`, Adam with eps 1e-5) | 0.0003 |
| | Clip range (`clip_range`), Value clip (`clip_range_vf`, 0 off) | 0.2, 0 |
| | Entropy weight (`ent_coef`), Value weight (`vf_coef`) | 0, 0.5 |
| | Max gradient norm (`max_grad_norm`, over both networks; as PyTorch's `clip_grad_norm_`: a NaN norm is not refused and makes the step NaN) | 0.5 |
| | Normalise advantages (`normalize_advantage`) | on |
| Normalise | Observations, Rewards: running statistics as SB3's VecNormalize (clip 10) | off |

## SAC

```
Environment ─Observation─▶ Dense ▶ ReLU ▶ Dense ▶ ReLU ▶ Policy Head ─▶┐
            ─Observation─▶ Dense ▶ ReLU ▶ Dense ▶ ReLU ▶ Q Head      ─▶ SAC
            ─Environment─────────────────────────────────────────────▶┘
```

The Pattern Browser has **SAC (MLP 256-256)** under Reinforcement Learning.
SAC learns continuous actions only (InvertedPendulum-v5, HalfCheetah-v5);
on CartPole's choice of actions Compile says to train it with PPO.

| Node | What it does |
|---|---|
| **Policy Head** | Feeding SAC, it adds the actor's output: a mean and a log standard deviation per action, squashed by tanh into the action's range (Properties says so). |
| **Q Head** | Ends the Q layers with one output, the value of an action in a state. The action joins the first Q layer's input, so the Q layers must start with a Dense. SAC builds two Q networks of this shape and a slowly following target copy of each. |
| **SAC** | The training settings (below). **Tuned for HalfCheetah-v5** loads RL Zoo's values (hover for the source); for InvertedPendulum the defaults are the tested ones. Properties also shows what the **Replay buffer** takes on the training device (and a note when it would not fit) and the **Target entropy** it aims for. |

The networks start as SB3's SAC does (PyTorch's default initialisation).

| Group | Setting (SB3 name) | Default |
|---|---|---|
| Run | Total steps (`total_timesteps`) | 1000000 |
| | Stop at a mean return / Target return | off / 4000 |
| | Seed (`seed`) | 1 |
| Replay buffer | Buffer size (`buffer_size`): steps kept on the training device | 1000000 (Preferences > Simulation > Replay buffer size sets a new node's) |
| | Random steps first (`learning_starts`): uniform random actions before learning | 100 |
| Updates | Batch size (`batch_size`), Learning rate (`learning_rate`, Adam) | 256, 0.0003 |
| | Discount (`gamma`), Target update (`tau`, Polyak) | 0.99, 0.005 |
| | Train every (`train_freq`), Gradient steps (`gradient_steps`), Target every (`target_update_interval`) | 1, 1, 1 |
| Exploration (entropy) | Entropy weight: Learned (`ent_coef="auto"`, starting at Weight) or Fixed | Learned, 1.0 |
| | Target entropy: Minus the actions (`"auto"`) or Typed | Minus the actions |
| Normalise | Observations (VecNormalize's `norm_obs`) | off |

Each environment step adds one step per copy to the buffer; after the random
steps, every Train every steps SAC draws Batch size steps from the buffer and
updates the entropy weight, then the twin Q networks, then the policy on the
updated Q networks, then moves the targets by Target update (SB3's order). One copy (the
Environment's Copies) is SB3's setting and the tuned one.

## TD3

```
Environment ─Observation─▶ Dense ▶ ReLU ▶ Dense ▶ ReLU ▶ Policy Head ─▶┐
            ─Observation─▶ Dense ▶ ReLU ▶ Dense ▶ ReLU ▶ Q Head      ─▶ TD3
            ─Environment─────────────────────────────────────────────▶┘
```

The Pattern Browser has **TD3 (MLP 400-300)** under Reinforcement Learning
(SB3's and RL Zoo's layers of 400 and 300). TD3 learns continuous actions
only; on a choice of actions Compile says to train it with PPO.

| Node | What it does |
|---|---|
| **Policy Head** | Feeding TD3, it adds one output per action through tanh: a deterministic action in the action's range (no log standard deviation; Log std at start is hidden). TD3 keeps a target copy of the policy for the Q targets. |
| **Q Head** | As under SAC: TD3 builds two Q networks and a target copy of each; the policy learns from the first Q network. |
| **TD3** | The training settings (below). **Tuned for HalfCheetah-v5** loads RL Zoo's values; for InvertedPendulum the defaults are the tested ones. Properties shows the **Replay buffer** on the training device and the **Exploration**. |

The networks start as SB3's TD3 does (PyTorch's default initialisation).

| Group | Setting (SB3 name) | Default |
|---|---|---|
| Run | Total steps (`total_timesteps`) | 1000000 |
| | Stop at a mean return / Target return | off / 4000 |
| | Seed (`seed`) | 1 |
| Replay buffer | Buffer size (`buffer_size`) | 1000000 (Preferences > Simulation > Replay buffer size sets a new node's) |
| | Random steps first (`learning_starts`) | 100 |
| Updates | Batch size (`batch_size`), Learning rate (`learning_rate`, Adam) | 256, 0.001 |
| | Discount (`gamma`), Target update (`tau`, Polyak) | 0.99, 0.005 |
| | Train every (`train_freq`), Gradient steps (`gradient_steps`) | 1, 1 |
| | Policy every (`policy_delay`): Q network updates for each policy update; the targets move with the policy | 2 |
| Exploration (noise) | Action noise: Gaussian (`NormalActionNoise`) or None; Noise std on the actions scaled to [-1, 1] | Gaussian, 0.1 |
| Target smoothing | Noise (`target_policy_noise`), Clipped at (`target_noise_clip`): noise on the target policy's actions for the Q targets | 0.2, 0.5 |
| Normalise | Observations (VecNormalize's `norm_obs`) | off |

The noise is the one default that is not SB3's: SB3's TD3 adds none unless
told, and then barely explores after the random steps; RL Zoo trains every
MuJoCo task with Gaussian noise of 0.1, so the node starts there. As SB3
does, the noise is added to the random steps' actions too.

Every step TD3 draws Batch size steps from the buffer and updates both Q
networks toward the smaller target Q of the target policy's smoothed
actions; every Policy every steps it updates the policy from the first Q
network and moves all three targets by Target update (SB3's order).

## DQN

```
Environment ─Observation─▶ Dense ▶ ReLU ▶ Dense ▶ ReLU ▶ Q Head ─▶┐
            ─Environment───────────────────────────────────────▶ DQN
```

The Pattern Browser has **DQN (MLP 64-64)** under Reinforcement Learning
(SB3's DQN MlpPolicy). DQN chooses among actions only (CartPole-v1, built
in); on continuous actions Compile says to train it with SAC, TD3 or PPO.
There is no Policy Head: the policy is the choice with the highest Q value.

| Node | What it does |
|---|---|
| **Q Head** | Feeding DQN, it adds one output per choice of action, from the observation alone (the action does not join the first layer, unlike under SAC and TD3), so its layers may start with any allowed layer. DQN keeps a target copy of the network. |
| **DQN** | The training settings (below). **Tuned for CartPole-v1** loads RL Zoo's values (its Q layers of 256 and 256 are noted, not set). Properties shows the **Replay buffer** on the training device and the **Exploration**: "random 100 % of the time at first, down to 4 % after 8,000 steps (16 %)". |

The network starts as SB3's DQN does (PyTorch's default initialisation).

| Group | Setting (SB3 name) | Default |
|---|---|---|
| Run | Total steps (`total_timesteps`) | 1000000 |
| | Stop at a mean return / Target return | off / 475 |
| | Seed (`seed`) | 1 |
| Replay buffer | Buffer size (`buffer_size`) | 1000000 (Preferences > Simulation > Replay buffer size sets a new node's) |
| | Random steps first (`learning_starts`) | 100 |
| Updates | Batch size (`batch_size`), Learning rate (`learning_rate`, Adam) | 32, 0.0001 |
| | Discount (`gamma`) | 0.99 |
| | Train every (`train_freq`), Gradient steps (`gradient_steps`) | 4, 1 |
| | Target every (`target_update_interval`, environment steps), Target update (`tau`: 1 copies the network) | 10000, 1.0 |
| | Max gradient norm (`max_grad_norm`): the gradients' overall norm is clipped to it | 10 |
| Exploration (epsilon) | Start (`exploration_initial_eps`), End (`exploration_final_eps`), Over (`exploration_fraction`, a share of the run) | 1.0, 0.05, 0.1 |
| Normalise | Observations (VecNormalize's `norm_obs`) | off |

Every Train every steps DQN draws Batch size steps from the buffer and moves
the Q network toward the reward plus the discounted best Q of the next state
under the target network (Huber loss, the gradients' norm clipped, Adam);
every Target every steps the target moves to it. While training it picks at
random with the share epsilon (one draw for all the copies, as SB3's
predict), which falls linearly from Start to End over the first Over of the
run.

## Compile and Train

The compile (F7, and the live check while you edit) says what is wrong, on
the node it concerns:

- an environment whose plugin is not loaded (load it in Plugin Manager);
- a head nothing reaches from the Environment's Observation, or a broken chain;
- a minibatch larger than one rollout (Copies x Steps per copy); for SAC, a
  batch larger than the replay buffer;
- SAC or TD3 on an environment with a choice of actions or on actions without
  finite bounds, or Q layers that do not start with a Dense; TD3's noise std
  of 0 with Gaussian noise; DQN on continuous actions, or an exploration
  start, end or share outside 0 to 1;
- a layer the networks cannot use yet (only Dense, activations, Layer Norm, Dropout);
- data, loss or optimizer nodes in the same graph (a graph trains either a
  model on data or a policy on an environment);
- a warning for a Softmax right before the Policy Head.

**Train** (F5) then runs PPO, SAC, TD3 or DQN on a worker: the toolbar shows the update and
the steps with **Stop**, and the **RL Training Dashboard** opens. Local
Debug does not apply to an RL graph (it runs one supervised batch).

## Where it trains

PPO, SAC, TD3 and DQN train on the device chosen in **Preferences > Devices**, the same choice
and the same policy as any other training run (the replay buffer lives
there too: on the CPU, in the computer's memory):

- **Strict residency**: the chosen device must start, or the run stops with the reason.
- **Compatibility**: if the chosen device fails its check, the run trains on
  the CPU and says so (the dashboard's "Runs on" line turns to the caution
  colour and names the device that failed; the note under it gives the reason).

The policy also decides what happens when a GPU provider's kernel fails
during the run (the fused networks, the joint Adam step, PPO's sampling, loss
and advantage kernels): under Strict the run stops and names the provider,
the operation and the part that failed; under Compatibility that part goes on
with ArrayFire operations on the same device and the run's trace records the
failure (for the networks, **Run details** also says how they now compute).
Every provider call is counted in the run's trace (Studio Debugger, Runtime:
provider groups), as in [Strict and the providers](gpu_providers.md#strict-and-the-providers).

Where the run actually trains is shown on every surface:

- the dashboard's line under the progress bar, for example
  "Runs on the CPU (Intel, ArrayFire CPU) · compatibility policy"; **Run
  details** gives the full route and how the networks compute;
- the toolbar line, for example `PPO update 22 · 5,632 steps · CPU` (hover:
  the same route);
- the Viewport's **Active Compute Runtime** (`ACTIVE RUN CPU`);
- Policy Playback's source line (`policy from this session's run (67,584 steps) · on CPU`):
  a trained policy acts on the device it trained on.

On an NVIDIA GPU with the CUDA GPU pack, PPO, SAC, TD3 and DQN run their networks as
fused kernels (SAC and TD3: the actor, both Q networks and their targets, with
the Q networks' input gradient for the actor's loss; DQN: the Q network and its
target), which trains several times
faster than layer by layer, when the networks fit them: Dense layers (at most 8, each at most 512 units) with ReLU, Leaky
ReLU, ELU, SELU, GELU, Swish, Tanh or Sigmoid between them. Otherwise (other
layers such as Dropout or LayerNorm, OpenCL, oneAPI, or CUDA without the
pack) it runs layer by layer, and small networks
(every hidden layer at most 256 units, such as the 64-64 pattern) usually
train faster on the CPU: each environment step runs many small GPU
operations, and some graphics drivers struggle under that load. Only then
does Compile add an Info line saying so, and the dashboard repeats it once
the run reports how it computes; the Engine never switches devices for you.
See [Why small networks are slow on a GPU](gpu_providers.md#why-small-networks-are-slow-on-a-gpu-and-what-cyxwiz-does-about-it). While a PPO
run is active, the device choices in Preferences are disabled until it ends,
and a second training run (Train on a model graph) is refused. A run on a GPU
that another training run (another Engine, a tool) is using waits for it:
the toolbar says "PPO waiting for the GPU" and the dashboard says why;
Stop ends the wait. See [What runs where on a GPU](gpu_providers.md).

## The RL Training Dashboard

It opens with **Train** on an RL graph, and from **Train > RL Training
Dashboard** at any time (with no run it says how to start one). It uses the
same parts and words as the [Training Dashboard](training_dashboard.md).

**Header.** A pill says what is happening: **Starting**, **Training**,
**Paused**, **Waiting for the GPU**, and after the run **Reached its target**
(the mean return of the last 100 episodes reached the target return, so it
stopped early), **Ran all its steps**, **Stopped** or **Failed**. Beside it
the update and the steps ("Update 121 · 30,976 of 200,000 steps"; a SAC, TD3
or DQN run: the steps, its updates and its random steps, "212,480 of 1,000,000
steps · 202,480 updates · learning started at 10,000 (random steps
before)", and for TD3 its policy updates, "the policy every 2nd (101,240)"),
after the run the steps and the time it took. Then the progress bar, a line
for the run and its target ("PPO on CartPole-v1 × 8 · stops early at a mean return of
475 over the last 100 episodes"), a line for where it runs, and when it ends
a sentence saying how ("Reached its target: mean return 477.4 ... at update
254"). **Run details** folds out the device route, how the networks compute
(fused kernels or ArrayFire operations), the steps per update (SAC, TD3, DQN:
the steps between reports and the updates so far) and the reference.

Buttons: **Pause** / **Resume** and **Stop** while it runs; **Play policy**
and **Save policy...** after it; **More** has **Export updates (CSV)...**
(every update's steps, mean return and length, losses, entropy, approximate
KL, clip fraction, explained variance, action spread and speed; it asks where
to save and says where it saved; a SAC, TD3 or DQN run's reports have its
gradient steps, critic and actor loss, mean Q, entropy weight or exploration
rate and replay buffer instead) and **Clear all...**, which asks first: the
run's Play and Save buttons go with it, so save the policy before.

**Cards.**
- **Mean return**: of the last 100 episodes, its change over the last 10
  updates, the best so far, the target, and the SB3 reference at this step.
- **Episode length**: of the last 100 episodes.
- **Time** (after the run **Total time**): the time the updates took (time
  paused or waiting for the GPU is left out), the speed now and on average,
  and at most how long the remaining steps take (less if the run reaches its
  target).
- **Updates**: how many, with the last policy and value loss, approximate KL
  and clip fraction (hover for entropy, explained variance and action spread).
- A SAC run shows, after the mean return, **Critic loss** (both Q networks,
  and how many updates it averages; before learning starts, how many random
  steps remain), **Actor loss** (with the mean Q), **Entropy weight** (learned
  or fixed, and its target) and the **Replay buffer** (steps held of its
  size, and where it lives), then episode length, **Speed** (environment
  steps a second and the updates each) and time. A TD3 run shows the same
  cards with **Exploration** (the noise std and the target smoothing) in
  place of the entropy weight. A DQN run shows **Loss** (Huber, and how many
  updates it averages), **Exploration** (the share of random choices now and
  how it falls), **Mean Q** of the chosen actions and the replay buffer.
- An environment's own figures (a curriculum's Difficulty) on a row of their own.

**Charts.** **Episodes** and **Policy diagnostics**, side by side when the
window is wide. Every update is drawn; each chart has Fit, Log Y, Legend,
Export and its own window. The mouse wheel scrolls the dashboard; **Ctrl +
wheel** zooms a chart (in a chart's own window the wheel zooms).
- **Episodes**: mean return and episode length by environment step. The
  target return is a dashed line. Under the return, the **SB3 reference**:
  the mean of Stable-Baselines3's recorded runs on the same environment with
  the same copies and settings
  (`cyxwiz-engine/resources/rl_reference/ppo_sb3_curves.json`, made by
  `generate_ppo_sb3_curves.py`; SAC's `sac_sb3_curves.json` from
  `generate_sac_sb3_curves.py`, and TD3's `td3_sb3_curves.json` from
  `generate_td3_sb3_curves.py`: InvertedPendulum at the defaults and
  HalfCheetah at its tuned settings; DQN's `dqn_sb3_curves.json` from
  `generate_dqn_sb3_curves.py`: CartPole at its tuned settings with Q layers
  of 256 and 256). With other settings, other layer widths or other copies
  there is no reference: a curve of other settings would mislead. The
  chart's "exact · all N values" counts the run's own updates.
- **Policy diagnostics**: the learner's main loss first (PPO's policy loss,
  SAC's and TD3's critic loss, DQN's loss), then policy loss, value loss,
  explained variance, entropy, approximate KL, clip fraction, and the action standard deviation
  (continuous actions). SAC: critic loss, actor loss, entropy weight and
  mean Q. TD3: critic loss, actor loss and mean Q. DQN: loss, exploration
  rate and mean Q.

A script can report to the same window with `pycyxwiz.rl_update_metric`
(see [Plotting](plotting.md#rl-training-dashboard)): the pill reads
**Reporting** while values arrive, then **From a script**; the cards show the
last episode reward (best, mean), episode length and policy loss.

## Play, save and load the trained policy

When a run finishes, or you stop it after at least one update, the Engine
keeps its trained policy until the next Train.

- **Play policy** (on the dashboard after the run, or **Play** under
  **Trained policy** in the PPO, SAC, TD3 or DQN node's Properties) opens the **Policy
  Playback** panel and plays one copy of the environment with the policy:
  - **Actions:** *Best* takes the policy's most likely action (the most
    likely choice, or the Gaussian mean; SAC: the squashed mean; TD3: its
    own action; DQN: the highest Q value); *Sampled* draws them as training
    does (TD3: with its noise; DQN: a random choice at its last epsilon).
  - **Speed:** *Real time* steps at the environment's own step time, *x4*
    four times faster, or *As fast as possible*.
  - The panel shows the episode, its step and return, the last returns and
    the action, with Pause / Resume, New episode and Stop.
  - **View > Panels > Policy Playback** (or its sidebar icon) shows or hides
    the panel. Closing it stops the playback; it starts closed at each launch
    (nothing plays yet).
- **Where you see it:**
  - **MuJoCo task:** the **MuJoCo Viewport** opens and shows the copy being
    played. Its own Run / Step / Reset are off while it does ("Policy playback
    is showing this scene"); Stop gives it back the model it had.
  - **The built-in CartPole:** the panel draws the cart and pole.
  - **An environment without a picture:** the panel shows the numbers and a
    return-per-episode line.
- **Save policy...** (dashboard, or **Save...** in Properties) writes one
  `.cyxmodel`. By default it goes in `<project>/exports/<graph name>.cyxmodel`.
  It holds:
  - the RL graph;
  - the policy and value networks with the log standard deviation;
  - the three Adam states;
  - `rl/policy.json`: the environment and its settings, the observation
    normalisation statistics and the PPO updates.
  - A SAC policy holds the actor, both Q networks and their targets, the
    learned entropy weight and their Adam states instead (the replay buffer
    is not saved). A TD3 policy holds the actor and its target, both Q
    networks and their targets, and their Adam states. A DQN policy holds
    the Q network, its target and its Adam state.
- **Load policy...** in the PPO, SAC, TD3 or DQN node's Properties loads a saved policy for
  this graph. It is refused, saying what differs, when it was trained on
  another environment or other layer widths.
- **File > Open Model File...** on a policy file shows "PPO on
  <environment>" (or "SAC on ...", "TD3 on ...", "DQN on ...") and offers **Play policy**, which plays it without
  changing the open graph, and **Open its graph**.
- **Loading needs the environment:** the saved graph is compiled as Train
  compiles it, so a MuJoCo task needs the MuJoCo plugin loaded.

Preferences > Simulation > Policy playback sets how the panel starts:
- Actions (default Best).
- Speed (default Real time).
- Whether to play the policy as soon as a run finishes (default off).

Preferences > Simulation > Replay buffer size sets the buffer a new SAC, TD3
or DQN node starts with (Tuned for sets the task's own).

## Code Gen

With **Code Gen** selected, **Generate** writes the same run as a Python
script: Stable-Baselines3's PPO (or SAC, TD3 with its NormalActionNoise, or DQN) on the Engine's own environments
(`pycyxwiz.rl`, the same physics), with the graph's layer widths,
activation and settings (and VecNormalize when normalising). Layers SB3's
MlpPolicy cannot express are named in a comment. When the graph does not
compile (for example its environment comes from a plugin that is not
loaded), the script stops at its first line with the reasons.

## How well it learns

On 2026-11-09 with the tuned settings:

| Environment | Engine (test) | SB3 2.7.1 on the same environment, 3 seeds |
|---|---|---|
| CartPole-v1, mean return 475 | 69,376 steps (test_ppo_trainer); 88,320 from the graph (test_rl_graph_training) | 59,648 / 64,256 / 72,192 |
| InvertedPendulum-v5, mean return 950 | 122,880 steps | 120,832 / 135,168 / 161,792 |
| HalfCheetah-v5, final mean return after 1,000,000 steps (2026-11-11, seeds 1-3, CUDA) | 4,675.9 / 5,087.8 / 3,787.0 | 2,973.2 / 4,651.1 / 4,660.6 |

SAC, 2026-11-18/19, seeds 1-3 on CUDA (a GTX 1050 Ti, `cyxwiz-rl-stability-gate --sac`):

| Environment | Engine | SB3 2.7.1 on the same environment, 3 seeds |
|---|---|---|
| InvertedPendulum-v5, mean return 950 (defaults, 1 copy) | 105,000 / 105,000 / 102,000 steps (12-16 minutes each) | 105,625 / 107,585 / 108,679 |
| HalfCheetah-v5, final mean return after 1,000,000 steps (RL Zoo's settings) | 9,663 / 8,348 / 11,099 (about 2-3 hours each, 100-160 steps a second) | 9,779 / 10,747 / 7,845 |

On the CPU SAC runs layer by layer and is much slower; HalfCheetah's
1,000,000 steps are a GPU run.

TD3, 2026-11-23, seeds 1-3 on CUDA (`cyxwiz-rl-stability-gate --td3`, layers of 400 and 300):

| Environment | Engine | SB3 2.7.1 on the same environment |
|---|---|---|
| InvertedPendulum-v5, mean return 950 (defaults with Gaussian noise 0.1, 1 copy) | 121,000 / 105,000 / 101,000 steps (13-18 minutes each) | 114,373 / 123,105 / 109,016 |
| HalfCheetah-v5, final mean return after 1,000,000 steps (RL Zoo's settings) | 9,789 / 9,772 / 8,906 (about 2 hours 20 minutes each) | 6 seeds: 10,275 / 10,355 / 10,325 / 9,999 / 7,734 / 8,973 |

On HalfCheetah the Engine's mean (9,489) is 98.7 % of SB3's six seeds'
(9,610). SB3's first three seeds happen to lie close together; seeds 4-6
show its usual spread.

DQN, 2026-11-24, CartPole-v1 at RL Zoo's settings (50,000 steps, Q layers of
256 and 256), 8 seeds each (`cyxwiz-rl-stability-gate --dqn`): the mean
return of the last 100 episodes over the second half of the run (25,000 to
50,000 steps), averaged over the seeds, is 154.4 on CUDA (about 1-2 minutes
a run) and 165.5 on the CPU, against SB3's 154.6 (its seeds range from 80.9
to 217.4). At these settings neither reaches a mean of 475 in 50,000 steps:
the training episodes still explore 4 % of the time, and DQN on CartPole
can lose what it learned late in a run (one SB3 seed fell from 252 to 42 in
its last 5,000 steps).

## Not yet

- Exporting the policy to ONNX, and continuing training from a
  saved policy (its optimizer states are already in the file).
- Algorithms other than PPO, SAC, TD3 and DQN (DQN's variants, Double and
  dueling DQN and prioritised replay, are not offered);
  learning-rate schedules (RL Zoo's CartPole settings use linear ones, the
  preset uses them as constants).
- Training on a MuJoCo Plant model with a reward graph (the environments are
  the plugin's fixed tasks).
