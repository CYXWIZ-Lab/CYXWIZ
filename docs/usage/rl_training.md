# Training a policy (PPO and SAC)

New to it? [Train, play and export a policy](rl_walkthrough.md) walks through the three examples step by step.

The Engine trains reinforcement learning policies itself, from a graph on
the canvas: no Python, no Stable-Baselines3. Two algorithms, each written to
match Stable-Baselines3's (losses, gradients, updates, normalisation and
defaults checked against PyTorch and SB3; see
`cyxwiz-engine/tests/computation_truth/README.md`):

- **PPO** (on-policy): learns from fresh rollouts, discrete or continuous
  actions, quick per step. The graph and settings are below.
- **SAC** (Soft Actor-Critic, off-policy): learns continuous actions from a
  replay buffer of past steps, so it needs far fewer environment steps for
  the same return (on HalfCheetah about twice PPO's return in the same
  1,000,000 steps), but each step costs more. See [SAC](#sac).

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

## Compile and Train

The compile (F7, and the live check while you edit) says what is wrong, on
the node it concerns:

- an environment whose plugin is not loaded (load it in Plugin Manager);
- a head nothing reaches from the Environment's Observation, or a broken chain;
- a minibatch larger than one rollout (Copies x Steps per copy); for SAC, a
  batch larger than the replay buffer;
- SAC on an environment with a choice of actions, or Q layers that do not
  start with a Dense;
- a layer the networks cannot use yet (only Dense, activations, Layer Norm, Dropout);
- data, loss or optimizer nodes in the same graph (a graph trains either a
  model on data or a policy on an environment);
- a warning for a Softmax right before the Policy Head.

**Train** (F5) then runs PPO or SAC on a worker: the toolbar shows the update and
the steps with **Stop**, and the **RL Training Dashboard** opens. Local
Debug does not apply to an RL graph (it runs one supervised batch).

## Where it trains

PPO and SAC train on the device chosen in **Preferences > Devices**, the same choice
and the same policy as any other training run (SAC's replay buffer lives
there too: on the CPU, in the computer's memory):

- **Strict residency**: the chosen device must start, or the run stops with the reason.
- **Compatibility**: if the chosen device fails its check, the run trains on
  the CPU and says so (the chip turns to the caution colour and reads
  "CPU · fallback"; the note under the progress bar gives the reason).

Where the run actually trains is shown on every surface:

- the dashboard chip, for example **CPU · Intel** or **CUDA · GeForce GTX 1050 Ti**;
  hover it for the full route (backend, device, policy);
- the toolbar line, for example `PPO update 22 · 5,632 steps · CPU` (hover:
  the same route);
- the Viewport's **Active Compute Runtime** (`ACTIVE RUN CPU`);
- Policy Playback's source line (`policy from this session's run (67,584 steps) · on CPU`):
  a trained policy acts on the device it trained on.

On an NVIDIA GPU with the CUDA GPU pack, PPO and SAC run their networks as
fused kernels (SAC: the actor, both Q networks and their targets, with the
Q networks' input gradient for the actor's loss), which trains several times
faster than layer by layer, when the networks fit them: Dense layers (at most 8, each at most 256 units) with ReLU, Leaky
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
the toolbar says "PPO waiting for the GPU" and the dashboard chip says why;
Stop ends the wait. See [What runs where on a GPU](gpu_providers.md).

## The RL Training Dashboard

- Status, the run (environment and copies), **Pause** / **Resume**, **Stop**.
- Progress: steps of the total, the update, the target return if set.
- Cards: episode reward and length (mean of the last 100 episodes), policy
  loss, speed (environment steps per second). A SAC run shows critic loss,
  actor loss (with the mean Q), the entropy weight (learned or fixed, and its
  target), the **Replay buffer** (steps held of its size, and where it lives),
  episode length and speed; before learning starts the critic card says how
  many random steps remain.
- **Episodes**: reward and length by environment step. Under the reward, the
  **SB3 reference**: the mean of Stable-Baselines3's recorded runs on the
  same environment with the same copies and settings
  (`cyxwiz-engine/resources/rl_reference/ppo_sb3_curves.json`, made by
  `generate_ppo_sb3_curves.py`; SAC's `sac_sb3_curves.json` from
  `generate_sac_sb3_curves.py`, InvertedPendulum at the defaults and
  HalfCheetah at its tuned settings). With other settings there is no
  reference: a curve of other settings would mislead.
- **Policy diagnostics**: policy loss, value loss, explained variance,
  entropy, approximate KL, clip fraction, and the action standard deviation
  (continuous actions). SAC: critic loss, actor loss, entropy weight and
  mean Q.

A script can report to the same window with `pycyxwiz.rl_update_metric`
(see [Plotting](plotting.md#rl-training-dashboard)).

## Play, save and load the trained policy

When a run finishes, or you stop it after at least one update, the Engine
keeps its trained policy until the next Train.

- **Play policy** (on the dashboard after the run, or **Play** under
  **Trained policy** in the PPO or SAC node's Properties) opens the **Policy
  Playback** panel and plays one copy of the environment with the policy:
  - **Actions:** *Best* takes the policy's most likely action (the most
    likely choice, or the Gaussian mean; SAC: the squashed mean); *Sampled*
    draws them as training does.
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
    is not saved).
- **Load policy...** in the PPO or SAC node's Properties loads a saved policy for
  this graph. It is refused, saying what differs, when it was trained on
  another environment or other layer widths.
- **File > Open Model File...** on a policy file shows "PPO on
  <environment>" (or "SAC on ...") and offers **Play policy**, which plays it without
  changing the open graph, and **Open its graph**.
- **Loading needs the environment:** the saved graph is compiled as Train
  compiles it, so a MuJoCo task needs the MuJoCo plugin loaded.

Preferences > Simulation > Policy playback sets how the panel starts:
- Actions (default Best).
- Speed (default Real time).
- Whether to play the policy as soon as a run finishes (default off).

Preferences > Simulation > Replay buffer size sets the buffer a new SAC node
starts with (Tuned for sets the task's own).

## Code Gen

With **Code Gen** selected, **Generate** writes the same run as a Python
script: Stable-Baselines3's PPO (or SAC) on the Engine's own environments
(`pycyxwiz.rl`, the same physics), with the graph's layer widths,
activation and settings (and VecNormalize when normalising). Layers SB3's
MlpPolicy cannot express are named in a comment.

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

## Not yet

- Exporting the policy to ONNX, and continuing training from a
  saved policy (its optimizer states are already in the file).
- Algorithms other than PPO and SAC (TD3 and DQN are planned);
  learning-rate schedules (RL Zoo's CartPole settings use linear ones, the
  preset uses them as constants).
- Training on a MuJoCo Plant model with a reward graph (the environments are
  the plugin's fixed tasks).
