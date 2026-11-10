# Training a policy (PPO)

The Engine trains reinforcement learning policies itself, from a graph on
the canvas: no Python, no Stable-Baselines3. The algorithm is PPO, written
to match Stable-Baselines3's PPO (its loss, gradients, advantage estimates,
normalisation and defaults are checked against PyTorch and SB3; see
`cyxwiz-engine/tests/computation_truth/README.md`).

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
| | Max gradient norm (`max_grad_norm`, over both networks) | 0.5 |
| | Normalise advantages (`normalize_advantage`) | on |
| Normalise | Observations, Rewards: running statistics as SB3's VecNormalize (clip 10) | off |

## Compile and Train

The compile (F7, and the live check while you edit) says what is wrong, on
the node it concerns:

- an environment whose plugin is not loaded (load it in Plugin Manager);
- a head nothing reaches from the Environment's Observation, or a broken chain;
- a minibatch larger than one rollout (Copies x Steps per copy);
- a layer the networks cannot use yet (only Dense, activations, Layer Norm, Dropout);
- data, loss or optimizer nodes in the same graph (a graph trains either a
  model on data or a policy on an environment);
- a warning for a Softmax right before the Policy Head.

**Train** (F5) then runs PPO on a worker: the toolbar shows the update and
the steps with **Stop**, and the **RL Training Dashboard** opens. Local
Debug does not apply to an RL graph (it runs one supervised batch).

## Where it trains

PPO trains on the device chosen in **Preferences > Devices**, the same choice
and the same policy as any other training run:

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

Small networks (every hidden layer at most 256 units, such as the 64-64
pattern) usually train faster on the CPU: each environment step runs many
small GPU operations, and some graphics drivers struggle under that load.
When the chosen device is a GPU, Compile adds an Info line saying so and the
dashboard repeats it; the Engine never switches devices for you. While a PPO
run is active, the device choices in Preferences are disabled until it ends,
and a second training run (Train on a model graph) is refused.

## The RL Training Dashboard

- Status, the run (environment and copies), **Pause** / **Resume**, **Stop**.
- Progress: steps of the total, the update, the target return if set.
- Cards: episode reward and length (mean of the last 100 episodes), policy
  loss, speed (environment steps per second).
- **Episodes**: reward and length by environment step. Under the reward, the
  **SB3 reference**: the mean of Stable-Baselines3's recorded runs on the
  same environment with the same copies and settings
  (`cyxwiz-engine/resources/rl_reference/ppo_sb3_curves.json`, made by
  `generate_ppo_sb3_curves.py`). With other settings there is no reference:
  a curve of other settings would mislead.
- **Policy diagnostics**: policy loss, value loss, explained variance,
  entropy, approximate KL, clip fraction, and the action standard deviation
  (continuous actions).

A script can report to the same window with `pycyxwiz.rl_update_metric`
(see [Plotting](plotting.md#rl-training-dashboard)).

## Play, save and load the trained policy

When a run finishes, or you stop it after at least one update, the Engine
keeps its trained policy until the next Train.

- **Play policy** (on the dashboard after the run, or **Play** under
  **Trained policy** in the PPO node's Properties) opens the **Policy
  Playback** panel and plays one copy of the environment with the policy:
  - **Actions:** *Best* takes the policy's most likely action (the most
    likely choice, or the Gaussian mean); *Sampled* draws them as training does.
  - **Speed:** *Real time* steps at the environment's own step time, *x4*
    four times faster, or *As fast as possible*.
  - The panel shows the episode, its step and return, the last returns and
    the action, with Pause / Resume, New episode and Stop.
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
- **Load policy...** in the PPO node's Properties loads a saved policy for
  this graph. It is refused, saying what differs, when it was trained on
  another environment or other layer widths.
- **File > Open Model File...** on a policy file shows "PPO on
  <environment>" and offers **Play policy**, which plays it without
  changing the open graph, and **Open its graph**.
- **Loading needs the environment:** the saved graph is compiled as Train
  compiles it, so a MuJoCo task needs the MuJoCo plugin loaded.

Preferences > Simulation > Policy playback sets how the panel starts:
- Actions (default Best).
- Speed (default Real time).
- Whether to play the policy as soon as a PPO run finishes (default off).

## Code Gen

With **Code Gen** selected, **Generate** writes the same run as a Python
script: Stable-Baselines3's PPO on the Engine's own environments
(`pycyxwiz.rl`, the same physics), with the graph's layer widths,
activation and settings (and VecNormalize when normalising). Layers SB3's
MlpPolicy cannot express are named in a comment.

## How well it learns

On 2026-11-09 with the tuned settings:

| Environment | Engine (test) | SB3 2.7.1 on the same environment, 3 seeds |
|---|---|---|
| CartPole-v1, mean return 475 | 69,376 steps (test_ppo_trainer); 88,320 from the graph (test_rl_graph_training) | 59,648 / 64,256 / 72,192 |
| InvertedPendulum-v5, mean return 950 | 122,880 steps | 120,832 / 135,168 / 161,792 |

## Not yet

- Exporting the policy to ONNX (TOFIX188), and continuing training from a
  saved policy (its optimizer states are already in the file).
- Algorithms other than PPO; learning-rate schedules (RL Zoo's CartPole
  settings use linear ones, the preset uses them as constants).
- Training on a MuJoCo Plant model with a reward graph (the environments are
  the plugin's fixed tasks).
