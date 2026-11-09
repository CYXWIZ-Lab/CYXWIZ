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

- Saving the trained policy and playing it in the MuJoCo viewport.
- Algorithms other than PPO; learning-rate schedules (RL Zoo's CartPole
  settings use linear ones, the preset uses them as constants).
- Training on a MuJoCo Plant model with a reward graph (the environments are
  the plugin's fixed tasks).
