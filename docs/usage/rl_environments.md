# Environments for reinforcement learning

A plugin can offer environments a policy learns in. The Engine is the
learner: it asks for many copies of one environment, sends an action to each
copy and reads back what each copy observed and earned. The plugin only
simulates. Plugin authors: see `IEnvironmentProvider` in the
[plugin API reference](../extending/plugin-api-reference.md).

Training a policy from the graph: see [Training a policy](rl_training.md)
(Environment, Policy Head, Value Head and PPO nodes). This page covers the
environments themselves and how to use them from Python.

## Built in: CartPole-v1

The Engine itself offers Gymnasium's classic-control **CartPole-v1**, with no
plugin installed (key `cyxwiz.builtin:CartPole-v1`): observes 4 values (cart
position and speed, pole angle and speed), takes one of 2 actions (0 pushes
left, 1 pushes right), +1 each step, ends when the pole leans past 12 degrees
or the cart leaves [-2.4, 2.4], cut at 500 steps; one step is 0.02 s. Its one
setting is `max_episode_steps` (0 keeps 500). Each step replayed in Gymnasium
gives the same observation, reward and ending to 1e-5.

## The MuJoCo plugin's environments

| Environment | Observes | Actions | Episode | One step |
|---|---|---|---|---|
| `InvertedPendulum-v5` | 4 (cart and pole position and speed) | 1, force in [-3, 3] | 1000 steps; ends when the pole leans past 0.2 rad | 0.04 s |
| `HalfCheetah-v5` | 17 (joint positions without x, and speeds) | 6, torques in [-1, 1] | 1000 steps | 0.05 s |

They are Gymnasium's MuJoCo tasks (version 5): the same model files
(Gymnasium's, MIT licence, in the plugin's `assets` folder), observations,
rewards, endings and start-state noise, so returns compare with Gymnasium's.
Replayed in Gymnasium from the same start state with the same actions they
give the same observations and rewards to 1e-5.

Settings (every one is checked: a word where a number belongs, a value under
its minimum, a fraction where a whole number belongs, or a name the
environment does not have is an error, never a silent default):

| Setting | Environments | Default |
|---|---|---|
| `frame_skip` | both | 2 (pendulum), 5 (cheetah): model steps per environment step |
| `reset_noise_scale` | both | 0.01 (pendulum), 0.1 (cheetah) |
| `forward_reward_weight`, `ctrl_cost_weight` | `HalfCheetah-v5` | 1.0, 0.1 |
| `max_episode_steps` | both | 0 = the environment's own limit |
| `threads` | both | 0 = one per core, up to the number of copies |

One seed gives the same runs on any number of threads: copy `i` draws its
randomness from the seed and `i` only. A copy whose episode ends starts its
next episode in the same step. On this development PC (12 threads) a batch
steps about 185,000 pendulum or 44,000 to 62,000 cheetah environment steps a
second with 64 copies, and 64,000 or 21,000 with one copy.

## From Python (`pycyxwiz.rl`)

Without the Engine running, in any Python that has numpy and gymnasium 1.1 or
newer (its same-step autoreset), with
the folder holding `pycyxwiz` and `_cyxwiz_envs` (the Engine's `bin` folder)
on `sys.path`:

```python
from pycyxwiz import rl

rl.load_plugin("plugins/simulation/mujoco")   # the folder holding plugin.json
print([e["id"] for e in rl.environments()])

env = rl.Env("InvertedPendulum-v5")             # a gymnasium.Env
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

envs = rl.VectorEnv("HalfCheetah-v5", 16, ctrl_cost_weight=0.05)  # a gymnasium VectorEnv
```

- The plugin cuts an episode at its step limit itself: add no `TimeLimit`
  wrapper.
- `rl.Env`: after an episode ends, `reset()` without a seed gives the episode
  the plugin already started; `reset(seed=...)` starts every copy again.
- `rl.VectorEnv` resets an ended copy in the same step (Gymnasium's
  same-step autoreset): its last observation is in `info["final_obs"]`,
  marked in `info["_final_obs"]`.
- A plugin permission the Engine would ask you about is refused in this
  process, which cannot ask.

Stable-Baselines3 trains on it as on any Gymnasium environment:

```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", rl.Env("InvertedPendulum-v5"), seed=0)
model.learn(total_timesteps=100_000)
```

## Checks

- `cyxwiz-plugin-check <plugin folder>` creates two copies of each
  environment a plugin offers, resets them and steps them 20 times.
- `test_mujoco_environments` (settings, autoreset, the same runs on 1, 3 and
  4 threads, steps a second); `plugins/simulation/mujoco/tests/mujoco_gymnasium_parity.py`
  replays its trajectories in Gymnasium; `test_mujoco_python_envs.py` checks
  `pycyxwiz.rl` with Gymnasium's environment checker and runs SB3 PPO.
  `cyxwiz-engine/tests/test_classic_control_gymnasium.py` replays CartPole
  steps in Gymnasium. Configure with
  `-DCYXWIZ_RL_ORACLE_PYTHON=<python with gymnasium>` to register the Python
  checks in ctest.
