# Reinforcement learning graphs

Policies the Engine trains itself with PPO, SAC, TD3 or DQN (see docs/usage/rl_training.md).

| File | Environment | Needs | Stops at |
|---|---|---|---|
| `ppo_cartpole.cyxgraph` | CartPole-v1 (built in), 8 copies | nothing | mean return 475 of the last 100 episodes |
| `ppo_inverted_pendulum.cyxgraph` | InvertedPendulum-v5, 4 copies | the Simulation (MuJoCo) plugin loaded | mean return 950 |
| `ppo_half_cheetah.cyxgraph` | HalfCheetah-v5, 1 copy, two layers of 256 with ReLU | the Simulation (MuJoCo) plugin loaded | runs its 1,000,000 steps (about 8-13 minutes on an NVIDIA GPU with the CUDA GPU pack, over an hour on the CPU); final mean return about 3,800-5,100 |
| `sac_inverted_pendulum.cyxgraph` | InvertedPendulum-v5 with SAC, 1 copy, two layers of 256 with ReLU, the SAC node's defaults | the Simulation (MuJoCo) plugin loaded | mean return 950 (about 100,000 steps, 12-16 minutes on an NVIDIA GPU with the CUDA GPU pack) |
| `sac_half_cheetah.cyxgraph` | HalfCheetah-v5 with SAC, 1 copy, two layers of 256 with ReLU, RL Zoo's settings | the Simulation (MuJoCo) plugin loaded; an NVIDIA GPU with the CUDA GPU pack advised | runs its 1,000,000 steps (about 2-3 hours on a GTX 1050 Ti); final mean return about 8,300-11,100, twice PPO's |
| `td3_inverted_pendulum.cyxgraph` | InvertedPendulum-v5 with TD3, 1 copy, layers of 400 and 300 with ReLU, the TD3 node's defaults (Gaussian noise 0.1) | the Simulation (MuJoCo) plugin loaded | mean return 950 (101,000-121,000 steps, 13-18 minutes on an NVIDIA GPU with the CUDA GPU pack) |
| `td3_half_cheetah.cyxgraph` | HalfCheetah-v5 with TD3, 1 copy, layers of 400 and 300 with ReLU, RL Zoo's settings | the Simulation (MuJoCo) plugin loaded; an NVIDIA GPU with the CUDA GPU pack advised | runs its 1,000,000 steps (about 2 hours 20 minutes on a GTX 1050 Ti); final mean return about 8,900-9,800 |
| `dqn_cartpole.cyxgraph` | CartPole-v1 (built in) with DQN, 1 copy, Q layers of 256 and 256 with ReLU, RL Zoo's settings | nothing | runs its 50,000 steps (about 1-2 minutes on an NVIDIA GPU with the CUDA GPU pack); final mean return about 130-250, as Stable-Baselines3's with the same settings |
| `ppo_half_cheetah_robust.cyxgraph` | HalfCheetah-v5 for rough ground, slopes, steps and obstacles: a curriculum (mark 1,500), pushes every 3 s, friction ±20 % and mass ±10 %, Look ahead (27 values observed), 2,000,000 steps | the Simulation (MuJoCo) plugin loaded | the dashboard's Difficulty climbs as the return passes the mark; compare its saved policy with the flat one in Policy Playback > Stress test |

Each `ppo_*` is the PPO (MLP) pattern with the PPO node's "Tuned for" settings filled
in (64-64 Tanh; HalfCheetah 256-256 ReLU, as RL Zoo trains it); each `sac_*` is
the SAC (MLP 256-256) pattern, HalfCheetah with the SAC node's "Tuned for"
settings; each `td3_*` the TD3 (MLP 400-300) pattern, HalfCheetah with the
TD3 node's "Tuned for" settings; `dqn_cartpole` the DQN (MLP 64-64) pattern
widened to RL Zoo's 256-256 with the DQN node's "Tuned for" settings. Open
one (File > Open), then Train (F5). The RL Training
Dashboard shows the episode return by environment step, with the
Stable-Baselines3 reference of the same settings under it, and the policy
diagnostics.

When the run finishes:
- **Play policy** on the dashboard plays the trained policy. CartPole is drawn
  in the Policy Playback panel; InvertedPendulum and HalfCheetah show in the
  MuJoCo Viewport.
- **Save policy...** writes it as a `.cyxmodel`.
- **File > Open Model File...** on that file plays it again later.

The same graphs as a Python script (Stable-Baselines3 on the Engine's own
environments through pycyxwiz.rl): Code Gen mode, Generate.
