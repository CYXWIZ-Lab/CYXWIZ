# Reinforcement learning graphs

Policies the Engine trains itself with PPO (see docs/usage/rl_training.md).

| File | Environment | Needs | Stops at |
|---|---|---|---|
| `ppo_cartpole.cyxgraph` | CartPole-v1 (built in), 8 copies | nothing | mean return 475 of the last 100 episodes |
| `ppo_inverted_pendulum.cyxgraph` | InvertedPendulum-v5, 4 copies | the MuJoCo Simulation plugin loaded | mean return 950 |
| `ppo_half_cheetah.cyxgraph` | HalfCheetah-v5, 1 copy, two layers of 256 with ReLU | the MuJoCo Simulation plugin loaded | runs its 1,000,000 steps (about 10-13 minutes on an NVIDIA GPU with the CUDA GPU pack, over an hour on the CPU); final mean return about 3,800-5,100 |

Each is the PPO (MLP) pattern with the PPO node's "Tuned for" settings filled
in (64-64 Tanh; HalfCheetah 256-256 ReLU, as RL Zoo trains it): open one (File > Open), then Train (F5). The RL Training
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
