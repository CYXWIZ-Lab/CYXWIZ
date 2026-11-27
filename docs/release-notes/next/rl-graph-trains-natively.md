kind: added
area: training
page: usage/rl_training.md

**Reinforcement learning policies train from a graph on the canvas, in the Engine.** An **Environment** node, ordinary layer nodes ending in a **Policy Head** and a **Value Head**, and the **PPO** node make a graph the Engine compiles and trains itself: no Python and no Stable-Baselines3 needed (the learners are checked against SB3 and PyTorch). Start from the **PPO (MLP 64-64)** pattern. Train runs it on a worker; the **RL Training Dashboard** shows progress, speed, return curves by environment step, entropy, KL, clip fraction and action spread, with Pause, Resume and Stop, and draws the recorded SB3 run of the same settings under your curve. The trained policy is saved as a `.cyxmodel` and played back in the viewport, and Code Gen writes the graph as an SB3 script on `pycyxwiz.rl`. The SAC, TD3 and DQN learners train from the same kind of graph. Examples: `examples/cyxgraph/rl`.
