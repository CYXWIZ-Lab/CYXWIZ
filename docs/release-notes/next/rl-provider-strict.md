kind: changed
area: training
page: usage/rl_training.md

Reinforcement learning runs (PPO, SAC, TD3, DQN) now apply the execution policy to their GPU provider kernels too: under Strict a failing fused-network, joint Adam or PPO kernel stops the run, naming the provider and the operation, instead of quietly moving to ArrayFire operations. Every provider call of an RL run is counted in its trace (Studio Debugger, Runtime: provider groups), as in graph training.
