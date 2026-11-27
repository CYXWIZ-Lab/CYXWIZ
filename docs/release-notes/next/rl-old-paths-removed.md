kind: removed
area: training
page: usage/rl_training.md

**Train RL and its generated Python script.** The graph trains with the Engine's own learners instead (PPO, SAC, TD3, DQN); Code Gen still writes an SB3 script from the graph. Also removed: the `pycyxwiz` functions `rl_should_stop`, `rl_is_paused`, `rl_set_stop` and `rl_set_paused`, which only that script used, and the MuJoCo plugin's **MuJoCoEnv** and **RLAgent** code-generation nodes. A saved graph with one of the old RL sketch nodes (Gym Environment, Replay Buffer, Policy Network, Value Network, RL Training) does not load and names what replaced it, or why it is gone; rebuild the graph from the **PPO** pattern.
