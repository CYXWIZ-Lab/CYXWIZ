"""Report reinforcement-learning metrics to the RL Training Dashboard.

Run in the Script Editor (F5). The RL Training Dashboard opens and shows
the episodes. A real training loop reports the same way (the stable-
baselines3 script that Train RL generates does).
"""
import math

import pycyxwiz

for episode in range(60):
    reward = 50.0 * (1 - math.exp(-episode / 15.0)) + 3.0 * math.sin(episode)
    pycyxwiz.rl_update_metric("episode_reward", reward)
    pycyxwiz.rl_update_metric("episode_length", 100.0 + 2.0 * episode)
    if episode % 5 == 0:  # once per policy update
        pycyxwiz.rl_update_metric("policy_loss", 0.5 * math.exp(-episode / 20.0))
        pycyxwiz.rl_update_metric("value_loss", 2.0 * math.exp(-episode / 25.0))
        pycyxwiz.rl_update_metric("explained_variance", 1 - math.exp(-episode / 30.0))

print("reported 60 episodes")
