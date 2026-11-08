"""The Engine's built-in CartPole-v1 against Gymnasium's (the oracle).

    python test_classic_control_gymnasium.py <folder of pycyxwiz and _cyxwiz_envs>

Through pycyxwiz.rl: Gymnasium's environment checker; then over several
episodes of random pushes, each step is replayed in Gymnasium's CartPole
from the observation before it, and the next observation, reward and ending
must match. Needs numpy and gymnasium.
"""

import os
import sys

TOLERANCE = 1e-5  # observations are float32; Gymnasium steps in float64


def main():
    folder = os.path.abspath(sys.argv[1])
    for d in (folder, r"C:\Program Files\ArrayFire\v3\lib"):
        if os.path.isdir(d) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(d)
    os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, folder)

    import numpy as np
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
    from pycyxwiz import rl

    ours = rl.Env("CartPole-v1")
    check_env(ours, skip_render_check=True)
    oracle = gym.make("CartPole-v1").unwrapped
    oracle.reset(seed=0)
    rng = np.random.default_rng(4)
    obs, _ = ours.reset(seed=4)
    worst, steps, episodes, failures = 0.0, 0, 0, 0
    lengths = []
    length = 0
    while episodes < 20:
        action = int(rng.integers(0, 2))
        oracle.state = np.array(obs, dtype=np.float64)
        oracle.steps_beyond_terminated = None
        want, want_reward, want_end, _, _ = oracle.step(action)
        obs, reward, terminated, truncated, _ = ours.step(action)
        worst = max(worst, float(np.max(np.abs(obs - want))), abs(reward - want_reward))
        steps += 1
        length += 1
        if terminated != want_end:
            print(f"FAIL step {steps}: terminated {terminated}, Gymnasium {want_end}")
            failures += 1
        if terminated or truncated:
            episodes += 1
            lengths.append(length)
            length = 0
            obs, _ = ours.reset()
    if worst > TOLERANCE:
        print(f"FAIL largest difference {worst:.3g}")
        failures += 1
    print(f"{'FAIL' if failures else 'ok  '} CartPole-v1: {steps} steps over {episodes} episodes "
          f"(lengths {min(lengths)}-{max(lengths)}), largest difference {worst:.3g}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
