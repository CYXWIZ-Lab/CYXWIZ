"""Replay the MuJoCo plugin's task trajectories in Gymnasium (the oracle).

    python mujoco_gymnasium_parity.py <test_mujoco_environments.exe> <work dir>

Runs the test with --trajectory, then for each task makes the Gymnasium env,
sets the same start state (qpos, qvel), applies the same actions and checks
that every observation, reward and termination matches. Needs gymnasium and
mujoco of the plugin's version (the Engine's python-embed has both).
"""

import json
import pathlib
import subprocess
import sys

import gymnasium as gym
import mujoco
import numpy as np

TOLERANCE = 1e-5  # the plugin reports float32; Gymnasium float64


def main() -> int:
    exe, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    work.mkdir(parents=True, exist_ok=True)
    trajectory = work / "mujoco_trajectories.json"
    subprocess.run([str(exe), "--trajectory", str(trajectory)], check=True, stdout=subprocess.DEVNULL)
    tasks = json.loads(trajectory.read_text(encoding="utf-8"))
    print(f"gymnasium {gym.__version__}, mujoco {mujoco.__version__}")
    failures = 0
    for task_id, run in tasks.items():
        env = gym.make(task_id).unwrapped
        env.reset(seed=0)
        env.set_state(np.array(run["qpos"]), np.array(run["qvel"]))
        worst = 0.0
        for index, step in enumerate(run["steps"]):
            obs, reward, terminated, _, _ = env.step(np.array(step["action"], dtype=np.float32))
            worst = max(worst, float(np.max(np.abs(obs - np.array(step["obs"])))), abs(reward - step["reward"]))
            if bool(terminated) != step["terminated"]:
                print(f"FAIL {task_id} step {index}: terminated {terminated}, plugin {step['terminated']}")
                failures += 1
                break
        steps = len(run["steps"])
        if worst > TOLERANCE:
            print(f"FAIL {task_id}: largest difference {worst:.3g} over {steps} steps")
            failures += 1
        else:
            print(f"ok   {task_id}: {steps} steps, largest difference {worst:.3g}")
        env.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
