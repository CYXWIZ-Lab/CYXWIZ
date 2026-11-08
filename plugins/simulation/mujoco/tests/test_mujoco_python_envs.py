"""The MuJoCo plugin's environments from Python (pycyxwiz.rl).

    python test_mujoco_python_envs.py <folder of pycyxwiz and _cyxwiz_envs> <plugin folder>

Loads the plugin through _cyxwiz_envs, checks rl.Env against Gymnasium's
own environment checker, the same start for the same seed, the step limit
and same-step autoreset in rl.VectorEnv, then trains Stable-Baselines3 PPO
for a few thousand steps on our InvertedPendulum (it must run, not learn).
Needs numpy, gymnasium and, for the PPO part, stable_baselines3 (the Engine's
python-embed has all of them).
"""

import os
import sys

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)
        print("FAIL:", what)


def main():
    folder, plugin = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    for d in (folder, r"C:\Program Files\ArrayFire\v3\lib"):
        if os.path.isdir(d) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(d)
    os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, folder)

    import numpy as np
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
    from pycyxwiz import rl

    plugin_id = rl.load_plugin(plugin)
    check(plugin_id == "com.cyxwiz.simulation.mujoco", "the MuJoCo plugin loads: " + plugin_id)
    ids = sorted(e["id"] for e in rl.environments() if e["plugin_id"] == plugin_id)
    check(ids == ["HalfCheetah-v5", "InvertedPendulum-v5"], f"its environments are listed: {ids}")
    check(any(e["id"] == "CartPole-v1" for e in rl.environments()), "the Engine's own CartPole is listed too")

    env = rl.Env("InvertedPendulum-v5")
    check(isinstance(env, gym.Env) and env.observation_space.shape == (4,) and env.action_space.shape == (1,),
          "rl.Env is a gymnasium.Env of the pendulum's spaces")
    check(float(env.action_space.high[0]) == 3.0, "the pendulum's force limit is 3")
    check_env(env, skip_render_check=True)  # raises on a broken contract
    a, _ = env.reset(seed=5)
    b, _ = env.reset(seed=5)
    c, _ = env.reset(seed=6)
    check(np.array_equal(a, b) and not np.array_equal(a, c), "one seed, one start; another seed, another start")

    try:
        rl.Env("HalfCheetah-v5", frame_skip="fast")
        check(False, "a bad setting is refused")
    except RuntimeError as error:
        check("Frame skip must be a number" in str(error), f"the plugin's reason comes through: {error}")
    try:
        rl.Env("Nope-v0")
        check(False, "an unknown environment is refused")
    except RuntimeError as error:
        check("No loaded plugin offers" in str(error), f"unknown environment: {error}")

    short = rl.Env("HalfCheetah-v5", max_episode_steps=3)
    short.reset(seed=1)
    ends = [short.step(short.action_space.sample())[3] for _ in range(3)]
    check(ends == [False, False, True], f"the plugin cuts the episode at its step limit: {ends}")
    following, _ = short.reset()
    check(following.shape == (17,), "reset after the end gives the episode the plugin already started")

    envs = rl.VectorEnv("InvertedPendulum-v5", 8, threads=2)
    obs, _ = envs.reset(seed=3)
    check(obs.shape == (8, 4), "8 copies observe 8 x 4")
    saw_end = False
    for _ in range(60):
        obs, rewards, terminated, truncated, info = envs.step(np.full((8, 1), 3.0, dtype=np.float32))
        if terminated.any():
            saw_end = True
            mask = info["_final_obs"]
            check(np.all(np.abs(info["final_obs"][mask][:, 1]) > 0.2), "final_obs holds the fallen poles")
            check(np.all(np.abs(obs[mask][:, 1]) <= 0.01), "the fallen copies start again in the same step")
            break
    check(saw_end, "pushed hard, the poles fall")
    envs.close()

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("stable_baselines3 not installed: PPO part skipped")
    else:
        model = PPO("MlpPolicy", rl.Env("InvertedPendulum-v5"), n_steps=512, batch_size=64, seed=0, verbose=0,
                    device="cpu")
        model.learn(total_timesteps=2048)
        check(model.num_timesteps >= 2048, "SB3 PPO trains on our pendulum")
        print(f"ok   SB3 PPO ran {model.num_timesteps} steps on InvertedPendulum-v5 (pycyxwiz.rl)")

    if FAILURES:
        print(f"test_mujoco_python_envs: {len(FAILURES)} failed")
        return 1
    print("test_mujoco_python_envs: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
