"""Plugin environments for reinforcement learning as Gymnasium environments: load_plugin(folder), environments(), Env(id, **settings) (a gymnasium.Env) and VectorEnv(id, num_envs, **settings) (a gymnasium VectorEnv with same-step autoreset). Stable-Baselines3, CleanRL or your own code then train on exactly the physics the Engine trains on. Needs numpy, gymnasium and the _cyxwiz_envs module built with the Engine; see docs/usage/rl_environments.md."""

# Compiled into pycyxwiz and installed as the submodule pycyxwiz.rl when
# pycyxwiz is imported (TOFIX195 S3.4).

import importlib as _importlib

__all__ = ["load_plugin", "environments", "Env", "VectorEnv"]

_classes = {}


def _native():
    try:
        return _importlib.import_module("_cyxwiz_envs")
    except ImportError as error:
        raise ImportError(
            "pycyxwiz.rl needs the _cyxwiz_envs module, which is built with the Engine and sits next to "
            "pycyxwiz. Add that folder to sys.path.") from error


def load_plugin(folder):
    """Loads and starts the plugin in `folder` (the one holding plugin.json); returns its id."""
    return _native().load_plugin(str(folder))


def environments():
    """The environments the loaded plugins offer: one dict each (key, id, sizes, action range, settings)."""
    return _native().environments()


def _settings(settings, extra):
    merged = dict(settings or {})
    merged.update(extra)
    return {str(key): str(value) for key, value in merged.items()}


def _spaces(info):
    import numpy as np
    from gymnasium import spaces
    observation = spaces.Box(-np.inf, np.inf, shape=(info["observation_size"],), dtype=np.float32)
    if info["discrete_actions"]:
        action = spaces.Discrete(info["action_count"])
    else:
        action = spaces.Box(np.array(info["action_low"], dtype=np.float32),
                            np.array(info["action_high"], dtype=np.float32), dtype=np.float32)
    return observation, action


def _make_classes():
    import gymnasium as gym
    import numpy as np
    from gymnasium.vector import AutoresetMode, VectorEnv as _GymVectorEnv
    from gymnasium.vector.utils import batch_space

    class Env(gym.Env):
        """One copy of a plugin environment as a gymnasium.Env."""

        metadata = {"render_modes": []}

        def __init__(self, environment, settings=None, **kwargs):
            native = _native()
            self._batch = native.Batch(environment, 1, _settings(settings, kwargs))
            self.environment_info = self._batch.info
            self.observation_space, self.action_space = _spaces(self.environment_info)
            self._discrete = self.environment_info["discrete_actions"]
            self._next_obs = None  # the episode the plugin started when the last one ended

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if seed is None and self._next_obs is not None:
                obs, self._next_obs = self._next_obs, None
                return obs, {}
            if seed is None:
                seed = int(self.np_random.integers(0, 2**63 - 1))
            self._next_obs = None
            return self._batch.reset(seed)[0], {}

        def step(self, action):
            actions = np.asarray(action, dtype=np.float32).reshape(1, -1)
            obs, rewards, terminated, truncated, final_obs = self._batch.step(actions)
            ended = bool(terminated[0] or truncated[0])
            if ended:
                self._next_obs = obs[0]
            return (final_obs[0] if ended else obs[0]), float(rewards[0]), bool(terminated[0]), bool(truncated[0]), {}

        def close(self):
            self._batch.close()

    class VectorEnv(_GymVectorEnv):
        """`num_envs` copies of a plugin environment stepped together (same-step autoreset:
        an ended copy's last observation is info["final_obs"], masked by info["_final_obs"])."""

        metadata = {"render_modes": [], "autoreset_mode": AutoresetMode.SAME_STEP}

        def __init__(self, environment, num_envs, settings=None, **kwargs):
            native = _native()
            self._batch = native.Batch(environment, int(num_envs), _settings(settings, kwargs))
            self.environment_info = self._batch.info
            self.num_envs = int(num_envs)
            self.single_observation_space, self.single_action_space = _spaces(self.environment_info)
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
            self.action_space = batch_space(self.single_action_space, self.num_envs)
            self._np_random_seed = None

        def reset(self, *, seed=None, options=None):
            if seed is None:
                seed = int(np.random.default_rng().integers(0, 2**63 - 1))
            if isinstance(seed, (list, tuple)):
                seed = seed[0]
            self._np_random_seed = seed
            return self._batch.reset(int(seed)), {}

        def step(self, actions):
            batch = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, -1)
            obs, rewards, terminated, truncated, final_obs = self._batch.step(batch)
            ended = terminated | truncated
            info = {"final_obs": final_obs, "_final_obs": ended} if ended.any() else {}
            return obs, rewards, terminated, truncated, info

        def close_extras(self, **kwargs):
            self._batch.close()

    return {"Env": Env, "VectorEnv": VectorEnv}


def __getattr__(name):
    # Env and VectorEnv need gymnasium: made on first use, so importing
    # pycyxwiz does not need it.
    if name in ("Env", "VectorEnv"):
        if not _classes:
            _classes.update(_make_classes())
        return _classes[name]
    raise AttributeError(f"module 'pycyxwiz.rl' has no attribute '{name}'")
