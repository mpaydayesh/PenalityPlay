import os
import sys
import types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import pytest

# Provide minimal numpy stub if numpy isn't installed
try:
    import numpy as np  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - executed only without numpy
    np = types.ModuleType("numpy")

    def array(obj, dtype=None):
        if isinstance(obj, (list, tuple)):
            return list(obj)
        return [obj]

    def asarray(obj, dtype=None):
        return array(obj, dtype=dtype)

    class Random:
        def uniform(self, low, high):
            return random.uniform(low, high)

    np.array = array
    np.asarray = asarray
    np.float32 = float
    np.ndarray = list
    np.random = Random()
    sys.modules["numpy"] = np

# Provide a minimal gymnasium stub if the package isn't installed
try:
    import gymnasium  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - executed only without gymnasium
    gymnasium = types.ModuleType("gymnasium")

    class Box:
        def __init__(self, low, high, dtype=float):
            self.low = list(low)
            self.high = list(high)
            self.dtype = dtype
            self.shape = (len(self.low),)

        def sample(self):
            return [random.uniform(l, h) for l, h in zip(self.low, self.high)]

        def contains(self, x):
            return (
                len(x) == len(self.low)
                and all(l <= v <= h for v, l, h in zip(x, self.low, self.high))
            )

    class Dict:
        def __init__(self, spaces_dict):
            self.spaces = spaces_dict

        def sample(self):
            return {k: space.sample() for k, space in self.spaces.items()}

        def contains(self, x):
            return set(x.keys()) == set(self.spaces.keys()) and all(
                self.spaces[k].contains(x[k]) for k in self.spaces
            )

    spaces = types.SimpleNamespace(Box=Box, Dict=Dict)
    gymnasium.Env = object
    gymnasium.spaces = spaces
    sys.modules["gymnasium"] = gymnasium
    sys.modules["gymnasium.spaces"] = spaces

from src.envs import PenaltyShootoutEnv


@pytest.fixture
def env():
    return PenaltyShootoutEnv()


def test_env_reset_returns_valid_observation(env):
    obs = env.reset()
    assert env.observation_space.contains(obs)


def test_env_step_returns_expected_tuple(env):
    env.reset()
    actions = env.action_space.sample()
    obs, rewards, done, info = env.step(actions)

    assert env.observation_space.contains(obs)
    assert isinstance(rewards, dict)
    assert set(rewards.keys()) == {"striker", "goalkeeper"}
    assert isinstance(done, bool)
    assert "goal_scored" in info
