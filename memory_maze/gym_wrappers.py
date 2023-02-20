from typing import (
    Any,
    Tuple,
    TypeVar,
)

from typing import Any, Tuple
import numpy as np
ObsType = TypeVar("ObsType")

import dm_env
from dm_env import specs

try:
    import gymnasium as gym
    from gymnasium import spaces
except:
    import gym
    from gym import spaces


class GymWrapper(gym.Env):

    def __init__(self, env: dm_env.Environment):
        self.env = env
        self.action_space = _convert_to_space(env.action_spec())
        self.observation_space = _convert_to_space(env.observation_spec())
        self.img = None

    def reset(self, *, seed: int | None = None, 
        options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        ts = self.env.reset()
        # workaround for applications that use render
        self.img = ts.observation
        return ts.observation, dict()

    def step(self, action) -> Tuple[Any, float, bool, bool, dict]:
        ts = self.env.step(action)
        assert not ts.first(), "dm_env.step() caused reset, reward will be undefined."
        assert ts.reward is not None
        done = ts.last()
        terminal = ts.last() and ts.discount == 0.0
        info = {}
        truncation = False
        if done and not terminal:
            truncation = True
            info['TimeLimit.truncated'] = truncation  # acme.GymWrapper understands this and converts back to dm_env.truncation()
        # workaround for applications that use render
        self.img = ts.observation
        return ts.observation, ts.reward, done, truncation, info

    def render(self) -> Any | None:
        ''' workaround for applications that use render
            It returns an image collected during step or reset 
        '''
        return self.img


def _convert_to_space(spec: Any) -> gym.Space:
    # Inverse of acme.gym_wrappers._convert_to_spec

    if isinstance(spec, specs.DiscreteArray):
        return spaces.Discrete(spec.num_values)

    if isinstance(spec, specs.BoundedArray):
        return spaces.Box(
            shape=spec.shape,
            dtype=spec.dtype,
            low=spec.minimum.item() if len(spec.minimum.shape) == 0 else spec.minimum,
            high=spec.maximum.item() if len(spec.maximum.shape) == 0 else spec.maximum)
    
    if isinstance(spec, specs.Array):
        return spaces.Box(
            shape=spec.shape,
            dtype=spec.dtype,
            low=-np.inf,
            high=np.inf)

    if isinstance(spec, tuple):
        return spaces.Tuple(_convert_to_space(s) for s in spec)

    if isinstance(spec, dict):
        return spaces.Dict({key: _convert_to_space(value) for key, value in spec.items()})

    raise ValueError(f'Unexpected spec: {spec}')
