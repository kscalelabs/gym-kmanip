from collections import OrderedDict as ODict
from typing import Any, Callable, Dict, List, OrderedDict

import numpy as np
from numpy.typing import NDArray

import gym_kmanip as k
from gym_kmanip.env_base import KManipEnv


class KManipEnvReal:

    def __init__(self):
        pass

    def k_render(self, cam: k.Cam) -> NDArray:
        return np.empty((cam.h, cam.w, cam.c), dtype=np.uint8)

    def k_reset(self):
        observation = {}
        terminated: bool = False
        timestamp: float = 0.0
        discount = 0.0
        reward = 0.0
        return terminated, reward, discount, observation, timestamp

    def k_step(self, action: Dict[str, Any]):
        observation = {}
        terminated: bool = False
        timestamp: float = 0.0
        discount = 0.0
        reward = 0.0
        return terminated, reward, discount, observation, timestamp


def new(gym_env: KManipEnv) -> KManipEnvReal:
    return KManipEnvReal()
