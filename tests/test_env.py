import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import pytest

import gym_kmanip


@pytest.mark.parametrize(
    "env_name",
    [
        "KManipSoloArm",
        "KManipSoloArmQPos",
        "KManipSoloArmVision",
        "KManipDualArm",
        "KManipDualArmQPos",
        "KManipDualArmVision",
        "KManipTorso",
        "KManipTorsoVision",
    ],
)
def test_env(env_name):
    env = gym.make(env_name)
    check_env(env.unwrapped)
    env.close()