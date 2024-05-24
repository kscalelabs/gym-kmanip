import math

from dm_control import viewer
import gymnasium as gym
import numpy as np

import gym_kmanip as k

# choose your environment
# ENV_NAME: str = "KManipSoloArm"
# ENV_NAME: str = "KManipSoloArmQPos"
# ENV_NAME: str = "KManipSoloArmVision"
ENV_NAME: str = "KManipDualArm"
# ENV_NAME: str = "KManipDualArmQPos"
# ENV_NAME: str = "KManipDualArmVision"
# ENV_NAME: str = "KManipTorso"
# ENV_NAME: str = "KManipTorsoVision"
env = gym.make(ENV_NAME)
env.reset()

# oscillate the action values between [-1, 1]
PERIOD = 10.0 # seconds

def policy(ts):
    action = env.action_space.sample()
    sim_time = env.unwrapped.env.physics.data.time
    val: float = math.sin(2 * math.pi * sim_time / PERIOD)
    for action_key, action_value in action.items():
        action[action_key] = np.ones_like(action_value) * val
    print(action)
    return action

viewer.launch(env.unwrapped.env, policy=policy)
