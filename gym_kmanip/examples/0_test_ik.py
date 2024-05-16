import math

from dm_control import viewer
import gymnasium as gym
import numpy as np

import gym_kmanip as k

# env_name = "KManipSoloArm"
# env_name = "KManipSoloArmVision"
env_name = "KManipDualArm"
# env_name = "KManipDualArmVision"
# env_name = "KManipTorso"
# env_name = "KManipTorsoVision"
env = gym.make(env_name)

# start pos for reach targets
pos_r = env.unwrapped.mj_env.physics.data.mocap_pos[k.MOCAP_ID_R].copy()
if "Solo" not in env_name:
    pos_l = env.unwrapped.mj_env.physics.data.mocap_pos[k.MOCAP_ID_L].copy()

# TODO: try larger amplitudes, get a sense of the arm range
AMPLITUDE_X: float = 0.08
AMPLITUDE_Y: float = 0.08
AMPLITUDE_Z: float = 0.04
PERIOD: float = 0.5 * math.pi


def policy(ts):
    action = env.action_space.sample()
    sim_time = env.unwrapped.mj_env.physics.data.time
    if "eer_pos" in action:
        action["eer_pos"] = np.array(
            [
                pos_r[0] + AMPLITUDE_X * math.sin(sim_time * PERIOD),
                pos_r[1] + AMPLITUDE_Y * math.cos(sim_time * PERIOD),
                pos_r[2] + AMPLITUDE_Z * math.sin(sim_time * PERIOD),
            ]
        )
        action["eer_orn"] = np.array([1, 0, 0, 0])
        action["grip_r"] = math.sin(sim_time * PERIOD)
    if "eel_pos" in action:
        action["eel_pos"] = np.array(
            [
                pos_l[0] - AMPLITUDE_X * math.sin(sim_time * PERIOD),
                pos_l[1] + AMPLITUDE_Y * math.cos(sim_time * PERIOD),
                pos_l[2] + AMPLITUDE_Z * math.sin(sim_time * PERIOD),
            ]
        )
        action["eel_orn"] = np.array([1, 0, 0, 0])
        action["grip_l"] = math.sin(sim_time * PERIOD)
    return action


viewer.launch(env.unwrapped.mj_env, policy=policy)
