import math
from dm_control import viewer
import gymnasium as gym
import numpy as np
import gym_kmanip as k
from scipy.spatial.transform import Rotation as R

# choose your environment
ENV_NAME: str = "KManipSoloArm"
# ENV_NAME: str = "KManipSoloArmQPos"
# ENV_NAME: str = "KManipSoloArmVision"
# ENV_NAME: str = "KManipDualArm"
# ENV_NAME: str = "KManipDualArmVision"
# ENV_NAME: str = "KManipTorso"
# ENV_NAME: str = "KManipTorsoVision"
env = gym.make(ENV_NAME)
env.reset()
# start pos for reach targets
pos_r = env.unwrapped.env.physics.data.mocap_pos[k.MOCAP_ID_R].copy()
if "Solo" not in ENV_NAME:
    pos_l = env.unwrapped.env.physics.data.mocap_pos[k.MOCAP_ID_L].copy()

# TODO: try larger amplitudes, get a sense of the arm range
AMPLITUDE_X = 0.08
AMPLITUDE_Y = 0.08
AMPLITUDE_Z = 0.04
PERIOD = 0.1
X_ANGLE_AMPLITUDE = 0.4  # radians
Y_ANGLE_AMPLITUDE = 0.4  # radians
Z_ANGLE_AMPLITUDE = 0.6  # radians

def policy(ts):
    action = env.action_space.sample()
    sim_time = env.unwrapped.env.physics.data.time
    if "eer_pos" in action:
        action["eer_pos"] = np.array(
            [
                pos_r[0] + AMPLITUDE_X * math.sin(sim_time * PERIOD),
                pos_r[1] + AMPLITUDE_Y * math.cos(sim_time * PERIOD),
                pos_r[2] + AMPLITUDE_Z * math.sin(sim_time * PERIOD),
            ]
        )
        # Oscillating quaternion
        angle_x = X_ANGLE_AMPLITUDE * math.sin(sim_time * PERIOD)
        angle_y = Y_ANGLE_AMPLITUDE * math.cos(sim_time * PERIOD)
        angle_z = Z_ANGLE_AMPLITUDE * math.sin(sim_time * PERIOD)
        rotation_quat = R.from_euler('xyz', [angle_x, angle_y, angle_z]).as_quat()
        action["eer_orn"] = rotation_quat
        action["grip_r"] = 0.0 #math.sin(sim_time * PERIOD)
    if "eel_pos" in action:
        action["eel_pos"] = np.array(
            [
                pos_l[0] - AMPLITUDE_X * math.sin(sim_time * PERIOD),
                pos_l[1] + AMPLITUDE_Y * math.cos(sim_time * PERIOD),
                pos_l[2] + AMPLITUDE_Z * math.sin(sim_time * PERIOD),
            ]
        )
        # Oscillating quaternion
        angle_x = X_ANGLE_AMPLITUDE * math.sin(sim_time * PERIOD)
        angle_y = Y_ANGLE_AMPLITUDE * math.cos(sim_time * PERIOD)
        angle_z = Z_ANGLE_AMPLITUDE * math.sin(sim_time * PERIOD)
        rotation_quat = R.from_euler('xyz', [angle_x, angle_y, angle_z]).as_quat()
        action["eel_orn"] = rotation_quat
        action["grip_l"] = 0.0 #math.sin(sim_time * PERIOD)

    return action

viewer.launch(env.unwrapped.env, policy=policy)
