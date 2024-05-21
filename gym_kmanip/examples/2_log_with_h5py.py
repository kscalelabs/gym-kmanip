import gymnasium as gym
import numpy as np

import gym_kmanip as k

# choose your environment
# ENV_NAME: str = "KManipSoloArm"
# ENV_NAME: str = "KManipSoloArmQPos"
ENV_NAME: str = "KManipSoloArmVision"
# ENV_NAME: str = "KManipDualArm"
# ENV_NAME: str = "KManipDualArmVision"
# ENV_NAME: str = "KManipTorso"
# ENV_NAME: str = "KManipTorsoVision"
env = gym.make(ENV_NAME, log_h5py=True, log_prefix="h5py_test")
env.reset()

for _ in range(k.MAX_EPISODE_STEPS):
    action = env.action_space.sample()
    if "eer_pos" in action:
        action["eer_pos"] = env.unwrapped.mj_env.physics.data.body("cube").xpos
        action["eer_orn"] = np.array([1, 0, 0, 0])
    if "eel_pos" in action:
        action["eel_pos"] = env.unwrapped.mj_env.physics.data.body("cube").xpos
        action["eel_orn"] = np.array([1, 0, 0, 0])
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()

"""
Look inside gym_kmanip/data for a .rrd file and run

> rerun gym_kmanip/data/rerun_test.foo.rrd

this will open a GUI where you can visualize the run.
"""