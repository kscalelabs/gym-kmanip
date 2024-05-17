import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import rerun as rr

import gym_kmanip as k

# choose your environment
# ENV_NAME: str = "KManipSoloArm"
# ENV_NAME: str = "KManipSoloArmVision"
# ENV_NAME: str = "KManipDualArm"
# ENV_NAME: str = "KManipDualArmVision"
ENV_NAME: str = "KManipTorso"
# ENV_NAME: str = "KManipTorsoVision"
env = gym.make(ENV_NAME)
env.reset()


DATASET_DIR: str = os.path.join(os.path.dirname(__file__), "data")
DATASET_NAME: str = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DATASET_OUTPUT_PATH = os.path.join(DATASET_DIR, DATASET_NAME)
rr.init(DATASET_OUTPUT_PATH)
rr.log("meta/env_name", ENV_NAME)

for _ in range(k.MAX_EPISODE_STEPS):
    action = env.action_space.sample()
    if "eer_pos" in action:
        action["eer_pos"] = env.unwrapped.mj_env.physics.data.body("cube").xpos
        action["eer_orn"] = np.array([1, 0, 0, 0])
        rr.log("data/eer_pose", rr.Transform3D(
            position=action["eer_pos"],
            orientation=action["eer_orn"],
        ))
    if "eel_pos" in action:
        action["eel_pos"] = env.unwrapped.mj_env.physics.data.body("cube").xpos
        action["eel_orn"] = np.array([1, 0, 0, 0])
        rr.log("data/eer_pose", rr.Transform3D(
            position=action["eel_pos"],
            orientation=action["eel_orn"],
        ))
    observation, reward, terminated, truncated, info = env.step(action)
    if "q_pos" in observation:
        rr.log("data/q_pos", observation["q_pos"])
    if "cam_head" in observation:
        rr.log("data/cam_head", observation["cam_head"])
    rr.log("data/reward", reward)
    rr.log("data/terminated", terminated)
    if terminated or truncated:
        observation, info = env.reset()

env.close()