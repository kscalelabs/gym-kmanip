import imageio
import gymnasium as gym
import numpy as np

import gym_kmanip as k

# choose your environment
# ENV_NAME: str = "KManipSoloArm"
# ENV_NAME: str = "KManipSoloArmVision"
# ENV_NAME: str = "KManipDualArm"
ENV_NAME: str = "KManipDualArmVision"
# ENV_NAME: str = "KManipTorso"
# ENV_NAME: str = "KManipTorsoVision"
env = gym.make(ENV_NAME)
observation, info = env.reset()
frames = []

for _ in range(k.MAX_EPISODE_STEPS):
    action = env.action_space.sample()
    if "eer_pos" in action:
        action["eer_pos"] = env.unwrapped.mj_env.physics.data.body("cube").xpos
        action["eer_orn"] = np.array([1, 0, 0, 0])
    if "eel_pos" in action:
        action["eel_pos"] = env.unwrapped.mj_env.physics.data.body("cube").xpos
        action["eel_orn"] = np.array([1, 0, 0, 0])
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave(f"viz_{ENV_NAME}.mp4", np.stack(frames), fps=k.FPS)