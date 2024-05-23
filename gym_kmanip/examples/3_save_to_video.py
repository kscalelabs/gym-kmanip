import imageio
import gymnasium as gym
import numpy as np

import gym_kmanip as k

# choose your environment
# ENV_NAME: str = "KManipSoloArm"
# ENV_NAME: str = "KManipSoloArmQPos"
# ENV_NAME: str = "KManipSoloArmVision"
# ENV_NAME: str = "KManipDualArm"
# ENV_NAME: str = "KManipDualArmQPos"
ENV_NAME: str = "KManipDualArmVision"
# ENV_NAME: str = "KManipTorso"
# ENV_NAME: str = "KManipTorsoVision"
env = gym.make(ENV_NAME)
env.reset()
frames = []

for _ in range(k.MAX_EPISODE_STEPS):
    action = env.action_space.sample()
    _, _, terminated, truncated, _ = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        env.reset()

env.close()
imageio.mimsave(f"viz_{ENV_NAME}.mp4", np.stack(frames), fps=k.FPS)
