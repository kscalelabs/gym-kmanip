import imageio
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
imageio.mimsave(f"viz_{env_name}.mp4", np.stack(frames), fps=k.FPS)