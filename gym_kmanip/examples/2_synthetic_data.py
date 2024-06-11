"""
Run:

python gym_kmanip/examples/2_synthetic_data.py
python gym_kmanip/examples/5_upload_dataset_to_hf.py
"""
import os
import pprint

import h5py
import gymnasium as gym
import numpy as np
import imageio
import gym_kmanip as k

# choose your environment
# ENV_NAME: str = "gym_kmanip/KManipSoloArm"
# ENV_NAME: str = "gym_kmanip/KManipSoloArmQPos"
ENV_NAME: str = "gym_kmanip/KManipSoloArmVision"
# ENV_NAME: str = "gym_kmanip/KManipDualArm"
# ENV_NAME: str = "gym_kmanip/KManipDualArmQPos"
# ENV_NAME: str = "gym_kmanip/KManipDualArmVision"
# ENV_NAME: str = "gym_kmanip/KManipTorso"
# ENV_NAME: str = "gym_kmanip/KManipTorsoVision"

ENV = ENV_NAME.split("/")[1]
DISTANCE_TERIMNATE = 0.015
NUM_EPISODES: int = 100

env = gym.make(
    ENV_NAME,
    log_h5py=True,
    log_rerun=True,
    log_prefix="sim_synth",
)

for episode in range(NUM_EPISODES):
    frames = []
    env.reset()
    # heuristic action moving towards cube
    cube_pos = env.unwrapped.env.physics.data.qpos[-7:-4].copy()
    for _ in range(k.MAX_EPISODE_STEPS):
        # this is fake action
        action  = env.action_space.sample()

        image = env.render()
        frames.append(image)

        eer_pos = env.unwrapped.env.physics.data.site("eer_site_pos").xpos.copy()
        raw_action = cube_pos - eer_pos
        raw_action /= np.linalg.norm(raw_action)
        distance = np.linalg.norm(cube_pos - eer_pos)
   
        print(f"raw_action: {raw_action}")
        action["eer_pos"] = raw_action
        # breakpoint()
        _, _, terminated, truncated, _ = env.step(action)

        if distance < DISTANCE_TERIMNATE:
            terminated = True
        if terminated or truncated:
            break

    imageio.mimsave(f"viz_{ENV}_{episode}.mp4", np.stack(frames), fps=k.FPS)

env.close()
log_path = os.path.join(env.unwrapped.log_dir, "episode_1.hdf5")
print(f"Opening hdf5 file at \n\t{log_path}")
f = h5py.File(log_path, "r")
print("\nroot level keys:\n")
print(f.keys())
print("\nmetadata:\n")
pprint.pprint(dict(f['metadata'].attrs.items()))
print("\ndata:\n")
print(f['action'])
print(f['observations/qpos'])
print(f['observations/qvel'])
print(f['observations/images/head'])
print(f['observations/images/grip_r'])