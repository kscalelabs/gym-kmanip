import os
import pprint

import h5py
import gymnasium as gym

import gym_kmanip as k

# choose your environment
# ENV_NAME: str = "KManipSoloArm"
# ENV_NAME: str = "KManipSoloArmQPos"
# ENV_NAME: str = "KManipSoloArmVision"
# ENV_NAME: str = "KManipDualArm"
ENV_NAME: str = "KManipDualArmVision"
# ENV_NAME: str = "KManipTorso"
# ENV_NAME: str = "KManipTorsoVision"
env = gym.make(ENV_NAME, log_h5py=True, log_prefix="h5py_test")
env.reset()

print(f'Running the {ENV_NAME} environment for {k.MAX_EPISODE_STEPS} steps')
for _ in range(k.MAX_EPISODE_STEPS):
    action = env.action_space.sample()
    _, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

env.close()

log_path = os.path.join(env.unwrapped.log_dir, "episode_1.hdf5")
print(f"Opening hdf5 file at \n\t{log_path}")
f = h5py.File(log_path, "r")
print("\nroot level keys:\n")
print(f.keys())
print("\nmetadata:\n")
pprint.pprint(dict(f['metadata'].attrs.items()))
print("\ndata:\n")
print(f['data/step/1/action/grip_r'][0])
print(f['data/step/1/state/camera/grip_r'][0])