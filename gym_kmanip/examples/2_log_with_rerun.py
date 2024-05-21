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
env = gym.make(ENV_NAME, log_rerun=True, log_prefix="rerun_test")
env.reset()

print(f'Running the {ENV_NAME} environment for {k.MAX_EPISODE_STEPS} steps')
for _ in range(k.MAX_EPISODE_STEPS):
    action = env.action_space.sample()
    _, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

env.close()

"""
Look inside gym_kmanip/data for a .rrd file and run

> rerun gym_kmanip/data/rerun_test.foo/foo.rrd

this will open a GUI where you can visualize the run.
"""