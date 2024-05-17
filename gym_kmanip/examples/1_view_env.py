from dm_control import viewer
import gymnasium as gym

import gym_kmanip

# env_name = "KManipSoloArm"
# env_name = "KManipSoloArmVision"
# env_name = "KManipDualArm"
# env_name = "KManipDualArmVision"
env_name = "KManipTorso"
# env_name = "KManipTorsoVision"
env = gym.make(env_name)
action_spec = env.unwrapped.mj_env.action_spec()


def random_policy(_):
    return env.action_space.sample()


"""
F1             Help
F2             Info
F5             Stereo
F6             Frame
F7             Label
--------------
Space          Pause
BackSpace      Reset
Ctrl A         Autoscale
0 - 4          Geoms
Shift 0 - 4    Sites
Speed Up       =
Slow Down      -
Switch Cam     [ ]
--------------
R drag                  Translate
L drag                  Rotate
Scroll                  Zoom
L dblclick              Select
R dblclick              Center
Ctrl R dblclick / Esc   Track
Ctrl [Shift] L/R drag   Perturb

"""

viewer.launch(env.unwrapped.mj_env, policy=random_policy)
