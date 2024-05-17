from collections import OrderedDict
import os
from typing import List, Tuple

from gymnasium.envs.registration import register

import numpy as np
from numpy.typing import NDArray

MAX_EPISODE_STEPS: int = 300
ASSETS_DIR: str = os.path.join(os.path.dirname(__file__), "assets")
FPS: int = 30
CONTROL_TIMESTEP: float = 0.02  # ms

Q_SOLO_ARM_HOME_DICT: OrderedDict[str, float] = OrderedDict()
Q_SOLO_ARM_HOME_DICT["joint_right_arm_1_x8_1_dof_x8"] = 0.0
Q_SOLO_ARM_HOME_DICT["joint_right_arm_1_x8_2_dof_x8"] = 0.75
Q_SOLO_ARM_HOME_DICT["joint_right_arm_1_x6_1_dof_x6"] = 1.0
Q_SOLO_ARM_HOME_DICT["joint_right_arm_1_x6_2_dof_x6"] = 1.0
Q_SOLO_ARM_HOME_DICT["joint_right_arm_1_x4_1_dof_x4"] = 2.0
Q_SOLO_ARM_HOME_DICT["joint_right_arm_1_hand_right_1_x4_3_dof_x4"] = -2.0
Q_SOLO_ARM_HOME_DICT["joint_right_arm_1_hand_right_1_x4_1_dof_x4"] = 0.0
Q_SOLO_ARM_HOME_DICT["joint_right_arm_1_hand_right_1_x4_2_dof_x4"] = 0.0
Q_SOLO_ARM_HOME_DICT["joint_right_arm_1_hand_right_1_slider_3"] = 0.005
Q_SOLO_ARM_HOME_DICT["joint_right_arm_1_hand_right_1_slider_1"] = 0.005
Q_SOLO_ARM_HOME: NDArray = np.array(
    [v for v in Q_SOLO_ARM_HOME_DICT.values()],
    dtype=np.float32,
)
Q_SOLO_ARM_KEYS: List[str] = list(Q_SOLO_ARM_HOME_DICT.keys())

Q_DUAL_ARM_HOME_DICT: OrderedDict[str, float] = OrderedDict()
Q_DUAL_ARM_HOME_DICT["joint_right_arm_1_x8_1_dof_x8"] = 0.0
Q_DUAL_ARM_HOME_DICT["joint_right_arm_1_x8_2_dof_x8"] = 0.75
Q_DUAL_ARM_HOME_DICT["joint_right_arm_1_x6_1_dof_x6"] = 1.0
Q_DUAL_ARM_HOME_DICT["joint_right_arm_1_x6_2_dof_x6"] = 1.0
Q_DUAL_ARM_HOME_DICT["joint_right_arm_1_x4_1_dof_x4"] = 2.0
Q_DUAL_ARM_HOME_DICT["joint_right_arm_1_hand_right_1_x4_3_dof_x4"] = -2.7
Q_DUAL_ARM_HOME_DICT["joint_right_arm_1_hand_right_1_x4_1_dof_x4"] = 0.0
Q_DUAL_ARM_HOME_DICT["joint_right_arm_1_hand_right_1_x4_2_dof_x4"] = 0.0
Q_DUAL_ARM_HOME_DICT["joint_right_arm_1_hand_right_1_slider_3"] = 0.005
Q_DUAL_ARM_HOME_DICT["joint_right_arm_1_hand_right_1_slider_1"] = 0.005
Q_DUAL_ARM_HOME_DICT["joint_left_arm_1_x8_1_dof_x8"] = 0.0
Q_DUAL_ARM_HOME_DICT["joint_left_arm_1_x8_2_dof_x8"] = -0.75
Q_DUAL_ARM_HOME_DICT["joint_left_arm_1_x6_1_dof_x6"] = -1.0
Q_DUAL_ARM_HOME_DICT["joint_left_arm_1_x6_2_dof_x6"] = -1.0
Q_DUAL_ARM_HOME_DICT["joint_left_arm_1_x4_1_dof_x4"] = 2.0
Q_DUAL_ARM_HOME_DICT["joint_left_arm_1_hand_left_1_x4_3_dof_x4"] = 0.0
Q_DUAL_ARM_HOME_DICT["joint_left_arm_1_hand_left_1_x4_1_dof_x4"] = 0.0
Q_DUAL_ARM_HOME_DICT["joint_left_arm_1_hand_left_1_x4_2_dof_x4"] = 0.0
Q_DUAL_ARM_HOME_DICT["joint_left_arm_1_hand_left_1_slider_3"] = 0.005
Q_DUAL_ARM_HOME_DICT["joint_left_arm_1_hand_left_1_slider_1"] = 0.005
Q_DUAL_ARM_HOME: NDArray = np.array(
    [v for v in Q_DUAL_ARM_HOME_DICT.values()],
    dtype=np.float32,
)
Q_DUAL_ARM_KEYS: List[str] = list(Q_DUAL_ARM_HOME_DICT.keys())

Q_FULL_BODY_HOME_DICT: OrderedDict[str, float] = OrderedDict()
Q_FULL_BODY_HOME_DICT["joint_head_1_x4_1_dof_x4"] = -1.0
Q_FULL_BODY_HOME_DICT["joint_head_1_x4_2_dof_x4"] = 0.0
Q_FULL_BODY_HOME_DICT["joint_right_arm_1_x8_1_dof_x8"] = 1.7
Q_FULL_BODY_HOME_DICT["joint_right_arm_1_x8_2_dof_x8"] = 1.6
Q_FULL_BODY_HOME_DICT["joint_right_arm_1_x6_1_dof_x6"] = 0.34
Q_FULL_BODY_HOME_DICT["joint_right_arm_1_x6_2_dof_x6"] = 1.6
Q_FULL_BODY_HOME_DICT["joint_right_arm_1_x4_1_dof_x4"] = 1.4
Q_FULL_BODY_HOME_DICT["joint_right_arm_1_hand_1_x4_1_dof_x4"] = -0.26
Q_FULL_BODY_HOME_DICT["joint_right_arm_1_hand_1_x4_2_dof_x4"] = 0.0
Q_FULL_BODY_HOME_DICT["joint_right_arm_1_hand_1_slider_1"] = 0.0
Q_FULL_BODY_HOME_DICT["joint_right_arm_1_hand_1_slider_2"] = 0.0
Q_FULL_BODY_HOME_DICT["joint_left_arm_2_x8_1_dof_x8"] = -1.7
Q_FULL_BODY_HOME_DICT["joint_left_arm_2_x8_2_dof_x8"] = -1.6
Q_FULL_BODY_HOME_DICT["joint_left_arm_2_x6_1_dof_x6"] = -0.34
Q_FULL_BODY_HOME_DICT["joint_left_arm_2_x6_2_dof_x6"] = -1.6
Q_FULL_BODY_HOME_DICT["joint_left_arm_2_x4_1_dof_x4"] = -1.4
Q_FULL_BODY_HOME_DICT["joint_left_arm_2_hand_1_x4_1_dof_x4"] = -1.7
Q_FULL_BODY_HOME_DICT["joint_left_arm_2_hand_1_x4_2_dof_x4"] = 0.0
Q_FULL_BODY_HOME_DICT["joint_left_arm_2_hand_1_slider_1"] = 0.0
Q_FULL_BODY_HOME_DICT["joint_left_arm_2_hand_1_slider_2"] = 0.0
Q_FULL_BODY_HOME: NDArray = np.array(
    [v for v in Q_FULL_BODY_HOME_DICT.values()],
    dtype=np.float32,
)
Q_FULL_BODY_KEYS: List[str] = list(Q_FULL_BODY_HOME_DICT.keys())

# IK joint masks (computed separately for each arm)
Q_MASK_R: NDArray = np.array([0, 1, 2, 3, 4, 5, 6])
Q_MASK_L: NDArray = np.array([9, 10, 11, 12, 13, 14, 15])

# mocap objects are set by hand poses
MOCAP_ID_R: int = 0
MOCAP_ID_L: int = 1

# IK hyperparameters
# TODO: more tuning
IK_RES_RAD: float = 0.04
IK_RES_REG: float = 1e-3
IK_JAC_RAD: float = 0.04
IK_JAC_REG: float = 1e-3

# image sizes depend on camera
CAM_HEAD_IMG_WIDTH: int = 640
CAM_HEAD_IMG_HEIGHT: int = 480
CAM_GRIP_IMG_WIDTH: int = 60
CAM_GRIP_IMG_HEIGHT: int = 40

# overhead camera is used for visualization
CAM_TOP_IMG_WIDTH: int = 640
CAM_TOP_IMG_HEIGHT: int = 480

# cube is randomly spawned on episode start
CUBE_SPAWN_RANGE_X: Tuple[float] = [0.1, 0.3]
CUBE_SPAWN_RANGE_Y: Tuple[float] = [0.5, 0.7]
CUBE_SPAWN_RANGE_Z: Tuple[float] = [0.52, 0.62]

# reward shaping
REWARD_SUCCESS_THRESHOLD: float = 2.0
REWARD_VEL_PENALTY: float = 0.01
REWARD_GRIP_DIST: float = 0.01
REWARD_TOUCH_CUBE: float = 1.0
REWARD_LIFT_CUBE: float = 1.0

# pre-compute gripper "slider" ranges for faster callback
EE_S_MIN: float = 0.0
EE_S_MAX: float = -0.034
EE_S_RANGE: float = EE_S_MAX - EE_S_MIN
EE_DEFAULT_ORN: NDArray = np.array([1, 0, 0, 0])

# ctrl ids for gripping
CTRL_ID_R_GRIP: int = 8
CTRL_ID_L_GRIP: int = 18

# exponential filtering
CTRL_ALPHA: float = 0.2

register(
    id="KManipSoloArm",
    entry_point="gym_kmanip.env:KManipEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
    nondeterministic=True,
    kwargs={
        "xml_filename": "_env_solo_arm.xml",
        "obs_list": [
            "q_pos",  # joint positions
            "q_vel",  # joint velocities
        ],
        "act_list": [
            "eer_pos",  # right end effector position
            "eer_orn",  # right end effector orientation
            "grip_r",  # right gripper
        ],
        "q_home": Q_SOLO_ARM_HOME,
        "q_dict": Q_SOLO_ARM_HOME_DICT,
        "q_keys": Q_SOLO_ARM_KEYS,
    },
)

register(
    id="KManipSoloArmVision",
    entry_point="gym_kmanip.env:KManipEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
    nondeterministic=True,
    kwargs={
        "xml_filename": "_env_solo_arm.xml",
        "obs_list": [
            "q_pos",  # joint positions
            "q_vel",  # joint velocities
            "cam_head",  # robot head camera
            "cam_grip_r",  # right gripper camera
        ],
        "act_list": [
            "eer_pos",  # right end effector position
            "eer_orn",  # right end effector orientation
            "grip_r",  # right gripper
        ],
        "q_home": Q_SOLO_ARM_HOME,
        "q_dict": Q_SOLO_ARM_HOME_DICT,
        "q_keys": Q_SOLO_ARM_KEYS,
    },
)

register(
    id="KManipDualArm",
    entry_point="gym_kmanip.env:KManipEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
    nondeterministic=True,
    kwargs={
        "xml_filename": "_env_dual_arm.xml",
        "obs_list": [
            "q_pos",  # joint positions
            "q_vel",  # joint velocities
        ],
        "act_list": [
            "eel_pos",  # left end effector position
            "eel_orn",  # left end effector orientation
            "eer_pos",  # right end effector position
            "eer_orn",  # right end effector orientation
            "grip_l",  # left gripper
            "grip_r",  # right gripper
        ],
        "q_home": Q_DUAL_ARM_HOME,
        "q_dict": Q_DUAL_ARM_HOME_DICT,
        "q_keys": Q_DUAL_ARM_KEYS,
    },
)

register(
    id="KManipDualArmVision",
    entry_point="gym_kmanip.env:KManipEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
    nondeterministic=True,
    kwargs={
        "xml_filename": "_env_dual_arm.xml",
        "obs_list": [
            "q_pos",  # joint positions
            "q_vel",  # joint velocities
            "cam_head",  # robot head camera
            "cam_grip_l",  # left gripper camera
            "cam_grip_r",  # right gripper camera
        ],
        "act_list": [
            "eel_pos",  # left end effector position
            "eel_orn",  # left end effector orientation
            "eer_pos",  # right end effector position
            "eer_orn",  # right end effector orientation
            "grip_l",  # left gripper
            "grip_r",  # right gripper
        ],
        "q_home": Q_DUAL_ARM_HOME,
        "q_dict": Q_DUAL_ARM_HOME_DICT,
        "q_keys": Q_DUAL_ARM_KEYS,
    },
)

register(
    id="KManipTorso",
    entry_point="gym_kmanip.env:KManipEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
    nondeterministic=True,
    kwargs={
        "xml_filename": "_env_full_body.xml",
        "obs_list": [
            "q_pos",  # joint positions
            "q_vel",  # joint velocities
        ],
        "act_list": [
            "eel_pos",  # left end effector position
            "eel_orn",  # left end effector orientation
            "eer_pos",  # right end effector position
            "eer_orn",  # right end effector orientation
            "grip_l",  # left gripper
            "grip_r",  # right gripper
        ],
        "q_home": Q_FULL_BODY_HOME,
        "q_dict": Q_FULL_BODY_HOME_DICT,
        "q_keys": Q_FULL_BODY_KEYS,
    },
)

register(
    id="KManipTorsoVision",
    entry_point="gym_kmanip.env:KManipEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
    nondeterministic=True,
    kwargs={
        "xml_filename": "_env_full_body.xml",
        "obs_list": [
            "q_pos",  # joint positions
            "q_vel",  # joint velocities
            "cam_head",  # robot head camera
            "cam_grip_l",  # left gripper camera
            "cam_grip_r",  # right gripper camera
        ],
        "act_list": [
            "eel_pos",  # left end effector position
            "eel_orn",  # left end effector orientation
            "eer_pos",  # right end effector position
            "eer_orn",  # right end effector orientation
            "grip_l",  # left gripper
            "grip_r",  # right gripper
        ],
        "q_home": Q_FULL_BODY_HOME,
        "q_dict": Q_FULL_BODY_HOME_DICT,
        "q_keys": Q_FULL_BODY_KEYS,
    },
)
