from collections import OrderedDict as ODict
from dataclasses import dataclass
import os
from typing import List, OrderedDict, Tuple

from gymnasium.envs.registration import register
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

ASSETS_DIR: str = os.path.join(os.path.dirname(__file__), "assets")
DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")

# this one is the best, fight me
DATE_FORMAT: str = "%mm%dd%Yy_%Hh%Mm"

# MuJoCo uses XML files
SOLO_ARM_MJCF: str = "_env_solo_arm.xml"
DUAL_ARM_MJCF: str = "_env_dual_arm.xml"
TORSO_MJCF: str = "_env_torso.xml"

# Vuer and Rerun use URDF files
SOLO_ARM_URDF: str = "stompy_tiny_solo_arm_glb.urdf"
DUAL_ARM_URDF: str = "stompy_dual_arm_tiny_glb.urdf"
TORSO_URDF: str = "stompy_tiny_glb/robot.urdf"

# Misc
MAX_EPISODE_STEPS: int = 64
FPS: int = 30
CONTROL_TIMESTEP: float = 0.02  # ms
MAX_Q_VEL: float = np.pi  # rad/s

# exponential filtering for control signal
CTRL_ALPHA: float = 1.0

# IK hyperparameters
IK_RES_RAD: float = 0.02
IK_RES_REG_PREV: float = 6e-3
IK_RES_REG_HOME: float = 2e-6
IK_JAC_RAD: float = 0.02
IK_JAC_REG: float = 9e-3

# Datasets are stored in HDF5 format on HuggingFace's LeRobot
H5PY_CHUNK_SIZE_BYTES: int = 1024**2 * 2
HF_LEROBOT_VERSION: str = "v1.4"
HF_LEROBOT_BATCH_SIZE: int = 32
HF_LEROBOT_NUM_WORKERS: int = 8

# Gym spaces dtypes
OBS_DTYPE: np.dtype = np.float64
ACT_DTYPE: np.dtype = np.float32

Q_SOLO_ARM_HOME_DICT: OrderedDict[str, float] = ODict()
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
    dtype=ACT_DTYPE,
)
Q_SOLO_ARM_KEYS: List[str] = list(Q_SOLO_ARM_HOME_DICT.keys())

Q_DUAL_ARM_HOME_DICT: OrderedDict[str, float] = ODict()
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
    dtype=ACT_DTYPE,
)
Q_DUAL_ARM_KEYS: List[str] = list(Q_DUAL_ARM_HOME_DICT.keys())

Q_TORSO_HOME_DICT: OrderedDict[str, float] = ODict()
Q_TORSO_HOME_DICT["joint_head_1_x4_1_dof_x4"] = -1.0
Q_TORSO_HOME_DICT["joint_head_1_x4_2_dof_x4"] = 0.0
Q_TORSO_HOME_DICT["joint_right_arm_1_x8_1_dof_x8"] = 1.7
Q_TORSO_HOME_DICT["joint_right_arm_1_x8_2_dof_x8"] = 1.6
Q_TORSO_HOME_DICT["joint_right_arm_1_x6_1_dof_x6"] = 0.34
Q_TORSO_HOME_DICT["joint_right_arm_1_x6_2_dof_x6"] = 1.6
Q_TORSO_HOME_DICT["joint_right_arm_1_x4_1_dof_x4"] = 1.4
Q_TORSO_HOME_DICT["joint_right_arm_1_hand_1_x4_1_dof_x4"] = -0.26
Q_TORSO_HOME_DICT["joint_right_arm_1_hand_1_slider_1"] = 0.0
Q_TORSO_HOME_DICT["joint_right_arm_1_hand_1_slider_2"] = 0.0
Q_TORSO_HOME_DICT["joint_right_arm_1_hand_1_x4_2_dof_x4"] = 0.0
Q_TORSO_HOME_DICT["joint_left_arm_2_x8_1_dof_x8"] = -1.7
Q_TORSO_HOME_DICT["joint_left_arm_2_x8_2_dof_x8"] = -1.6
Q_TORSO_HOME_DICT["joint_left_arm_2_x6_1_dof_x6"] = -0.34
Q_TORSO_HOME_DICT["joint_left_arm_2_x6_2_dof_x6"] = -1.6
Q_TORSO_HOME_DICT["joint_left_arm_2_x4_1_dof_x4"] = -1.4
Q_TORSO_HOME_DICT["joint_left_arm_2_hand_1_x4_1_dof_x4"] = -1.7
Q_TORSO_HOME_DICT["joint_left_arm_2_hand_1_slider_1"] = 0.0
Q_TORSO_HOME_DICT["joint_left_arm_2_hand_1_slider_2"] = 0.0
Q_TORSO_HOME_DICT["joint_left_arm_2_hand_1_x4_2_dof_x4"] = 0.0
Q_TORSO_HOME: NDArray = np.array(
    [v for v in Q_TORSO_HOME_DICT.values()],
    dtype=ACT_DTYPE,
)
Q_TORSO_KEYS: List[str] = list(Q_TORSO_HOME_DICT.keys())

# MuJoCo will have different IDs for q and ctrl based on environment
Q_ID_R_MASK_SOLO: NDArray = np.array([0, 1, 2, 3, 4, 5, 6])
CTRL_ID_R_GRIP_SOLO: NDArray = np.array([8, 9])

Q_ID_R_MASK_DUAL: NDArray = np.array([0, 1, 2, 3, 4, 5, 6])
Q_ID_L_MASK_DUAL: NDArray = np.array([10, 11, 12, 13, 14, 15, 16])
CTRL_ID_R_GRIP_DUAL: NDArray = np.array([8, 9])
CTRL_ID_L_GRIP_DUAL: NDArray = np.array([18, 19])

Q_ID_R_MASK_TORSO: NDArray = np.array([2, 3, 4, 5, 6, 7])
Q_ID_L_MASK_TORSO: NDArray = np.array([11, 12, 13, 14, 15, 16])
CTRL_ID_R_GRIP_TORSO: NDArray = np.array([8, 9])
CTRL_ID_L_GRIP_TORSO: NDArray = np.array([17, 18])

# mocap objects are set by hand poses
MOCAP_ID_R: int = 0
MOCAP_ID_L: int = 1


@dataclass
class Cam:
    w: int  # image width
    h: int  # image height
    c: int  # image channels
    fl: int  # focal length
    pp: Tuple[int]  # principal point
    name: str  # name
    log_name: str  # name used for hierarchical logging
    low: int = 0
    high: int = 255
    dtype = np.uint8


CAMERAS: OrderedDict[str, Cam] = ODict()
CAMERAS["head"] = Cam(640, 480, 3, 448, (320, 240), "head", "camera/head")
CAMERAS["top"] = Cam(640, 480, 3, 448, (320, 240), "top", "camera/top")
CAMERAS["grip_r"] = Cam(640, 480, 3, 448, (30, 20), "grip_r", "camera/grip_r")
CAMERAS["grip_l"] = Cam(60, 40, 3, 45, (30, 20), "grip_l", "camera/grip_l")

# cube is randomly spawned on episode start
CUBE_SPAWN_RANGE: NDArray = np.array(
    [
        [0.1, 0.12],  # X
        [0.5, 0.712],  # Y
        [0.6, 0.605],  # Z
    ]
)

# ee control will expect values in range [-1, 1]
# this will define the max "delta" around the current ee pose
EE_POS_DELTA: NDArray = np.array(
    [
        0.01, # X (meters)
        0.01, # Y (meters)
        0.01, # Z (meters)
    ]
)
EE_ORN_DELTA: NDArray = np.array(
    [
        0.1, # X (radians)
        0.1, # Y (radians)
        0.1, # Z (radians)
    ]
)
# default orientation if ee_orn not specified
EE_DEFAULT_ORN: NDArray = np.array([1, 0, 0, 0])

# prevent rounding errors
EPSILON: float = 1e-6

# q control will expect values in range [-1, 1]
# this will define the max "delta" around the current q pos
Q_POS_DELTA: NDArray = 0.1 # radians

# pre-compute gripper "slider" ranges for faster callback
EE_S_MIN: float = -0.029 # closed
EE_S_MAX: float = 0.005 # open
EE_S_DELTA: float = 0.0001

# reward shaping
REWARD_SUCCESS_THRESHOLD: float = 2.0
REWARD_VEL_PENALTY: float = 0.01
REWARD_GRIP_DIST: float = 0.01
REWARD_TOUCH_CUBE: float = 1.0
REWARD_LIFT_CUBE: float = 1.0

# MuJoCo and Scipy/Rerun use different quaternion conventions
# https://github.com/clemense/quaternion-conventions
XYZW_2_WXYZ: NDArray = np.array([3, 0, 1, 2])
WXYZ_2_XYZW: NDArray = np.array([1, 2, 3, 0])
MJ_TO_VUER_ROT: R = R.from_euler("z", np.pi) * R.from_euler("x", np.pi / 2)
VUER_TO_MJ_ROT: R = MJ_TO_VUER_ROT.inv()

# Vuer is used for teleop
VUER_IMG_QUALITY: int = 20

# real robot uses cv2 for camera capture
CAMERA_FPS: int = 30
BGR_TO_RGB: NDArray = np.array([2, 1, 0], dtype=np.uint8)

def mj2vuer_pos(pos: NDArray) -> NDArray:
    return MJ_TO_VUER_ROT.apply(pos)


def mj2vuer_orn(orn: NDArray, offset: NDArray = None) -> NDArray:
    rot = R.from_quat(orn[XYZW_2_WXYZ]) * MJ_TO_VUER_ROT
    if offset is not None:
        rot = R.from_quat(offset[XYZW_2_WXYZ]) * rot
    return rot.as_euler("xyz")


def vuer2mj_pos(pos: NDArray) -> NDArray:
    return VUER_TO_MJ_ROT.apply(pos)


def vuer2mj_orn(orn: R) -> NDArray:
    rot = orn * VUER_TO_MJ_ROT
    return rot.as_quat()[WXYZ_2_XYZW]


register(
    id="gym_kmanip/KManipSoloArm",
    entry_point="gym_kmanip.env_base:KManipEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
    nondeterministic=True,
    kwargs={
        "mjcf_filename": SOLO_ARM_MJCF,
        "urdf_filename": SOLO_ARM_URDF,
        "obs_list": [
            "q_pos",  # joint positions
            "q_vel",  # joint velocities
            "cube_pos",  # cube position
            "cube_orn",  # cube orientation
        ],
        "act_list": [
            "eer_pos",  # right end effector position
            "eer_orn",  # right end effector orientation
            "grip_r",  # right gripper
        ],
        "q_pos_home": Q_SOLO_ARM_HOME,
        "q_dict": Q_SOLO_ARM_HOME_DICT,
        "q_keys": Q_SOLO_ARM_KEYS,
        "q_id_r_mask": Q_ID_R_MASK_SOLO,
        "ctrl_id_r_grip": CTRL_ID_R_GRIP_SOLO,
    },
)

register(
    id="gym_kmanip/KManipSoloArmQPos",
    entry_point="gym_kmanip.env_base:KManipEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
    nondeterministic=True,
    kwargs={
        "mjcf_filename": SOLO_ARM_MJCF,
        "urdf_filename": SOLO_ARM_URDF,
        "obs_list": [
            "q_pos",  # joint positions
            "q_vel",  # joint velocities
            "cube_pos",  # cube position
            "cube_orn",  # cube orientation
        ],
        "act_list": [
            "q_pos_r",  # joint positions for right arm
            "grip_r",  # right gripper
        ],
        "q_pos_home": Q_SOLO_ARM_HOME,
        "q_dict": Q_SOLO_ARM_HOME_DICT,
        "q_keys": Q_SOLO_ARM_KEYS,
        "q_id_r_mask": Q_ID_R_MASK_SOLO,
        "ctrl_id_r_grip": CTRL_ID_R_GRIP_SOLO,
    },
)


# register(
#     id="gym_kmanip/KManipSoloArmVision",
#     entry_point="gym_kmanip.env_base:KManipEnv",
#     max_episode_steps=MAX_EPISODE_STEPS,
#     nondeterministic=True,
#     kwargs={
#         "mjcf_filename": SOLO_ARM_MJCF,
#         "urdf_filename": SOLO_ARM_URDF,
#         "obs_list": [
#             "q_pos",  # joint positions
#             "q_vel",  # joint velocities
#             "camera/head",  # robot head camera
#             "camera/grip_r",  # right gripper camera
#         ],
#         "act_list": [
#             "eer_pos",  # right end effector position
#             "eer_orn",  # right end effector orientation
#             "grip_r",  # right gripper
#             # "q_pos",  # joint positions for right arm
#         ],
#         "q_pos_home": Q_SOLO_ARM_HOME,
#         "q_dict": Q_SOLO_ARM_HOME_DICT,
#         "q_keys": Q_SOLO_ARM_KEYS,
#         "q_id_r_mask": Q_ID_R_MASK_SOLO,
#         "ctrl_id_r_grip": CTRL_ID_R_GRIP_SOLO,
#     },
# )

register(
    id="gym_kmanip/KManipSoloArmVision",
    entry_point="gym_kmanip.env_base:KManipEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
    nondeterministic=True,
    kwargs={
        "mjcf_filename": SOLO_ARM_MJCF,
        "urdf_filename": SOLO_ARM_URDF,
        "obs_list": [
            # without qpos formatting won't work
            "q_pos",  # joint positions
            "q_vel",  # joint velocities
            "camera/head",  # robot head camera
            "camera/grip_r",  # right gripper camera
        ],
        "act_list": [
            "q_pos",  # joint positions for right arm
            "eer_pos",  # right end effector position
            "eer_orn",  # right end effector orientation
            "grip_r",  # right gripper
        ],
        "q_pos_home": Q_SOLO_ARM_HOME,
        "q_dict": Q_SOLO_ARM_HOME_DICT,
        "q_keys": Q_SOLO_ARM_KEYS,
        "q_id_r_mask": Q_ID_R_MASK_SOLO,
        "ctrl_id_r_grip": CTRL_ID_R_GRIP_SOLO,
    },
)
