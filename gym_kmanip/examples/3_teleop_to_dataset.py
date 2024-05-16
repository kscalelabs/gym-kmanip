# TODO: WIP - DOES NOT WORK

import asyncio
from copy import deepcopy
import os
import time
import math
from typing import List, Dict
import math

from dm_control import viewer
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
import rerun as rr
from scipy.spatial.transform import Rotation as R
from vuer import Vuer, VuerSession
from vuer.schemas import Hands, ImageBackground, PointLight, Urdf

import gym_kmanip as k

# web urdf is used for vuer
URDF_WEB: str = (
    "https://raw.githubusercontent.com/kscalelabs/webstompy/master/urdf/stompy_tiny_glb/robot.urdf"
)

# starting positions for robot trunk relative to world frames
START_POS_TRUNK_VUER: NDArray = np.array([0, 1, 0])
START_EUL_TRUNK_VUER: NDArray = np.array([-math.pi / 2, 0, 0])

# starting positions for robot end effectors are defined relative to robot trunk frame
# which is right in the middle of the chest
START_POS_EER_VUER: NDArray = np.array([-0.2, -0.2, -0.2])
START_POS_EEL_VUER: NDArray = np.array([0.2, -0.2, -0.2])
START_POS_EER_VUER += START_POS_TRUNK_VUER
START_POS_EEL_VUER += START_POS_TRUNK_VUER

# env_name = "KManipSoloArm"
# env_name = "KManipSoloArmVision"
env_name = "KManipDualArm"
# env_name = "KManipDualArmVision"
# env_name = "KManipTorso"
# env_name = "KManipTorsoVision"
env = gym.make(env_name)

# camera streaming is done throgh OpenCV
IMAGE_WIDTH: int = 1280
IMAGE_HEIGHT: int = 480
aspect_ratio: float = IMAGE_WIDTH / IMAGE_HEIGHT
CAMERA_FPS: int = 60
VUER_IMG_QUALITY: int = 20
BGR_TO_RGB: NDArray = np.array([2, 1, 0], dtype=np.uint8)
CAMERA_DISTANCE: int = 5
IMAGE_POS: NDArray = np.array([0, 0, -10])
IMAGE_EUL: NDArray = np.array([0, 0, 0])
img_lock = asyncio.Lock()
img: NDArray[np.uint8] = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

# umi dataset requires these values
DATASET_DIR: str = os.path.join(os.path.dirname(__file__), "data")
DATASET_NAME: str = "test.zarr"
DATASET_OUTPUT_PATH = os.path.join(DATASET_DIR, DATASET_NAME)

required_datasets = {
    "data/robot0_demo_end_pose",
    "data/robot0_demo_start_pose",
    "data/robot0_eef_pos",
    "data/robot0_eef_rot_axis_angle",
    "data/robot0_gripper_width",
    "meta/episode_ends",
    "data/camera0_rgb",
}


async def update_image() -> None:
    global cam
    if not cam.isOpened():
        raise ValueError("Camera is not available")
    start = time.time()
    ret, frame = cam.read()
    if ret:
        async with img_lock:
            global img
            img = frame[:, :, BGR_TO_RGB]
    else:
        print("Failed to read frame")
    print(f"Time to update image: {time.time() - start}")


# Vuer rendering params
MAX_FPS: int = 60
VUER_LIGHT_POS: NDArray = np.array([0, 2, 2])
VUER_LIGHT_INTENSITY: float = 10.0

# Vuer hand tracking and pinch detection params
HAND_FPS: int = 30
INDEX_FINGER_ID: int = 9
THUMB_FINGER_ID: int = 4
MIDLE_FINGER_ID: int = 14
PINCH_DIST_OPENED: float = 0.10  # 10cm
PINCH_DIST_CLOSED: float = 0.01  # 1cm

# pre-compute gripper "slider" ranges for faster callback
EE_S_MIN: float = -0.034
EE_S_MAX: float = 0.0
EE_S_RANGE: float = EE_S_MAX - EE_S_MIN

# global variables get updated by various async functions
q_lock = asyncio.Lock()
q: Dict[str, float] = deepcopy(START_Q)
goal_pos_eer: NDArray = START_POS_EER_VUER
goal_orn_eer: NDArray = p.getQuaternionFromEuler(START_EUL_TRUNK_VUER)
goal_pos_eel: NDArray = START_POS_EEL_VUER
goal_orn_eel: NDArray = p.getQuaternionFromEuler(START_EUL_TRUNK_VUER)


async def ik(arm: str) -> None:
    start_time = time.time()
    if arm == "right":
        global goal_pos_eer, goal_orn_eer
        ee_id = pb_eer_id
        ee_chain = EER_CHAIN_ARM
        pos = goal_pos_eer
        orn = goal_orn_eer
    else:
        global goal_pos_eel, goal_orn_eel
        ee_id = pb_eel_id
        ee_chain = EEL_CHAIN_ARM
        pos = goal_pos_eel
        orn = goal_orn_eel
    # print(f"ik {arm} {pos} {orn}")
    pb_q = p.calculateInverseKinematics(
        pb_robot_id,
        ee_id,
        pos,
        orn,
        pb_joint_lower_limit,
        pb_joint_upper_limit,
        pb_joint_ranges,
        pb_start_q,
    )
    async with q_lock:
        global q
        for i, val in enumerate(pb_q):
            joint_name = IK_Q_LIST[i]
            if joint_name in ee_chain:
                q[joint_name] = val
                p.resetJointState(pb_robot_id, pb_q_map[joint_name], val)
    print(f"ik {arm} took {time.time() - start_time} seconds")


app = Vuer()


@app.add_handler("HAND_MOVE")
async def hand_handler(event, _):
    # right hand
    rindex_pos: NDArray = np.array(event.value["rightLandmarks"][INDEX_FINGER_ID])
    rthumb_pos: NDArray = np.array(event.value["rightLandmarks"][THUMB_FINGER_ID])
    rpinch_dist: NDArray = np.linalg.norm(rindex_pos - rthumb_pos)
    # index finger to thumb pinch turns on tracking
    if rpinch_dist < PINCH_DIST_CLOSED:
        print("Pinch detected in right hand")
        global goal_pos_eer, goal_orn_eer
        goal_pos_eer = np.multiply(rthumb_pos[PB_TO_VUER_AXES], PB_TO_VUER_AXES_SIGN)
        print(f"goal_pos_eer {goal_pos_eer}")
        # pinching with middle finger controls gripper
        rmiddl_pos: NDArray = np.array(event.value["rightLandmarks"][MIDLE_FINGER_ID])
        rgrip_dist: float = np.linalg.norm(rthumb_pos - rmiddl_pos) / PINCH_DIST_OPENED
        print(f"right gripper at {rgrip_dist}")
        _s: float = EE_S_MIN + rgrip_dist * EE_S_RANGE
        async with q_lock:
            q["joint_right_arm_1_hand_1_slider_1"] = _s
            q["joint_right_arm_1_hand_1_slider_2"] = _s
        # orientation is calculated from wrist rotation matrix
        wrist_rotation: NDArray = np.array(event.value["rightHand"]).reshape(4, 4)[:3, :3]
        goal_orn_eer = R.from_matrix(wrist_rotation).as_quat()
    # left hand
    lindex_pos: NDArray = np.array(event.value["leftLandmarks"][INDEX_FINGER_ID])
    lthumb_pos: NDArray = np.array(event.value["leftLandmarks"][THUMB_FINGER_ID])
    lpinch_dist: NDArray = np.linalg.norm(lindex_pos - lthumb_pos)
    # index finger to thumb pinch turns on tracking
    if lpinch_dist < PINCH_DIST_CLOSED:
        print("Pinch detected in left hand")
        global goal_pos_eel, goal_orn_eel
        goal_pos_eel = np.multiply(lthumb_pos[PB_TO_VUER_AXES], PB_TO_VUER_AXES_SIGN)
        print(f"goal_pos_eel {goal_pos_eel}")
        # pinching with middle finger controls gripper
        lmiddl_pos: NDArray = np.array(event.value["leftLandmarks"][MIDLE_FINGER_ID])
        lgrip_dist: float = np.linalg.norm(lthumb_pos - lmiddl_pos) / PINCH_DIST_OPENED
        _s: float = EE_S_MIN + lgrip_dist * EE_S_RANGE
        print(f"left gripper at {lgrip_dist}")
        async with q_lock:
            q["joint_left_arm_2_hand_1_slider_1"] = _s
            q["joint_left_arm_2_hand_1_slider_2"] = _s
        # orientation is calculated from wrist rotation matrix
        wrist_rotation: NDArray = np.array(event.value["leftHand"]).reshape(4, 4)[:3, :3]
        goal_orn_eer = R.from_matrix(wrist_rotation).as_quat()


@app.spawn(start=True)
async def main(session: VuerSession):
    session.upsert @ PointLight(intensity=VUER_LIGHT_INTENSITY, position=VUER_LIGHT_POS)
    session.upsert @ Hands(fps=HAND_FPS, stream=True, key="hands")
    await asyncio.sleep(0.1)
    session.upsert @ Urdf(
        src=URDF_WEB,
        jointValues=START_Q,
        position=START_POS_TRUNK_VUER,
        rotation=START_EUL_TRUNK_VUER,
        key="robot",
    )
    global q, img
    while True:
        await asyncio.gather(
            ik("left"),  # ~1ms
            ik("right"),  # ~1ms
            update_image(),  # ~10ms
            asyncio.sleep(1 / MAX_FPS),  # ~16ms @ 60fps
        )
        async with q_lock:
            session.upsert @ Urdf(
                src=URDF_WEB,
                jointValues=q,
                position=START_POS_TRUNK_VUER,
                rotation=START_EUL_TRUNK_VUER,
                key="robot",
            )
        async with img_lock:
            session.upsert(
                ImageBackground(
                    img,
                    format="jpg",
                    quality=VUER_IMG_QUALITY,
                    interpolate=True,
                    fixed=True,
                    aspect=aspect_ratio,
                    distanceToCamera=CAMERA_DISTANCE,
                    position=IMAGE_POS,
                    rotation=IMAGE_EUL,
                    key="video",
                ),
                to="bgChildren",
            )