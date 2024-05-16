import asyncio
import os
import time
import math
from typing import Dict
import math

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
import rerun as rr
from scipy.spatial.transform import Rotation as R
from vuer import Vuer, VuerSession
from vuer.schemas import Box, Capsule, Hands, PointLight, Urdf

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

# conversion between MuJoCo and Vuer axes
MJ_TO_VUER_AXES: NDArray = np.array([0, 2, 1], dtype=np.uint8)
MJ_TO_VUER_AXES_SIGN: NDArray = np.array([-1, 1, 1], dtype=np.int8)

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

# env_name = "KManipSoloArm"
# env_name = "KManipSoloArmVision"
env_name = "KManipDualArm"
# env_name = "KManipDualArmVision"
# env_name = "KManipTorso"
# env_name = "KManipTorsoVision"
env = gym.make(env_name)
env.reset()
mj_data = env.unwrapped.mj_env.physics.data

# global variables get updated by various async functions
q_lock = asyncio.Lock()
q: Dict[str, float] = env.q_dict
mj_q: NDArray = mj_data.qpos.copy()

# global variables for gripper
grip_r: float = 0.0
grip_l: float = 0.0

# gobal variables for ee site position and orientation
eer_site_pos: NDArray = mj_data.mocap_pos[k.MOCAP_ID_R].copy()
eer_site_orn: NDArray = mj_data.mocap_quat[k.MOCAP_ID_R].copy()
eel_site_pos: NDArray = mj_data.mocap_pos[k.MOCAP_ID_L].copy()
eel_site_orn: NDArray = mj_data.mocap_quat[k.MOCAP_ID_L].copy()

# gobal variables for cube position and orientation
cube_pos: NDArray = mj_data.body("cube").xpos
cube_orn: NDArray = mj_data.body("cube").xmat


async def run_env() -> None:
    start_time = time.time()
    async with q_lock:
        global cube_pos, cube_orn
        cube_pos = mj_data.body("cube").xpos
        cube_orn = mj_data.body("cube").xmat
        action = env.action_space.sample()
        if "eer_pos" in action:
            global eer_site_pos, eer_site_orn, grip_r
            action["eer_pos"] = eer_site_pos
            action["eer_orn"] = eer_site_orn
            action["grip_r"] = grip_r
        if "eel_pos" in action:
            global eel_site_pos, eel_site_orn, grip_l
            action["eel_pos"] = eel_site_pos
            action["eel_orn"] = eel_site_orn
            action["grip_l"] = grip_l
        global q
        for i, val in enumerate(mj_data.qpos):
            joint_name = env.q_keys[i]
            q[joint_name] = val
    print(f"env step took {time.time() - start_time} seconds")


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
        # pinching with middle finger controls gripper
        rmiddl_pos: NDArray = np.array(event.value["rightLandmarks"][MIDLE_FINGER_ID])
        rgrip_dist: float = np.linalg.norm(rthumb_pos - rmiddl_pos) / PINCH_DIST_OPENED
        # orientation is calculated from wrist rotation matrix
        wrist_rotation: NDArray = np.array(event.value["rightHand"]).reshape(4, 4)[:3, :3]
        async with q_lock:
            global eer_site_pos, eer_site_orn, grip_r
            eer_site_pos = np.multiply(rthumb_pos[MJ_TO_VUER_AXES], MJ_TO_VUER_AXES_SIGN)
            print(f"goal_pos_eer {eer_site_pos}")
            eer_site_orn = R.from_matrix(wrist_rotation).as_quat()
            print(f"goal_orn_eer {eer_site_orn}")
            grip_r = rgrip_dist
            print(f"right gripper at {grip_r}")
    # left hand
    lindex_pos: NDArray = np.array(event.value["leftLandmarks"][INDEX_FINGER_ID])
    lthumb_pos: NDArray = np.array(event.value["leftLandmarks"][THUMB_FINGER_ID])
    lpinch_dist: NDArray = np.linalg.norm(lindex_pos - lthumb_pos)
    # index finger to thumb pinch turns on tracking
    if lpinch_dist < PINCH_DIST_CLOSED:
        print("Pinch detected in left hand")
        # pinching with middle finger controls gripper
        lmiddl_pos: NDArray = np.array(event.value["leftLandmarks"][MIDLE_FINGER_ID])
        lgrip_dist: float = np.linalg.norm(lthumb_pos - lmiddl_pos) / PINCH_DIST_OPENED
        # orientation is calculated from wrist rotation matrix
        wrist_rotation: NDArray = np.array(event.value["leftHand"]).reshape(4, 4)[:3, :3]
        async with q_lock:
            global eel_site_pos, eel_site_orn, grip_l
            eel_site_pos = np.multiply(lthumb_pos[MJ_TO_VUER_AXES], MJ_TO_VUER_AXES_SIGN)
            print(f"goal_pos_eel {eel_site_pos}")
            eel_site_orn = R.from_matrix(wrist_rotation).as_quat()
            print(f"goal_orn_eel {eel_site_orn}")
            grip_l = lgrip_dist
            print(f"left gripper at {grip_l}")


@app.spawn(start=True)
async def main(session: VuerSession):
    session.upsert @ PointLight(intensity=VUER_LIGHT_INTENSITY, position=VUER_LIGHT_POS)
    session.upsert @ Hands(fps=HAND_FPS, stream=True, key="hands")
    await asyncio.sleep(0.1)
    session.upsert @ Urdf(
        src=URDF_WEB,
        jointValues=env.q_dict,
        position=START_POS_TRUNK_VUER,
        rotation=START_EUL_TRUNK_VUER,
        key="robot",
    )
    global q, img
    while True:
        await asyncio.gather(
            run_env(),  # ~1ms
            # update_image(),  # ~10ms
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
            session.upsert @ Box(
                args=[0.1, 0.1, 0.1, 101, 101, 101],
                position=[0, 0.05, 0],
                color="red",
                materialType="standard",
                material=dict(color="#23aaff"),
                key="fox-1",
            )
            session.upsert @ Capsule(
                args=[0.05, 0.1, 101, 101],
                position=eer_site_pos,
                rotation=eer_site_orn,
                color="blue",
                materialType="standard",
                material=dict(color="#23aaff"),
                key="eel-site",
            )
            session.upsert @ Capsule(
                args=[0.05, 0.1, 101, 101],
                position=eer_site_pos,
                rotation=eer_site_orn,
                color="red",
                materialType="standard",
                material=dict(color="#23aaff"),
                key="eer-site",
            )