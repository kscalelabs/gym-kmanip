import asyncio
import os
import time
from datetime import datetime
import math
from typing import Dict
import math

from dm_control import viewer
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
import rerun as rr
from scipy.spatial.transform import Rotation as R
from vuer import Vuer, VuerSession
from vuer.schemas import Box, Capsule, Hands, PointLight, Urdf

import gym_kmanip as k

# choose your environment
# ENV_NAME: str = "KManipSoloArm"
# ENV_NAME: str = "KManipSoloArmVision"
# ENV_NAME: str = "KManipDualArm"
# ENV_NAME: str = "KManipDualArmVision"
ENV_NAME: str = "KManipTorso"
# ENV_NAME: str = "KManipTorsoVision"

# dataset is recorded as a rerun replay
LOG_DATASET: bool = False
if LOG_DATASET:
    DATASET_DIR: str = os.path.join(os.path.dirname(__file__), "data")
    DATASET_NAME: str = f"teleop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    DATASET_OUTPUT_PATH = os.path.join(DATASET_DIR, DATASET_NAME)
    rr.init(DATASET_OUTPUT_PATH)
    rr.log("meta/env_name", ENV_NAME)

# is this environment bimanual?
BIMANUAL: bool = True
if "Solo" in ENV_NAME:
    BIMANUAL = False

# is this environment vision enabled?
VISUAL: bool = False
if "Vision" in ENV_NAME:
    VISUAL = True

# pick the appropriate urdf
if "SoloArm" in ENV_NAME:
    URDF_NAME: str = "stompy_tiny_solo_arm_glb"
if "DualArm" in ENV_NAME:
    URDF_NAME: str = "stompy_dual_arm_tiny_glb"
if "Torso" in ENV_NAME:
    URDF_NAME: str = "stompy_tiny_glb"

# web urdf is used by Vuer
URDF_LINK: str = (
    f"https://raw.githubusercontent.com/kscalelabs/webstompy/master/urdf/{URDF_NAME}/robot.urdf"
)
if LOG_DATASET:
    rr.log("meta/urdf_link", URDF_LINK)

# conversion functions between MuJoCo and Vuer axes
MJ_TO_VUER_AXES: NDArray = np.array([0, 2, 1], dtype=np.uint8)
MJ_TO_VUER_AXES_SIGN: NDArray = np.array([-1, 1, 1], dtype=np.int8)


def mj2vuer_pos(pos: NDArray) -> NDArray:
    return np.multiply(pos[MJ_TO_VUER_AXES], MJ_TO_VUER_AXES_SIGN)


def mj2vuer_orn(orn: NDArray) -> NDArray:
    return R.from_quat(orn).as_euler("xyz")


def vuer2mj_pos(pos: NDArray) -> NDArray:
    return np.multiply(pos[MJ_TO_VUER_AXES], MJ_TO_VUER_AXES_SIGN)


def vuer2mj_orn(orn: NDArray) -> NDArray:
    return R.from_euler("xyz", orn).as_quat()


env = gym.make(ENV_NAME)
# viewer.launch(env.unwrapped.mj_env)
mj_data = env.unwrapped.mj_env.physics.data
mj_model = env.unwrapped.mj_env.physics.model
env.reset()

# robot position and orientation
robot_pos: NDArray = mj_data.body("robot_root").xpos
robot_orn: NDArray = mj_data.body("robot_root").xquat
if LOG_DATASET:
    rr.log("meta/robot_pose_start", rr.Transform3D(pos=robot_pos, quat=robot_orn))

# cube and ee sites use visualizer geoms
cube_size: NDArray = mj_model.geom("cube").size
eer_site_size: NDArray = mj_model.site("hand_r_orn").size
eel_site_size: NDArray = mj_model.site("hand_l_orn").size

# global variables get updated by various async functions
async_lock = asyncio.Lock()
q: Dict[str, float] = env.unwrapped.q_dict
mj_q: NDArray = mj_data.qpos.copy()

# gobal variables for ee and gripper
eer_site_pos: NDArray = mj_data.mocap_pos[k.MOCAP_ID_R].copy()
eer_site_orn: NDArray = mj_data.mocap_quat[k.MOCAP_ID_R].copy()
grip_r: float = 0.0
if LOG_DATASET:
    rr.log("meta/eer_pose_start", rr.Transform3D(pos=eer_site_pos, quat=eer_site_orn))
if BIMANUAL:
    eel_site_pos: NDArray = mj_data.mocap_pos[k.MOCAP_ID_L].copy()
    eel_site_orn: NDArray = mj_data.mocap_quat[k.MOCAP_ID_L].copy()
    grip_l: float = 0.0
    if LOG_DATASET:
        rr.log(
            "meta/eel_pose_start", rr.Transform3D(pos=eel_site_pos, quat=eel_site_orn)
        )

# gobal variables for cube position and orientation
cube_pos: NDArray = mj_data.body("cube").xpos
cube_orn: NDArray = mj_data.body("cube").xquat
if LOG_DATASET:
    rr.log("meta/cube_pose_start", rr.Transform3D(pos=cube_pos, quat=cube_orn))


async def run_env() -> None:
    start_time = time.time()
    action = env.action_space.sample()
    async with async_lock:
        global cube_pos, cube_orn
        cube_pos = mj_data.body("cube").xpos
        cube_orn = mj_data.body("cube").xquat
        if LOG_DATASET:
            rr.log("data/cube_pose", rr.Transform3D(pos=cube_pos, quat=cube_orn))
        global eer_site_pos, eer_site_orn, grip_r
        action["eer_pos"] = eer_site_pos
        action["eer_orn"] = eer_site_orn
        action["grip_r"] = grip_r
        if LOG_DATASET:
            rr.log("data/eer_pose", rr.Transform3D(pos=eer_site_pos, quat=eer_site_orn))
            rr.log("data/grip_r", grip_r)
        if BIMANUAL:
            global eel_site_pos, eel_site_orn, grip_l
            action["eel_pos"] = eel_site_pos
            action["eel_orn"] = eel_site_orn
            action["grip_l"] = grip_l
            if LOG_DATASET:
                rr.log(
                    "data/eel_pose", rr.Transform3D(pos=eel_site_pos, quat=eel_site_orn)
                )
                rr.log("data/grip_l", grip_l)
        _q = mj_data.qpos[: env.unwrapped.q_len]
        if LOG_DATASET:
            rr.log("data/q_pos", _q)
        global q
        for i, val in enumerate(_q):
            joint_name = env.unwrapped.q_keys[i]
            q[joint_name] = val
    _, reward, terminated, _, _ = env.step(action)
    if LOG_DATASET:
        rr.log("data/reward", reward)
        rr.log("data/terminated", terminated)
    print(f"env step took {(time.time() - start_time) * 1000:.2f}ms")


# Vuer rendering params
MAX_FPS: int = 60
VUER_LIGHT_POS: NDArray = np.array([0, 2, 2])
VUER_LIGHT_INTENSITY: float = 10.0

# Vuer hand tracking and pinch detection params
HAND_FPS: int = 30
FINGER_INDEX: int = 9
FINGER_THUMB: int = 4
FINGER_MIDLE: int = 14
PINCH_OPEN: float = 0.10  # 10cm
PINCH_CLOSE: float = 0.01  # 1cm

app = Vuer()


@app.add_handler("HAND_MOVE")
async def hand_handler(event, _):
    # right hand
    rindex_pos: NDArray = np.array(event.value["rightLandmarks"][FINGER_INDEX])
    rthumb_pos: NDArray = np.array(event.value["rightLandmarks"][FINGER_THUMB])
    rpinch_dist: NDArray = np.linalg.norm(rindex_pos - rthumb_pos)
    # index finger to thumb pinch turns on tracking
    if rpinch_dist < PINCH_CLOSE:
        print("Pinch detected in right hand")
        # pinching with middle finger controls gripper
        rmiddl_pos: NDArray = np.array(event.value["rightLandmarks"][FINGER_MIDLE])
        rgrip_dist: float = np.linalg.norm(rthumb_pos - rmiddl_pos) / PINCH_OPEN
        # orientation is calculated from wrist rotation matrix
        wrist_rotation: NDArray = np.array(event.value["rightHand"])
        wrist_rotation = wrist_rotation.reshape(4, 4)[:3, :3]
        wrist_rotation = R.from_matrix(wrist_rotation).as_quat()
        async with async_lock:
            global eer_site_pos, eer_site_orn, grip_r
            eer_site_pos = vuer2mj_pos(rthumb_pos)
            print(f"goal_pos_eer {eer_site_pos}")
            eer_site_orn = vuer2mj_orn(wrist_rotation)
            print(f"goal_orn_eer {eer_site_orn}")
            grip_r = rgrip_dist
            print(f"right gripper at {grip_r}")
    if BIMANUAL:
        # left hand
        lindex_pos: NDArray = np.array(event.value["leftLandmarks"][FINGER_INDEX])
        lthumb_pos: NDArray = np.array(event.value["leftLandmarks"][FINGER_THUMB])
        lpinch_dist: NDArray = np.linalg.norm(lindex_pos - lthumb_pos)
        # index finger to thumb pinch turns on tracking
        if lpinch_dist < PINCH_CLOSE:
            print("Pinch detected in left hand")
            # pinching with middle finger controls gripper
            lmiddl_pos: NDArray = np.array(event.value["leftLandmarks"][FINGER_MIDLE])
            lgrip_dist: float = np.linalg.norm(lthumb_pos - lmiddl_pos) / PINCH_OPEN
            # orientation is calculated from wrist rotation matrix
            wrist_rotation: NDArray = np.array(event.value["leftHand"])
            wrist_rotation = wrist_rotation.reshape(4, 4)[:3, :3]
            wrist_rotation = R.from_matrix(wrist_rotation).as_quat()
            async with async_lock:
                global eel_site_pos, eel_site_orn, grip_l
                eel_site_pos = vuer2mj_pos(lthumb_pos)
                print(f"goal_pos_eel {eel_site_pos}")
                eel_site_orn = vuer2mj_orn(wrist_rotation)
                print(f"goal_orn_eel {eel_site_orn}")
                grip_l = lgrip_dist
                print(f"left gripper at {grip_l}")


@app.spawn(start=True)
async def main(session: VuerSession):
    global q
    global cube_pos, cube_orn
    global eer_site_pos, eer_site_orn
    global eel_site_pos, eel_site_orn
    session.upsert @ PointLight(intensity=VUER_LIGHT_INTENSITY, position=VUER_LIGHT_POS)
    session.upsert @ Hands(fps=HAND_FPS, stream=True, key="hands")
    await asyncio.sleep(0.1)
    session.upsert @ Urdf(
        src=URDF_LINK,
        jointValues=env.unwrapped.q_dict,
        position=mj2vuer_pos(robot_pos),
        rotation=mj2vuer_orn(robot_orn),
        key="robot",
    )
    session.upsert @ Box(
        args=cube_size,
        position=mj2vuer_pos(cube_pos),
        rotation=mj2vuer_orn(cube_orn),
        materialType="standard",
        material=dict(color="#ff0000"),
        key="cube",
    )
    session.upsert @ Capsule(
        args=eer_site_size,
        position=mj2vuer_pos(eer_site_pos),
        rotation=mj2vuer_orn(eer_site_orn),
        materialType="standard",
        material=dict(color="#0000ff"),
        key="eer-site",
    )
    if BIMANUAL:
        session.upsert @ Capsule(
            args=eel_site_size,
            position=mj2vuer_pos(eel_site_pos),
            rotation=mj2vuer_orn(eel_site_orn),
            materialType="standard",
            material=dict(color="#ff0000"),
            key="eel-site",
        )
    while True:
        await asyncio.gather(
            run_env(),  # ~1ms
            # update_image(),  # ~10ms
            asyncio.sleep(1 / MAX_FPS),  # ~16ms @ 60fps
        )
        async with async_lock:
            session.upsert @ Urdf(
                jointValues=q,
                position=mj2vuer_pos(robot_pos),
                rotation=mj2vuer_orn(robot_orn),
                key="robot",
            )
            session.upsert @ Box(
                position=mj2vuer_pos(cube_pos),
                rotation=mj2vuer_orn(cube_orn),
                key="cube",
            )
            session.upsert @ Capsule(
                position=mj2vuer_pos(eer_site_pos),
                rotation=mj2vuer_orn(eer_site_orn),
                key="eer-site",
            )
            if BIMANUAL:
                session.upsert @ Capsule(
                    position=mj2vuer_pos(eel_site_pos),
                    rotation=mj2vuer_orn(eel_site_orn),
                    key="eel-site",
                )
