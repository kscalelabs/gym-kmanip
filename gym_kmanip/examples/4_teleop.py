import asyncio
from collections import OrderedDict as ODict
from typing import List, OrderedDict, Tuple
import time

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R
from vuer import Vuer, VuerSession
from vuer.schemas import Box, Hands, Plane, PointLight, Sphere, Urdf

import gym_kmanip as k

# choose your environment
# ENV_NAME: str = "KManipSoloArm"
# ENV_NAME: str = "KManipSoloArmQPos"
# ENV_NAME: str = "KManipSoloArmVision"
# ENV_NAME: str = "KManipDualArm"
# ENV_NAME: str = "KManipDualArmQPos"
# ENV_NAME: str = "KManipDualArmVision"
ENV_NAME: str = "KManipTorso"
# ENV_NAME: str = "KManipTorsoVision"
env = gym.make(
    ENV_NAME,
    # log_rerun=True,
    # log_h5py=True,
    # log_prefix="teleop",
)
env.reset()
mj_data = env.unwrapped.env.physics.data
mj_model = env.unwrapped.env.physics.model

# is this environment bimanual?
BIMANUAL: bool = True
if "Solo" in ENV_NAME:
    BIMANUAL = False

# is this environment vision enabled?
VISUAL: bool = False
if "Vision" in ENV_NAME:
    VISUAL = True

# Vuer requires a web link to the urdf for the headset
URDF_WEB_PATH: str = (
    f"https://raw.githubusercontent.com/kscalelabs/webstompy/master/urdf/{env.unwrapped.urdf_filename}"
)

# global variables get updated by various async functions
async_lock = asyncio.Lock()
q: OrderedDict = env.unwrapped.q_dict

# environment reset is controlled by gestures
reset: bool = False
# seconds to wait before next reset allowed, prevents accidental resets
RESET_BACKOFF: float = 1.0  # seconds
last_reset: float = time.time()

# sphere indicators for control
SPHERE_ARGS: List[float] = [0.02, 10, 10]

# gobal variables for hand pose and grip
hr_pos: NDArray = mj_data.mocap_pos[k.MOCAP_ID_R].copy()
hr_orn: NDArray = np.zeros(3)
# ee control action [-1, 1]
eer_pos: NDArray = np.zeros(3)
eer_orn: NDArray = np.zeros(3)
# grip action [-1, 1] or [open, closed]
grip_r: float = 0.0
if BIMANUAL:
    hl_pos: NDArray = mj_data.mocap_pos[k.MOCAP_ID_L].copy()
    hl_orn: NDArray = np.zeros(3)
    eel_pos: NDArray = np.zeros(3)
    eel_orn: NDArray = np.zeros(3)
    grip_l: float = 0.0
# NOTE: these are not .copy() and will be updated by mujoco in the background
cube_pos: NDArray = mj_data.body("cube").xpos
cube_orn: NDArray = mj_data.body("cube").xquat
cube_size: NDArray = mj_model.geom("cube").size
robot_pos: NDArray = mj_data.body("robot_root").xpos
robot_orn: NDArray = mj_data.body("robot_root").xquat
# table is easier to construct from base vuer plane primitize than load from stl
table_pos: NDArray = mj_data.body("table").xpos
TABLE_SIZE: NDArray = np.array([0.4, 0.8])
TABLE_ROT: NDArray = (
    R.from_euler("z", np.pi / 2) * R.from_euler("x", -np.pi / 2)
).as_euler("xyz")


async def run_env() -> None:
    start_time = time.time()
    action = env.action_space.sample()
    async with async_lock:
        global eer_pos, eer_orn, grip_r
        action["eer_pos"] = eer_pos
        action["eer_orn"] = eer_orn
        action["grip_r"] = grip_r
        if BIMANUAL:
            global eel_pos, eel_orn, grip_l
            action["eel_pos"] = eel_pos
            action["eel_orn"] = eel_orn
            action["grip_l"] = grip_l
        _q = mj_data.qpos[: env.unwrapped.q_len]
        global q
        for i, val in enumerate(_q):
            joint_name = env.unwrapped.q_keys[i]
            q[joint_name] = val
    env.step(action)
    print(f"env step took {(time.time() - start_time) * 1000:.2f}ms")
    async with async_lock:
        global reset, last_reset
        if reset and time.time() - last_reset > RESET_BACKOFF:
            print("environment reset")
            env.reset()
            reset = False
            last_reset = time.time()


# Vuer rendering params
MAX_FPS: int = 60
VUER_LIGHT_POS: NDArray = np.array([0, 2, 2])
VUER_LIGHT_INTENSITY: float = 10.0

# Vuer hand tracking and pinch detection params
HAND_FPS: int = 30
FINGER_INDEX: int = 9
FINGER_THUMB: int = 4
FINGER_MIDLE: int = 14
FINGER_PINKY: int = 24
PINCH_OPEN: float = 0.10  # 10cm
PINCH_CLOSE: float = 0.01  # 1cm

app = Vuer()


@app.add_handler("HAND_MOVE")
async def hand_handler(event, _):
    global hr_pos, hr_orn, eer_pos, eer_orn, grip_r, reset
    # right hand
    rindex_pos: NDArray = np.array(event.value["rightLandmarks"][FINGER_INDEX])
    rthumb_pos: NDArray = np.array(event.value["rightLandmarks"][FINGER_THUMB])
    # orientation is calculated from wrist rotation matrix
    rwrist_orn: NDArray = np.array(event.value["rightHand"])
    rwrist_orn = rwrist_orn.reshape(4, 4)[:3, :3]
    rwrist_orn = R.from_matrix(rwrist_orn).as_euler("xyz")
    # index finger to thumb pinch turns on tracking
    rpinch_dist: NDArray = np.linalg.norm(rindex_pos - rthumb_pos)
    if rpinch_dist < PINCH_CLOSE:
        print("Pinch detected in right hand")
        # pinching with middle finger controls gripper
        rmiddl_pos: NDArray = np.array(event.value["rightLandmarks"][FINGER_MIDLE])
        rgrip_dist: float = np.linalg.norm(rthumb_pos - rmiddl_pos) / PINCH_OPEN
        # async with async_lock:
        #     global hr_pos, hr_orn, eer_pos, eer_orn, grip_r
        eer_pos = np.clip(hr_pos - rthumb_pos, -1, 1)
        print(f"eer_pos action {eer_pos}")
        eer_orn = np.clip(hr_orn - rwrist_orn, -1, 1)
        print(f"eer_orn action {eer_orn}")
        grip_r = rgrip_dist
        print(f"grip_r action {grip_r}")
    # pinky to thumb resets the environment (starts recording new episode)
    rpinky_pos: NDArray = np.array(event.value["rightLandmarks"][FINGER_PINKY])
    rpinky_dist: NDArray = np.linalg.norm(rthumb_pos - rpinky_pos)
    if rpinky_dist < PINCH_CLOSE:
        print("Reset detected in right hand")
        # async with async_lock:
        #     global reset, hr_pos, hr_orn
        reset = True
        # reset the hand indicator to the pinky
        hr_pos = rthumb_pos
        hr_orn = rwrist_orn
    if BIMANUAL:
        global hl_pos, hl_orn, eel_pos, eel_orn, grip_l
        # left hand
        lindex_pos: NDArray = np.array(event.value["leftLandmarks"][FINGER_INDEX])
        lthumb_pos: NDArray = np.array(event.value["leftLandmarks"][FINGER_THUMB])
        lpinch_dist: NDArray = np.linalg.norm(lindex_pos - lthumb_pos)
        # orientation is calculated from wrist rotation matrix
        lwrist_orn: NDArray = np.array(event.value["leftHand"])
        lwrist_orn = lwrist_orn.reshape(4, 4)[:3, :3]
        lwrist_orn = R.from_matrix(lwrist_orn).as_euler("xyz")
        # index finger to thumb pinch turns on tracking
        if lpinch_dist < PINCH_CLOSE:
            print("Pinch detected in left hand")
            # pinching with middle finger controls gripper
            lmiddl_pos: NDArray = np.array(event.value["leftLandmarks"][FINGER_MIDLE])
            lgrip_dist: float = np.linalg.norm(lthumb_pos - lmiddl_pos) / PINCH_OPEN
            # async with async_lock:
            #     global hl_pos, hl_orn, eel_pos, eel_orn, grip_l
            eel_pos = np.clip(hl_pos - lthumb_pos, -1, 1)
            print(f"eel_pos action {eel_pos}")
            eel_orn = np.clip(hl_orn - lwrist_orn, -1, 1)
            print(f"eel_orn action {eel_orn}")
            grip_l = lgrip_dist
            print(f"grip_l action {grip_l}")
        # pinky to thumb resets the environment (starts recording new episode)
        lpinky_pos: NDArray = np.array(event.value["leftLandmarks"][FINGER_PINKY])
        lpinky_dist: NDArray = np.linalg.norm(lthumb_pos - lpinky_pos)
        if lpinky_dist < PINCH_CLOSE:
            print("Reset detected in left hand")
            # async with async_lock:
            #     global hl_pos
            # reset the hand indicator
            hl_pos = lthumb_pos
            hl_orn = lwrist_orn


@app.spawn(start=True)
async def main(session: VuerSession):
    global q
    global cube_pos, cube_orn
    global hr_pos, hr_orn
    global hl_pos, hl_orn
    session.upsert @ PointLight(intensity=VUER_LIGHT_INTENSITY, position=VUER_LIGHT_POS)
    session.upsert @ Hands(fps=HAND_FPS, stream=True, key="hands")
    await asyncio.sleep(0.1)
    session.upsert @ Urdf(
        src=URDF_WEB_PATH,
        jointValues=env.unwrapped.q_dict,
        position=k.mj2vuer_pos(robot_pos),
        rotation=k.mj2vuer_orn(robot_orn),
        key="robot",
    )
    session.upsert @ Box(
        args=cube_size,
        position=k.mj2vuer_pos(cube_pos),
        rotation=k.mj2vuer_orn(cube_orn),
        materialType="standard",
        material=dict(color="#ff0000"),
        key="cube",
    )
    session.upsert @ Plane(
        args=TABLE_SIZE,
        position=k.mj2vuer_pos(table_pos),
        rotation=TABLE_ROT,
        materialType="standard",
        material=dict(color="#cbc1ae"),
        key="table",
    )
    session.upsert @ Sphere(
        args=SPHERE_ARGS,
        position=hr_pos,
        rotation=hr_orn,
        materialType="standard",
        material=dict(color="#0000ff"),
        key="hand_r",
    )
    if BIMANUAL:
        session.upsert @ Sphere(
            args=SPHERE_ARGS,
            position=hl_pos,
            rotation=hl_orn,
            materialType="standard",
            material=dict(color="#ff0000"),
            key="hand_l",
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
                position=k.mj2vuer_pos(robot_pos),
                rotation=k.mj2vuer_orn(robot_orn),
                key="robot",
            )
            session.upsert @ Box(
                position=k.mj2vuer_pos(cube_pos),
                rotation=k.mj2vuer_orn(cube_orn),
                key="cube",
            )
            session.upsert @ Sphere(
                position=hr_pos,
                rotation=hr_orn,
                key="hand_r",
            )
            if BIMANUAL:
                session.upsert @ Sphere(
                    position=hl_pos,
                    rotation=hl_orn,
                    key="hand_l",
                )
