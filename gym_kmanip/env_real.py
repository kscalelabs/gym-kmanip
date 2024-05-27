import asyncio
from collections import OrderedDict as ODict
from typing import Any, Callable, Dict, List, OrderedDict
import time

import cv2
import numpy as np
from numpy.typing import NDArray

import gym_kmanip as k
from gym_kmanip.env_base import KManipEnv


async def get_image(cam: cv2.VideoCapture, kcam: k.Cam) -> NDArray[np.uint8]:
    start = time.time()
    if not cam.isOpened():
        print(f"camera {kcam.name} is not opened")
        return np.zeros((kcam.h, kcam.w, 3), dtype=np.uint8)
    ret, frame = cam.read()
    if ret:
        img = frame[:, :, k.BGR_TO_RGB]
    else:
        print("Failed to read frame")
    print(f"Time to update image: {time.time() - start}")
    return img


async def q_command():
    pass


class KManipEnvReal:

    def __init__(self, gym_env: KManipEnv, random=None):
        self.gym_env: KManipEnv = gym_env
        self.cv2_cams: OrderedDict[str, cv2.VideoCapture] = ODict()
        for cam in self.gym_env.cameras:
            print(f"Opening camera {cam.name} on device {cam.device_id}")
            self.cv2_cams[cam.name] = cv2.VideoCapture(cam.device_id)
            self.cv2_cams[cam.name].set(cv2.CAP_PROP_FRAME_WIDTH, cam.w)
            self.cv2_cams[cam.name].set(cv2.CAP_PROP_FRAME_HEIGHT, cam.h)
            self.cv2_cams[cam.name].set(cv2.CAP_PROP_FPS, cam.fps)

    def k_render(self, cam: k.Cam) -> NDArray:
        return np.empty((cam.h, cam.w, cam.c), dtype=np.uint8)

    def k_reset(self):
        # gather up the async tasks
        tasks: List[asyncio.Task] = []
        for cam in self.gym_env.cameras:
            tasks.append(get_image(self.cv2_cams[cam.name], cam))
        # add task for getting robot joint angles
        tasks.append(q_command())
        # run the async tasks
        obs_raw = asyncio.run(asyncio.gather(*tasks))
        # TODO: convert to required format
        observation = {}
        terminated: bool = False
        timestamp: float = 0.0
        discount = 0.0
        reward = 0.0
        return terminated, reward, discount, observation, timestamp

    def k_step(self, action: Dict[str, Any]):
        observation = {}
        terminated: bool = False
        timestamp: float = 0.0
        discount = 0.0
        reward = 0.0
        return terminated, reward, discount, observation, timestamp

    def k_close(self):
        for cam in self.cv2_cams.values():
            cam.release()
        cv2.destroyAllWindows()


def new(gym_env: KManipEnv) -> KManipEnvReal:
    return KManipEnvReal()
