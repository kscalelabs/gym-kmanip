import os
from typing import Any, Dict

import h5py
from numpy.typing import NDArray

import gym_kmanip as k
from gym_kmanip.log_base import LogBase

# Based off of this format, for compatibility with Huggingface's LeRobot
# https://github.com/tonyzhaozh/act/blob/main/record_sim_episodes.py


class LogH5py(LogBase):
    def __init__(self, log_dir: str):
        super().__init__(log_dir, log_type="h5py")
        self.f: h5py.File = None

    def reset(self, info: Dict[str, Any]):
        log_path: str = os.path.join(self.log_dir, f"episode_{info['episode']}.hdf5")
        f = h5py.File(log_path, "w", rdcc_nbytes=k.H5PY_CHUNK_SIZE_BYTES)
        f.attrs["sim"] = info["sim"]
        g = f.create_group("metadata")
        for key, value in info.items():
            try:
                g.attrs[key] = value
            except TypeError:
                print(f"Could not save {key}={value}")
        f.create_group("observations/images")
        f.create_dataset("observations/qpos", (k.MAX_EPISODE_STEPS, info["q_len"]))
        f.create_dataset("observations/qvel", (k.MAX_EPISODE_STEPS, info["q_len"]))
        f.create_dataset("action", (k.MAX_EPISODE_STEPS, info["a_len"]))
        self.f = f

    def reset_cam(self, cam: k.Cam):
        g = self.f.create_group(f"metadata/{cam.log_name}")
        g.attrs["resolution"] = [cam.w, cam.h]
        g.attrs["focal_length"] = cam.fl
        g.attrs["principal_point"] = cam.pp
        self.f.create_dataset(
            f"/observations/images/{cam.name}",
            (k.MAX_EPISODE_STEPS, cam.h, cam.w, cam.c),
            dtype=cam.dtype,
            chunks=(1, cam.h, cam.w, cam.c),
        )

    def step(
        self,
        action: Dict[str, NDArray],
        observation: Dict[str, NDArray],
        info: Dict[str, Any],
    ):
        id: int = info["step"] - 1
        self.f["action"][id] = action["grip_r"]
        self.f["observations/qpos"][id] = observation["q_pos"]
        self.f["observations/qvel"][id] = observation["q_vel"]
        for cam in info["cameras"]:
            self.f[f"/observations/images/{cam.name}"][id] = observation[cam.log_name]
        self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.close()