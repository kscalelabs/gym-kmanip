import os
from typing import Any, Dict

import h5py
from numpy.typing import NDArray

import gym_kmanip as k

# Based off of this format, for compatibility with Huggingface's LeRobot
# https://github.com/tonyzhaozh/act/blob/main/record_sim_episodes.py


def new(log_dir: str, info: Dict[str, Any]) -> h5py.File:
    assert os.path.exists(log_dir), f"Directory {log_dir} does not exist"
    log_path: str = os.path.join(log_dir, f"episode_{info['episode']}.hdf5")
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
    return f


def end(f: h5py.File) -> None:
    if f is not None:
        f.close()


def cam(f: h5py.File, cam: k.Cam) -> None:
    g = f.create_group(f"metadata/{cam.log_name}")
    g.attrs["resolution"] = [cam.w, cam.h]
    g.attrs["focal_length"] = cam.fl
    g.attrs["principal_point"] = cam.pp
    f.create_dataset(
        f"/observations/images/{cam.name}",
        (k.MAX_EPISODE_STEPS, cam.h, cam.w, cam.c),
        dtype=cam.dtype,
        chunks=(1, cam.h, cam.w, cam.c),
    )


def step(
    f: h5py.File,
    action: Dict[str, NDArray],
    observation: Dict[str, NDArray],
    info: Dict[str, Any],
) -> None:

    id: int = info["step"] - 1
    # pfb30 - what do we save as a action!
    f["action"][id] = observation["q_pos"][:f["action"].shape[1]]
    f["observations/qpos"][id] = observation["q_pos"]
    f["observations/qvel"][id] = observation["q_vel"]
    for cam in info["cameras"]:
        f[f"/observations/images/{cam.name}"][id] = observation[cam.log_name]
    f.flush()
