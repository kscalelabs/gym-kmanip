import os
from typing import Any, Dict

import h5py
from numpy.typing import NDArray

import gym_kmanip as k


def new(log_filename: str, data_dir_path: str) -> h5py.Group:
    log_path = os.path.join(data_dir_path, f"{log_filename}.hdf5")
    f = h5py.File(log_path, "a")
    g = f.create_group("state")
    return g


def end(g: h5py.Group) -> None:
    if g is not None:
        g.file.close()


def meta(g: h5py.Group, **kwargs) -> None:
    for key, value in kwargs.items():
        g.attrs[key] = value


def cam(g: h5py.Group, cam: k.Cam) -> None:
    group_name: str = cam.log_name
    g.create_group(group_name)
    g[group_name].attrs["resolution"] = [cam.w, cam.h]
    g[group_name].attrs["focal_length"] = cam.fl
    g[group_name].attrs["principal_point"] = cam.pp


def step(
    g: h5py.Group,
    action: Dict[str, NDArray],
    observation: Dict[str, NDArray],
    info: Dict[str, Any],
) -> None:
    step_group = g.create_group(f"step/{info['step']}")
    step_group.attrs["episode"] = info["episode"]
    step_group.attrs["sim_time"] = info["sim_time"]
    step_group.attrs["cpu_time"] = info["cpu_time"]
    step_group.attrs["reward"] = info["reward"]
    step_group.attrs["is_success"] = info["is_success"]
    action_group = step_group.create_group("action")
    for key, value in action.items():
        action_group.create_dataset(key, data=value)
    observation_group = step_group.create_group("observation")
    for key, value in observation.items():
        observation_group.create_dataset(key, data=value)
    info_group = step_group.create_group("info")
    for key, value in info.items():
        if isinstance(value, (int, float, str)):
            info_group.attrs[key] = value
        else:
            info_group.create_dataset(key, data=value)
