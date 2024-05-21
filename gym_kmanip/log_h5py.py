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
    g.create_group(f"world/camera/{cam.name}")
    g[cam.name].attrs["resolution"] = [cam.w, cam.h]
    g[cam.name].attrs["focal_length"] = cam.fl
    g[cam.name].attrs["principal_point"] = cam.pp


def step(
    g: h5py.Group,
    action: Dict[str, NDArray],
    observation: Dict[str, NDArray],
    info: Dict[str, Any],
) -> None:
    pass
