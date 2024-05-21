import os
from typing import Any, Dict, List

import h5py
from numpy.typing import NDArray

import gym_kmanip as k


def make_log(log_filename: str, data_dir_path: str) -> h5py.Group:
    log_path = os.path.join(data_dir_path, f"{log_filename}.hdf5")
    f = h5py.File(log_path, "a")
    g = f.create_group("state")
    return g


def log_metadata(g: h5py.Group, **kwargs) -> None:
    for key, value in kwargs.items():
        g.attrs[key] = value


def log_cam(g: h5py.Group, cam: k.Cam) -> None:
    pass


def log_step(
    g: h5py.Group,
    action: Dict[str, NDArray],
    observation: Dict[str, NDArray],
    info: Dict[str, Any],
) -> None:
    pass
