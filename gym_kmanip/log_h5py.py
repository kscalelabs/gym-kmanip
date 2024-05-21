import os
from typing import Any, Dict

import h5py
from numpy.typing import NDArray

import gym_kmanip as k


def new(log_dir: str, info: Dict[str, Any]) -> h5py.Group:
    assert os.path.exists(log_dir), f"Directory {log_dir} does not exist"
    log_path: str = os.path.join(log_dir, f"episode_{info['episode']}.hdf5")
    f = h5py.File(log_path, "a")
    g = f.create_group("metadata")
    for key, value in info.items():
        try:
            g.attrs[key] = value
        except TypeError:
            print(f"Could not save {key}={value}")
    g = f.create_group("data")
    return g


def end(g: h5py.Group) -> None:
    if g is not None:
        g.file.close()


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
    # TODO: no way this is efficient, there is a "chunk" abstraction that
    # should probably be used here
    # https://docs.h5py.org/en/stable/high/dataset.html#chunked-storage
    step_group = g.create_group(f"step/{info['step']}")
    step_group.attrs["episode"] = info["episode"]
    step_group.attrs["sim_time"] = info["sim_time"]
    step_group.attrs["cpu_time"] = info["cpu_time"]
    step_group.attrs["reward"] = info["reward"]
    step_group.attrs["is_success"] = info["is_success"]
    action_group = step_group.create_group("action")
    for key, value in action.items():
        action_group.create_dataset(key, data=value)
    state_group = step_group.create_group("state")
    for key, value in observation.items():
        state_group.create_dataset(key, data=value)
