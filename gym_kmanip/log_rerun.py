import os
from typing import Any, Dict, List

from numpy.typing import NDArray
import rerun as rr
import rerun.blueprint as rrb

import gym_kmanip as k


def new(log_dir: str, info: Dict[str, Any]) -> None:
    assert os.path.exists(log_dir), f"Directory {log_dir} does not exist"
    # Blueprint is the GUI layout for ReRun
    time_series_views: List[rrb.SpaceView] = []
    if "q_pos" in info["obs_list"]:
        time_series_views.append(
            rrb.TimeSeriesView(origin="/state/q_pos", name="q_pos"),
        )
    if "q_vel" in info["obs_list"]:
        time_series_views.append(
            rrb.TimeSeriesView(origin="/state/q_vel", name="q_vel"),
        )
    if len(info["act_list"]) > 0:
        time_series_views.append(
            rrb.TimeSeriesView(origin="/action", name="action"),
        )
    camera_views: List[rrb.SpaceView] = []
    for cam in info["cameras"]:
        camera_views.append(rrb.Spatial2DView(origin=cam.log_name, name=cam.name))
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(
                    origin="/world",
                    name="scene",
                ),
                rrb.Horizontal(*camera_views),
            ),
            rrb.Vertical(*time_series_views),
        ),
    )
    rr.init("gym_kmanip", default_blueprint=blueprint)
    log_path: str = os.path.join(log_dir, f"episode_{info['episode']}.rrd")
    rr.save(log_path, default_blueprint=blueprint)
    rr.send_blueprint(blueprint=blueprint)
    # TODO: log metadata from info dict


def end():
    rr.disconnect()


def cam(cam: k.Cam) -> None:
    rr.log(
        f"world/camera/{cam.name}",
        rr.Pinhole(
            resolution=[cam.w, cam.h],
            focal_length=cam.fl,
            principal_point=cam.pp,
        ),
    )


def step(
    action: Dict[str, NDArray],
    observation: Dict[str, NDArray],
    info: Dict[str, Any],
) -> None:
    rr.set_time_seconds("sim_time", info["sim_time"])
    rr.set_time_seconds("cpu_time", info["cpu_time"])
    rr.set_time_sequence("episode", info["episode"])
    rr.set_time_sequence("step", info["step"])
    if "eer_pos" in action:
        rr.log(
            "world/eer",
            rr.Transform3D(
                translation=action["eer_pos"],
                rotation=rr.Quaternion(xyzw=action["eer_orn"][k.WXYZ_2_XYZW]),
            ),
        )
    if "eel_pos" in action:
        rr.log(
            "world/eel",
            rr.Transform3D(
                translation=action["eel_pos"],
                rotation=rr.Quaternion(xyzw=action["eel_orn"][k.WXYZ_2_XYZW]),
            ),
        )
    if "grip_r" in action:
        rr.log("action/grip_r", rr.Scalar(action["grip_r"]))
    if "grip_l" in action:
        rr.log("action/grip_l", rr.Scalar(action["grip_l"]))
    for i, key in enumerate(info["q_keys"]):
        rr.log(f"state/q_pos/{key}", rr.Scalar(observation["q_pos"][i]))
        rr.log(f"state/q_vel/{key}", rr.Scalar(observation["q_vel"][i]))
    if "cube_pos" in observation and "cube_orn" in observation:
        rr.log(
            "world/cube",
            rr.Transform3D(
                translation=observation["cube_pos"],
                rotation=rr.Quaternion(xyzw=observation["cube_orn"][k.WXYZ_2_XYZW]),
            ),
        )
    for cam in info["cameras"]:
        rr.log(cam.log_name, rr.Image(observation[cam.log_name]))
        # TODO: camera position and orientation
        # _quat: NDArray = np.empty(4)
        # mujoco.mju_mat2Quat(
        #     _quat, self.mj_env.physics.data.camera(cam.name).xmat
        # )
        # rr.log(
        #     f"world/{cam.name}",
        #     rr.Transform3D(
        #         translation=self.mj_env.physics.data.camera(cam.name).xpos,
        #         rotation=rr.Quaternion(xyzw=_quat[k.WXYZ_2_XYZW]),
        #     ),
        # )
