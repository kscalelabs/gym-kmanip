from collections import OrderedDict
from datetime import datetime
import os
import time
from typing import List
import uuid

from dm_control import mujoco
from dm_control.suite import base
from dm_control.rl import control
from dm_env import TimeStep, StepType
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
import rerun as rr

import gym_kmanip as k
from gym_kmanip.ik_mujoco import ik
from gym_kmanip.rr_blueprint import make_blueprint


# Task contains the mujoco logic, based on dm_control suite
class KManipTask(base.Task):
    def __init__(self, gym_env: gym.Env, random=None):
        self.gym_env: gym.Env = gym_env
        super().__init__(random=random)

    def initialize_episode(self, physics):
        physics.reset()
        # TODO: jitter starting joint angles
        np.copyto(physics.data.qpos[: self.gym_env.q_len], self.gym_env.q_home)
        # randomize cube spawn
        cube_pos_x = self.random.uniform(*k.CUBE_SPAWN_RANGE_X)
        cube_pos_y = self.random.uniform(*k.CUBE_SPAWN_RANGE_Y)
        cube_pos_z = self.random.uniform(*k.CUBE_SPAWN_RANGE_Z)
        box_start_idx = physics.model.name2id("cube_joint", "joint")
        np.copyto(
            physics.data.qpos[box_start_idx : box_start_idx + 3],
            np.array([cube_pos_x, cube_pos_y, cube_pos_z]),
        )
        super().initialize_episode(physics)

    def before_step(self, action, physics):
        q_pos: NDArray = physics.data.qpos[:].copy()

        ctrl: NDArray = physics.data.ctrl.copy().astype(np.float32)
        if "eer_pos" in action:
            np.copyto(physics.data.mocap_pos[k.MOCAP_ID_R], action["eer_pos"])
        if "eer_orn" in action:
            np.copyto(physics.data.mocap_quat[k.MOCAP_ID_R], action["eer_orn"])
        if "eel_pos" in action:
            np.copyto(physics.data.mocap_pos[k.MOCAP_ID_L], action["eel_pos"])
        if "eel_orn" in action:
            np.copyto(physics.data.mocap_quat[k.MOCAP_ID_L], action["eel_orn"])
        if "grip_r" in action:
            grip_slider_r: float = k.EE_S_MIN + float(action["grip_r"]) * k.EE_S_RANGE
            ctrl[k.CTRL_ID_R_GRIP] = grip_slider_r
            ctrl[k.CTRL_ID_R_GRIP + 1] = grip_slider_r
        if "grip_l" in action:
            grip_slider_l: float = k.EE_S_MIN + float(action["grip_l"]) * k.EE_S_RANGE
            ctrl[k.CTRL_ID_L_GRIP] = grip_slider_l
            ctrl[k.CTRL_ID_L_GRIP + 1] = grip_slider_l
        if "eer_pos" in action:
            ctrl[k.Q_MASK_R] = ik(
                physics,
                goal_pos=action["eer_pos"],
                goal_orn=action["eer_orn"],
                ee_site="eer_site_pos",
                q_mask=k.Q_MASK_R,
                q_home=self.gym_env.qpos_prev,
            )
            self.gym_env.qpos_prev = q_pos
        if "eel_pos" in action:
            ctrl[k.Q_MASK_L] = ik(
                physics,
                goal_pos=action["eel_pos"],
                goal_orn=action["eel_orn"],
                ee_site="eel_site_pos",
                q_mask=k.Q_MASK_L,
                q_home=self.gym_env.qpos_prev,
            )
            self.gym_env.qpos_prev = q_pos
        # exponential filter for smooth control
        ctrl = k.CTRL_ALPHA * ctrl + (1 - k.CTRL_ALPHA) * physics.data.ctrl
        # TODO: debug why is this needed, try to remove
        physics.data.qpos[:] = q_pos
        physics.data.qvel[:] = 0
        physics.data.qacc[:] = 0
        super().before_step(ctrl, physics)

    def get_observation(self, physics) -> dict:
        obs = OrderedDict()
        if "q_pos" in self.gym_env.obs_list:
            obs["q_pos"] = physics.data.qpos.copy()
        if "q_vel" in self.gym_env.obs_list:
            obs["q_vel"] = physics.data.qvel.copy()

        for obs_name in self.gym_env.obs_list:
            if "camera" in obs_name:
                cam: k.Cam = k.CAMERAS[obs_name.split("/")[-1]]
                obs[obs_name] = physics.render(
                    height=cam.h,
                    width=cam.w,
                    camera_id=cam.name,
                ).copy()
        return obs

    def get_reward(self, physics) -> float:
        reward: float = 0
        # penalty for high velocity
        reward -= k.REWARD_VEL_PENALTY * np.linalg.norm(physics.data.qvel)
        # reward for gripper distance to cube
        cube_pos = physics.named.data.xpos["cube"]
        if "grip_l" in self.gym_env.act_list:
            grip_pos_l = physics.named.data.xpos["eel_site"]
            dist_l = np.linalg.norm(cube_pos - grip_pos_l)
            reward += k.REWARD_GRIP_DIST * (1 / (dist_l + 1e-6))
        if "grip_r" in self.gym_env.act_list:
            grip_pos_r = physics.named.data.xpos["eer_site"]
            dist_r = np.linalg.norm(cube_pos - grip_pos_r)
            reward += k.REWARD_GRIP_DIST * (1 / (dist_r + 1e-6))
        # contact detection for cube, hands, table
        touch_grip_l: bool = False
        touch_grip_r: bool = False
        touch_table: bool = False
        for i in range(physics.data.ncon):
            a = physics.model.id2name(physics.data.contact[i].geom1, "geom")
            b = physics.model.id2name(physics.data.contact[i].geom2, "geom")
            if a == "cube" and b == "left_gripper_finger":
                touch_grip_l = True
            if a == "cube" and b == "right_gripper_finger":
                touch_grip_r = True
            if a == "cube" and b == "table":
                touch_table = True
        if touch_grip_r or touch_grip_l:  # cube has been touched
            reward += k.REWARD_TOUCH_CUBE
            if not touch_table:  # cube is lifted
                reward += k.REWARD_LIFT_CUBE
        return reward


class KManipEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": k.FPS}

    def __init__(
        self,
        seed: int = 0,
        xml_filename: str = "_env_solo_arm.xml",
        render_mode: str = "rgb_array",
        obs_list: List[str] = [
            "q_pos",  # joint positions
            "q_vel",  # joint velocities
            "camera/top",  # overhead camera
            "camera/head",  # robot head camera
            "camera/grip_l",  # left gripper camera
            "camera/grip_r",  # right gripper camera
        ],
        act_list: List[str] = [
            "eel_pos",  # left end effector position
            "eel_orn",  # left end effector orientation
            "eer_pos",  # right end effector position
            "eer_orn",  # right end effector orientation
            "grip_l",  # left gripper
            "grip_r",  # right gripper
        ],
        q_home: NDArray = None,
        q_dict: OrderedDict[str, float] = None,
        q_keys: List[str] = None,
        log: bool = False,
    ):
        super().__init__()
        self.render_mode: str = render_mode
        self.seed: int = seed
        self.episode_step: int = 0
        self.q_home: NDArray = q_home
        self.qpos_prev: NDArray = q_home
        self.q_len: int = len(q_home)
        # joint dictionaries and keys are needed for teleop
        self.q_dict: OrderedDict[str, float] = q_dict
        self.q_keys: List[str] = q_keys
        # optionally log using rerun
        self.log: bool = log
        if log:
            log_uuid: str = str(uuid.uuid4())[:8]
            log_datetime: str = datetime.now().strftime("%Y%m%d%H%M")
            log_filename: str = f"{log_uuid}.{log_datetime}.rrd"
            log_path: str = os.path.join(k.DATA_DIR, log_filename)
            blueprint = make_blueprint(self.q_keys, obs_list, act_list)
            rr.init("gym_kmanip", default_blueprint=blueprint)
            rr.save(log_path, default_blueprint=blueprint)
            rr.send_blueprint(blueprint=blueprint)
            rr.log("meta/env_name", self.__class__.__name__)
            rr.log("meta/seed", seed)
        # create dm_control task
        self.mj_env = control.Environment(
            mujoco.Physics.from_xml_path(os.path.join(k.ASSETS_DIR, xml_filename)),
            KManipTask(self, random=seed),
            control_timestep=k.CONTROL_TIMESTEP,
        )
        # observation space
        self.obs_list = obs_list
        _obs_dict: OrderedDict[str, spaces.Space] = OrderedDict()
        if "q_pos" in obs_list:
            _obs_dict["q_pos"] = spaces.Box(
                low=np.array([-2 * np.pi] * self.q_len),
                high=np.array([2 * np.pi] * self.q_len),
                dtype=np.float64,
            )
        if "q_vel" in obs_list:
            _obs_dict["q_vel"] = spaces.Box(
                low=np.array([-np.inf] * self.q_len),
                high=np.array([np.inf] * self.q_len),
                dtype=np.float64,
            )
        for obs_name in obs_list:
            if "camera" in obs_name:
                cam: k.Cam = k.CAMERAS[obs_name.split("/")[-1]]
                _obs_dict[obs_name] = spaces.Box(
                    low=cam.low,
                    high=cam.high,
                    shape=(cam.h, cam.w, 3),
                    dtype=cam.dtype,
                )
                if self.log:
                    rr.log(
                        f"world/{cam.name}",
                        rr.Pinhole(
                            resolution=[cam.w, cam.h],
                            focal_length=cam.fl,
                            principal_point=cam.pp,
                        ),
                    )
        self.observation_space = spaces.Dict(_obs_dict)
        # action space
        self.act_list = act_list
        _action_dict: OrderedDict[str, spaces.Space] = OrderedDict()
        if "eel_pos" in act_list:
            _action_dict["eel_pos"] = spaces.Box(
                low=-1, high=1, shape=(3,), dtype=np.float32
            )
        if "eel_orn" in act_list:
            _action_dict["eel_orn"] = spaces.Box(
                low=-1, high=1, shape=(4,), dtype=np.float32
            )
        if "eer_pos" in act_list:
            _action_dict["eer_pos"] = spaces.Box(
                low=-1, high=1, shape=(3,), dtype=np.float32
            )
        if "eer_orn" in act_list:
            _action_dict["eer_orn"] = spaces.Box(
                low=-1, high=1, shape=(4,), dtype=np.float32
            )
        if "grip_l" in act_list:
            _action_dict["grip_l"] = spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32
            )
        if "grip_r" in act_list:
            _action_dict["grip_r"] = spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32
            )
        self.action_space = spaces.Dict(_action_dict)

    def render(self):
        # TODO: when is render actually used?
        cam: k.Cam = k.CAMERAS["top"]
        return self.mj_env.physics.render(cam.h, cam.w, camera_id=cam.name)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        ts: TimeStep = self.mj_env.reset()
        self.episode_step = 0
        ts.observation["q_pos"] = ts.observation["q_pos"][: self.q_len]
        ts.observation["q_vel"] = ts.observation["q_vel"][: self.q_len]
        info = {
            "cube_pos": ts.observation["q_pos"][-7:-4],
            "cube_orn": ts.observation["q_pos"][-4:],
            "is_success": False,
        }
        return ts.observation, info

    def step(self, action):
        ts: TimeStep = self.mj_env.step(action)
        self.episode_step += 1
        ts.observation["q_pos"] = ts.observation["q_pos"][: self.q_len]
        ts.observation["q_vel"] = ts.observation["q_vel"][: self.q_len]
        terminated: bool = ts.step_type == StepType.LAST
        info = {
            "cube_pos": ts.observation["q_pos"][-7:-4],
            "cube_orn": ts.observation["q_pos"][-4:],
            "is_success": ts.reward > k.REWARD_SUCCESS_THRESHOLD,
        }
        if self.log:
            start_time = time.time()
            rr.set_time_seconds("timestep", self.mj_env.physics.data.time)
            rr.set_time_sequence("episode_step", self.episode_step)
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
            for i, key in enumerate(self.q_keys):
                rr.log(f"state/q_pos/{key}", rr.Scalar(ts.observation["q_pos"][i]))
                rr.log(f"state/q_vel/{key}", rr.Scalar(ts.observation["q_vel"][i]))
            rr.log(
                "world/cube",
                rr.Transform3D(
                    translation=info["cube_pos"],
                    rotation=rr.Quaternion(xyzw=info["cube_orn"][k.WXYZ_2_XYZW]),
                ),
            )
            for obs_name in self.obs_list:
                if "camera" in obs_name:
                    cam: k.Cam = k.CAMERAS[obs_name.split("/")[-1]]
                    rr.log(f"camera/{cam.name}", rr.Image(ts.observation[obs_name]))
                    pos = self.mj_env.physics.data.camera(cam.name).xpos.copy()
                    orn = self.mj_env.physics.data.camera(cam.name).xmat.copy()
                    rr.log(
                        f"world/{cam.name}",
                        rr.Transform3D(
                            translation=pos,
                        ),
                    )
            print(f"logging took {(time.time() - start_time) * 1000:.2f}ms")
        return ts.observation, ts.reward, terminated, False, info

    def close(self):
        if self.log:
            rr.disconnect()
