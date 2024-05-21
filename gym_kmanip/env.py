from collections import OrderedDict
from datetime import datetime
import os
import time
from typing import Any, Callable, Dict, List
import uuid

from dm_control import mujoco
from dm_control.suite import base
from dm_control.rl import control
from dm_env import TimeStep, StepType
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray

import gym_kmanip as k
from gym_kmanip.ik_mujoco import ik


# Task contains the mujoco logic, based on dm_control suite
class KManipTask(base.Task):
    def __init__(self, gym_env: gym.Env, random=None):
        self.gym_env: gym.Env = gym_env
        super().__init__(random=random)

    def initialize_episode(self, physics):
        physics.reset()
        # TODO: jitter starting joint angles
        np.copyto(physics.data.qpos[: self.gym_env.q_len], self.gym_env.q_pos_home)
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
            grip_slider_r = k.EE_S_MIN + action["grip_r"] * k.EE_S_RANGE
            ctrl[self.gym_env.ctrl_id_r_grip[0]] = grip_slider_r
            ctrl[self.gym_env.ctrl_id_r_grip[1]] = grip_slider_r
        if "grip_l" in action:
            grip_slider_l = k.EE_S_MIN + action["grip_l"] * k.EE_S_RANGE
            ctrl[self.gym_env.ctrl_id_l_grip[0]] = grip_slider_l
            ctrl[self.gym_env.ctrl_id_l_grip[1]] = grip_slider_l
        if "eer_pos" in action:
            ctrl[self.gym_env.q_id_r_mask] = ik(
                physics,
                goal_pos=action["eer_pos"],
                goal_orn=action["eer_orn"],
                ee_site="eer_site_pos",
                q_mask=self.gym_env.q_id_r_mask,
                q_pos_home=self.gym_env.q_pos_home,
                q_pos_prev=self.gym_env.q_pos_prev,
            )
            self.gym_env.q_pos_prev = q_pos
        if "eel_pos" in action:
            ctrl[self.gym_env.q_id_l_mask] = ik(
                physics,
                goal_pos=action["eel_pos"],
                goal_orn=action["eel_orn"],
                ee_site="eel_site_pos",
                q_mask=self.gym_env.q_id_l_mask,
                q_pos_home=self.gym_env.q_pos_home,
                q_pos_prev=self.gym_env.q_pos_prev,
            )
            self.gym_env.q_pos_prev = q_pos
        if "q_pos" in action:
            ctrl[:] = action["q_pos"]
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
            obs["q_pos"] = obs["q_pos"][: self.gym_env.q_len]
        if "q_vel" in self.gym_env.obs_list:
            obs["q_vel"] = physics.data.qvel.copy()
            obs["q_vel"] = obs["q_vel"][: self.gym_env.q_len]
        if "cube_pos" in self.gym_env.obs_list:
            obs["cube_pos"] = physics.data.qpos[-7:-4].copy()
            # TODO: normalize to spawn range
        if "cube_orn" in self.gym_env.obs_list:
            obs["cube_orn"] = physics.data.qpos[-4:].copy()
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
        render_mode: str = "rgb_array",
        obs_list: List[str] = [
            "q_pos",  # joint positions
            "q_vel",  # joint velocities
            "cube_pos",  # cube position
            "cube_orn",  # cube orientation
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
            "q_pos",  # joint positions
        ],
        mjcf_filename: str = k.SOLO_ARM_MJCF,
        urdf_filename: str = k.SOLO_ARM_URDF,
        q_pos_home: NDArray = None,
        q_dict: OrderedDict[str, float] = None,
        q_keys: List[str] = None,
        q_id_r_mask: NDArray = None,
        q_id_l_mask: NDArray = None,
        ctrl_id_r_grip: NDArray = None,
        ctrl_id_l_grip: NDArray = None,
        log_prefix: str = "test",
        log_rerun: bool = False,
        log_h5py: bool = False,
    ):
        super().__init__()
        self.render_mode: str = render_mode
        self.seed: int = seed
        # counters for episode number and step number
        self.step_idx: int = 0
        self.episode_idx: int = 0
        # home position of the robot
        self.q_pos_home: NDArray = q_pos_home
        # used for smooth control
        self.q_pos_prev: NDArray = q_pos_home
        self.q_len: int = len(q_pos_home)
        # joint dictionaries and keys are needed for teleop
        self.q_dict: OrderedDict[str, float] = q_dict
        self.q_keys: List[str] = q_keys
        assert len(q_keys) == self.q_len, "q parameters do not match"
        # masks for the right and left arms
        self.q_id_r_mask: NDArray = q_id_r_mask
        self.q_id_l_mask: NDArray = q_id_l_mask
        # control ids for the grippers
        self.ctrl_id_r_grip: NDArray = ctrl_id_r_grip
        self.ctrl_id_l_grip: NDArray = ctrl_id_l_grip
        # optionally log using rerun (viz/debug) or h5py (data)
        self.log_rerun: bool = log_rerun
        self.log_h5py: bool = log_h5py
        # prefix, uuid, datetime are used to create log filename
        self.log_prefix: str = log_prefix
        self.log_filename: str = None
        if log_h5py or log_rerun:
            self.reset_log_filename()
        if log_h5py:
            from gym_kmanip.log_h5py import make_log, log_cam, log_metadata, log_step

            self.log_h5py_funcs: Dict[str, Callable] = {
                "make_log": make_log,
                "log_cam": log_cam,
                "log_metadata": log_metadata,
                "log_step": log_step,
            }
            self.h5py_grp = make_log(self.log_filename)
        if log_rerun:
            from gym_kmanip.log_rerun import make_log, log_cam, log_metadata, log_step

            self.log_rerun_funcs: Dict[str, Callable] = {
                "make_log": make_log,
                "log_cam": log_cam,
                "log_metadata": log_metadata,
                "log_step": log_step,
            }
            make_log(
                log_filename=self.log_filename,
                data_dir_path=k.DATA_DIR,
                obs_list=obs_list,
                act_list=act_list,
            )
        # robot descriptions
        self.mjcf_filename: str = mjcf_filename
        self.urdf_filename: str = urdf_filename
        # create dm_control task
        self.mj_env = control.Environment(
            mujoco.Physics.from_xml_path(os.path.join(k.ASSETS_DIR, mjcf_filename)),
            KManipTask(self, random=seed),
            control_timestep=k.CONTROL_TIMESTEP,
        )
        # TODO: eventually we wil need a self.real_robot_env, keeping the gymnasium env as the
        #       common interface to both the real robot and the mujoco sim robot
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
                low=np.array([-k.MAX_Q_VEL] * self.q_len),
                high=np.array([k.MAX_Q_VEL] * self.q_len),
                dtype=np.float64,
            )
        if "cube_pos" in obs_list:
            _obs_dict["cube_pos"] = spaces.Box(
                low=-1, high=1, shape=(3,), dtype=np.float64
            )
        if "cube_orn" in obs_list:
            _obs_dict["cube_orn"] = spaces.Box(
                low=-1, high=1, shape=(4,), dtype=np.float64
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
                if self.log_rerun:
                    self.log_rerun_funcs["log_cam"](cam)
                if self.log_h5py:
                    self.log_h5py_funcs["log_cam"](self.h5py_grp, cam)
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
        # information space
        self.info: Dict[str, Any] = {
            "step": self.step_idx,
            "episode": self.episode_idx,
            "is_success": False,
            "q_keys": self.q_keys,
            "obs_list": self.obs_list,
            "act_list": self.act_list,
        }

    def render(self):
        # TODO: when is render actually used?
        cam: k.Cam = k.CAMERAS["top"]
        return self.mj_env.physics.render(cam.h, cam.w, camera_id=cam.name)

    def reset_log_filename(self) -> str:
        self.log_filename = ".".join(
            self.log_prefix,
            str(uuid.uuid4())[:6],
            datetime.now().strftime(k.DATE_FORMAT),
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        ts: TimeStep = self.mj_env.reset()
        self.step_idx = 0
        self.episode_idx += 1
        self.info["step"] = (self.step_idx,)
        self.info["episode"] = (self.episode_idx,)
        self.info["is_success"] = False
        if self.log_h5py or self.log_rerun:
            self.reset_log_filename()
        if self.log_h5py:
            self.h5py_grp = self.log_h5py_funcs["make_log"](
                self.log_filename, k.DATA_DIR
            )
            self.log_h5py_funcs["log_metadata"](self.h5py_grp, **self.info)
        if self.log_rerun:
            self.log_rerun_funcs["make_log"](
                self.log_filename, k.DATA_DIR, self.obs_list, self.act_list
            )
            self.log_rerun_funcs["log_metadata"](**self.info)
        return ts.observation, self.info

    def step(self, action):
        ts: TimeStep = self.mj_env.step(action)
        self.step_idx += 1
        self.info["step"] = self.step_idx
        self.info["episode"] = self.episode_idx
        self.info["sim_time"] = self.mj_env.physics.data.time
        self.info["cpu_time"] = time.time()
        self.info["reward"] = ts.reward
        self.info["is_success"] = ts.reward > k.REWARD_SUCCESS_THRESHOLD
        terminated: bool = ts.step_type == StepType.LAST
        self.info["terminated"] = terminated
        if self.log_rerun:
            start_time = time.time()
            self.log_rerun_funcs["log_step"](action, ts.observation, self.info)
            print(f"logging w/ rerun took {(time.time() - start_time) * 1000:.2f}ms")
        if self.log_h5py:
            start_time = time.time()
            self.log_h5py_funcs["log_step"](
                self.h5py_grp, action, ts.observation, self.info
            )
            print(f"logging w/ h5py took {(time.time() - start_time) * 1000:.2f}ms")
        return ts.observation, ts.reward, terminated, False, self.info

    def close(self):
        # TODO: close out log files
        pass
