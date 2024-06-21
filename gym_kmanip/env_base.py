from collections import OrderedDict as ODict
from datetime import datetime
import os
import time
from typing import Any, Callable, Dict, List, OrderedDict
import uuid

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray

import gym_kmanip as k
from gym_kmanip.log_base import LogBase
from gym_kmanip.log_h5py import LogH5py
from gym_kmanip.log_rerun import LogRerun


class KManipEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": k.FPS}

    def __init__(
        self,
        seed: int = 0,
        render_mode: str = "rgb_array",
        obs_type: k.ObservationType = k.ObservationType.state,
        cam_type: k.CameraType = None,
        control_type: k.ControlType = None,
        sim: bool = True,
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
        # camera properties
        self.cameras: List[k.Cam] = []
        if cam_type:
            # In python 3.12, I should be able to iterate the members of current instance (as in the first commented line).
            # But that doesn't work in 3.10. So I have to enumerate all possible camera 
            # values and check for their presence in the cam_type flag.
            # for cam_flag in list(cam_type):
            for cam_flag in k.CameraType:
                if cam_flag in cam_type:
                    cam: k.Cam = k.CAMERAS[cam_flag]
                    self.cameras.append(cam)
        # optionally log using rerun (viz/debug) or h5py (data)
        if log_h5py or log_rerun:
            _log_dir_name: str = "{}.{}.{}".format(
                log_prefix,
                str(uuid.uuid4())[:6],
                datetime.now().strftime(k.DATE_FORMAT),
            )
            self.log_dir = os.path.join(k.DATA_DIR, _log_dir_name)
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"Creating log dir at {self.log_dir}")

        self.loggers: List[LogBase] = []
        if log_h5py:
            self.loggers.append(LogH5py(self.log_dir))
        if log_rerun:
            self.loggers.append(LogRerun(self.log_dir))

        # robot descriptions
        self.mjcf_filename: str = mjcf_filename
        self.urdf_filename: str = urdf_filename

        self._build_observation_space(obs_type)
        self._build_action_space(control_type)
        # create either a sim or real environment
        self.sim: bool = sim
        if self.sim:
            from gym_kmanip.env_sim import new

            self.env = new(self)
        else:
            from gym_kmanip.env_real import new

            self.env = new(self)
        # information dict
        self.info: Dict[str, Any] = {
            "step": self.step_idx,
            "episode": self.episode_idx,
            "is_success": False,
            "q_keys": self.q_keys,
            "q_len": self.q_len,
            "a_len": self.action_len,
            "obs_list": self.obs_list,
            "act_list": self.act_list,
            "cameras": self.cameras,
            "sim": self.sim,
        }

    def _build_observation_space(self, obs_type: k.ObservationType):
        _obs_dict: OrderedDict[str, spaces.Space] = ODict()
        self.obs_list = ["q_pos", "q_vel"]
        _obs_dict["q_pos"] = spaces.Box(
            low=-1,
            high=1,
            shape=(self.q_len,),
            dtype=k.OBS_DTYPE,
        )
        _obs_dict["q_vel"] = spaces.Box(
            low=-1,
            high=1,
            shape=(self.q_len,),
            dtype=k.OBS_DTYPE,
        )

        if k.ObservationType.state in obs_type:
            self.obs_list.extend(["cube_pos", "cube_orn"])
            _obs_dict["cube_pos"] = spaces.Box(
                low=-1, high=1, shape=(3,), dtype=k.OBS_DTYPE
            )
            _obs_dict["cube_orn"] = spaces.Box(
                low=-1, high=1, shape=(4,), dtype=k.OBS_DTYPE
            )

        if k.ObservationType.image in obs_type:
            for cam in self.cameras:
                self.obs_list.append(cam.log_name) # TODO: We probably don't need this!
                _obs_dict[cam.log_name] = spaces.Box(
                    low=cam.low,
                    high=cam.high,
                    shape=(cam.h, cam.w, 3),
                    dtype=cam.dtype,
                )
        self.observation_space = spaces.Dict(_obs_dict)

    def _build_action_space(self, control_type: k.ControlType):
        self.act_list = []
        if k.ControlType.end_effector_right in control_type:
            self.act_list.extend(["eer_pos", "eer_orn", "grip_r"])
        if k.ControlType.end_effector_left in control_type:
            self.act_list.extend(["eel_pos", "eel_orn", "grip_l"])
        if k.ControlType.joints_right in control_type:
            self.act_list.extend(["q_pos_r", "grip_r"])
        if k.ControlType.joints_left in control_type:
            self.act_list.extend(["q_pos_l", "grip_l"])

        _action_dict: OrderedDict[str, spaces.Space] = ODict()
        if "eel_pos" in self.act_list:
            _action_dict["eel_pos"] = spaces.Box(
                low=-1, high=1, shape=(3,), dtype=k.ACT_DTYPE
            )
        if "eel_orn" in self.act_list:
            _action_dict["eel_orn"] = spaces.Box(
                low=-1, high=1, shape=(3,), dtype=k.ACT_DTYPE
            )
        if "eer_pos" in self.act_list:
            _action_dict["eer_pos"] = spaces.Box(
                low=-1, high=1, shape=(3,), dtype=k.ACT_DTYPE
            )
        if "eer_orn" in self.act_list:
            _action_dict["eer_orn"] = spaces.Box(
                low=-1, high=1, shape=(3,), dtype=k.ACT_DTYPE
            )
        if "grip_l" in self.act_list:
            _action_dict["grip_l"] = spaces.Box(
                low=-1, high=1, shape=(1,), dtype=k.ACT_DTYPE
            )
        if "grip_r" in self.act_list:
            _action_dict["grip_r"] = spaces.Box(
                low=-1, high=1, shape=(1,), dtype=k.ACT_DTYPE
            )
        if "q_pos_r" in self.act_list:
            _action_dict["q_pos_r"] = spaces.Box(
                low=-1,
                high=1,
                shape=(len(self.q_id_r_mask),),
                dtype=k.ACT_DTYPE,
            )
        if "q_pos_l" in self.act_list:
            _action_dict["q_pos_l"] = spaces.Box(
                low=-1,
                high=1,
                shape=(len(self.q_id_l_mask),),
                dtype=k.ACT_DTYPE,
            )
        self.action_space = spaces.Dict(_action_dict)
        self.action_len: int = len(self.action_space.spaces)

    def render(self):
        # TODO: when is this actually used?
        return self.env.k_render(k.CAMERAS[k.CameraType.top])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        terminated, reward, _, observation, sim_time = self.env.k_reset()
        self.step_idx = 0
        self.episode_idx += 1
        self.info["step"] = self.step_idx
        self.info["episode"] = self.episode_idx
        self.info["sim_time"] = sim_time
        self.info["cpu_time"] = time.time()
        self.info["reward"] = reward
        self.info["is_success"] = False
        self.info["terminated"] = terminated
        for logger in self.loggers:
            logger.reset(self.info)
            for cam in self.cameras:
                logger.reset_cam(cam)
        return observation, self.info

    def step(self, action):
        terminated, reward, _, observation, sim_time = self.env.k_step(action)
        self.step_idx += 1
        self.info["step"] = self.step_idx
        self.info["episode"] = self.episode_idx
        self.info["sim_time"] = sim_time
        self.info["cpu_time"] = time.time()
        self.info["reward"] = reward
        self.info["is_success"] = reward > k.REWARD_SUCCESS_THRESHOLD
        self.info["terminated"] = terminated
        for logger in self.loggers:
            start_time = time.time()
            logger.step(action, observation, self.info)
            print(f"logging w/ {logger.log_type} took {(time.time() - start_time) * 1000:.2f}ms")
        return observation, reward, terminated, False, self.info

    def close(self):
        for logger in self.loggers:
            logger.close()
        self.env.k_close()
        super().close()
