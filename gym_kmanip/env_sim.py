from collections import OrderedDict as ODict
import os

from dm_env import TimeStep, StepType
from dm_control import mujoco
from dm_control.suite import base
from dm_control.rl import control
import numpy as np
from numpy.typing import NDArray

import gym_kmanip as k
from gym_kmanip.env_base import KManipEnv
from gym_kmanip.ik_mujoco import ik


# Task contains the mujoco logic, based on dm_control suite
class KManipTask(base.Task):
    def __init__(self, gym_env: KManipEnv, random=None):
        self.gym_env: KManipEnv = gym_env
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
        ctrl: NDArray = physics.data.ctrl.copy().astype(k.ACT_DTYPE)
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
        obs = ODict()
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
        for cam in self.gym_env.cameras:
            obs[cam.log_name] = physics.render(
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


class KManipEnvSim(control.Environment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def k_render(self, cam: k.Cam) -> NDArray:
        return self._physics.render(cam.h, cam.w, camera_id=cam.name)

    def k_reset(self, *args, **kwargs):
        ts: TimeStep = super().reset(*args, **kwargs)
        terminated: bool = False
        sim_time: float = 0.0
        return terminated, ts.reward, ts.discount, ts.observation, sim_time

    def k_step(self, *args, **kwargs):
        ts: TimeStep = super().step(*args, **kwargs)
        terminated: bool = ts.step_type == StepType.LAST
        sim_time: float = self._physics.data.time
        return terminated, ts.reward, ts.discount, ts.observation, sim_time


def new(gym_env: KManipEnv) -> KManipEnvSim:
    return KManipEnvSim(
        mujoco.Physics.from_xml_path(os.path.join(k.ASSETS_DIR, gym_env.mjcf_filename)),
        KManipTask(gym_env, random=gym_env.seed),
        control_timestep=k.CONTROL_TIMESTEP,
    )
