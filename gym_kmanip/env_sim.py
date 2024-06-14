from collections import OrderedDict as ODict
import os

from dm_env import TimeStep, StepType
from dm_control import mujoco
from dm_control.suite import base
from dm_control.rl import control
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

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
        np.copyto(physics.data.ctrl[: self.gym_env.q_len], self.gym_env.q_pos_home)
        # TODO: random start velocity
        np.copyto(physics.data.qvel[: self.gym_env.q_len], np.zeros(self.gym_env.q_len))
        # randomize cube spawn
        box_start_idx = physics.model.name2id("cube_joint", "joint")
        np.copyto(
            physics.data.qpos[box_start_idx : box_start_idx + 3],
            np.random.uniform(k.CUBE_SPAWN_RANGE[:, 0], k.CUBE_SPAWN_RANGE[:, 1]),
        )
        super().initialize_episode(physics)

    def before_step(self, action, physics):
        q_pos: NDArray = physics.data.qpos.copy()
        ctrl: NDArray = physics.data.ctrl.copy().astype(k.ACT_DTYPE)
        if "grip_r" in action:
            # grip action will be [-1, 1], need to undo that here
            grip_r: float = action["grip_r"] * k.EE_S_DELTA
            # print(f"grip_r delta: {grip_r}")
            grip_r += physics.data.qpos[self.gym_env.ctrl_id_r_grip[0]].copy()
            # print(f"grip_r raw: {grip_r}")
            # clip to range
            grip_r = np.clip(grip_r, k.EE_S_MIN, k.EE_S_MAX)
            # print(f"grip_r clipped: {grip_r}")
            ctrl[self.gym_env.ctrl_id_r_grip[0]] = grip_r
            ctrl[self.gym_env.ctrl_id_r_grip[1]] = grip_r
        if "grip_l" in action:
            # grip action will be [-1, 1], need to undo that here
            grip_l: float = action["grip_l"] * k.EE_S_DELTA
            grip_l += physics.data.qpos[self.gym_env.ctrl_id_l_grip[0]].copy()
            # clip to range
            grip_l = np.clip(grip_l, k.EE_S_MIN, k.EE_S_MAX)
            ctrl[self.gym_env.ctrl_id_l_grip[0]] = grip_l
            ctrl[self.gym_env.ctrl_id_l_grip[1]] = grip_l
        if "eer_pos" in action:
            # ee_pos will be normalized to [-1, 1], need to undo that here
            eer_pos: NDArray = action["eer_pos"] * k.EE_POS_DELTA
            eer_pos += physics.data.site("eer_site_pos").xpos.copy()
            np.copyto(physics.data.mocap_pos[k.MOCAP_ID_R], eer_pos)
            # ee_orn will be [-1, 1] in euler angles, need to add to current orientation
            eer_orn: NDArray = action["eer_orn"] * k.EE_ORN_DELTA
            eer_orn += R.from_matrix(physics.data.site("eer_site_pos").xmat.reshape(3, 3)).as_euler("xyz")
            # TODO: clip to ee orn limits
            eer_orn = R.from_euler("xyz", eer_orn).as_quat()[k.XYZW_2_WXYZ]
            np.copyto(physics.data.mocap_quat[k.MOCAP_ID_R], eer_orn)
            ctrl[self.gym_env.q_id_r_mask] = ik(
                physics,
                goal_pos=eer_pos,
                goal_orn=eer_orn,
                ee_site="eer_site_pos",
                q_mask=self.gym_env.q_id_r_mask,
                q_pos_home=self.gym_env.q_pos_home,
                q_pos_prev=q_pos,
            )
        if "eel_pos" in action:
            # pos will be normalized to [-1, 1], need to undo that here
            eel_pos: NDArray = action["eel_pos"] * k.EE_POS_DELTA
            eel_pos += physics.data.site("eel_site_pos").xpos.copy()
            np.copyto(physics.data.mocap_pos[k.MOCAP_ID_L], eel_pos)
            # orn will be [-1, 1], potentially invalid quaternion, need to normalize
            eel_orn: NDArray = action["eel_orn"] * k.EE_ORN_DELTA
            eel_orn += R.from_matrix(physics.data.site("eel_site_pos").xmat.reshape(3, 3)).as_euler("xyz")
            # TODO: clip to ee orn limits
            eel_orn = R.from_euler("xyz", eel_orn).as_quat()[k.XYZW_2_WXYZ]
            np.copyto(physics.data.mocap_quat[k.MOCAP_ID_L], eel_orn)
            ctrl[self.gym_env.q_id_l_mask] = ik(
                physics,
                goal_pos=eel_pos,
                goal_orn=eel_orn,
                ee_site="eel_site_pos",
                q_mask=self.gym_env.q_id_l_mask,
                q_pos_home=self.gym_env.q_pos_home,
                q_pos_prev=q_pos,
            )
        if "q_pos_r" in action:
            ctrl[self.gym_env.q_id_r_mask] = q_pos[self.gym_env.q_id_r_mask] + action["q_pos_r"] * k.Q_POS_DELTA
        if "q_pos_l" in action:
            ctrl[self.gym_env.q_id_l_mask] = q_pos[self.gym_env.q_id_l_mask] + action["q_pos_l"] * k.Q_POS_DELTA
        # exponential filter for smooth control
        # print(f"ctrl raw: {ctrl}")
        ctrl = k.CTRL_ALPHA * ctrl + (1 - k.CTRL_ALPHA) * physics.data.ctrl.copy()
        # print(f"ctrl filtered: {ctrl}")
        # pfb30 - actual change
        # physics.data.qpos[self.gym_env.q_id_r_mask] = action[self.gym_env.q_id_r_mask] 
        ctrl[:] = action
        print(ctrl)
        super().before_step(ctrl, physics)

    def get_observation(self, physics) -> dict:
        obs = ODict()
        if "q_pos" in self.gym_env.obs_list:
            q_pos: NDArray = physics.data.qpos[: self.gym_env.q_len].copy()
            # normalize to joint position limits
            q_pos -= physics.model.jnt_range[:, 0][: self.gym_env.q_len]
            q_pos /= (
                physics.model.jnt_range[:, 1][: self.gym_env.q_len]
                - physics.model.jnt_range[:, 0][: self.gym_env.q_len]
            )
            # clip to [-1, 1]
            q_pos = np.clip(q_pos, -1, 1)
            obs["q_pos"] = q_pos
        if "q_vel" in self.gym_env.obs_list:
            q_vel: NDArray = physics.data.qvel.copy()
            # normalize to joint velocity limits
            q_vel /= k.MAX_Q_VEL
            # clip to [-1, 1]
            q_vel = np.clip(q_vel, -1, 1)
            obs["q_vel"] = q_vel[: self.gym_env.q_len]
            # pfb30 
            obs["q_vel"] = physics.data.ctrl
        if "cube_pos" in self.gym_env.obs_list:
            cube_pos: NDArray = physics.data.qpos[-7:-4].copy()
            # normalize cube pos to spawn range
            cube_pos -= k.CUBE_SPAWN_RANGE[:, 0]
            cube_pos /= k.CUBE_SPAWN_RANGE[:, 1] - k.CUBE_SPAWN_RANGE[:, 0]
            # clip to [-1, 1]
            cube_pos = np.clip(cube_pos, -1, 1)
            obs["cube_pos"] = cube_pos
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
            reward += k.REWARD_GRIP_DIST * (1 / (dist_l + k.EPSILON))
        if "grip_r" in self.gym_env.act_list:
            grip_pos_r = physics.named.data.xpos["eer_site"]
            dist_r = np.linalg.norm(cube_pos - grip_pos_r)
            reward += k.REWARD_GRIP_DIST * (1 / (dist_r + k.EPSILON))
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
    
    def k_close(self):
        pass


def new(gym_env: KManipEnv) -> KManipEnvSim:
    return KManipEnvSim(
        mujoco.Physics.from_xml_path(os.path.join(k.ASSETS_DIR, gym_env.mjcf_filename)),
        KManipTask(gym_env, random=gym_env.seed),
        control_timestep=k.CONTROL_TIMESTEP,
    )
