"""Shared helpers for the privileged-state ACT sanity experiment.

* ``env_state(env)``: the cube's current pose as [x, y, z, sin(yaw), cos(yaw)].
  The target square centre is a class constant (0.55, 0.15), so it is not included.
* ``abs_target(env)``: the integrated TCP target as [x, y, z, yaw, gripper], where yaw is
  the rotation about world z of the target relative to the home orientation, computed
  with the same decomposition the env uses to clamp it (``_limit_orientation``).
* ``abs_to_delta(env, a)``: the inverse map, ``delta = clip((target - current)/max_step,
  -1, 1)`` with yaw wrapped to (-pi, pi]. Roll/pitch deltas are 0 (the expert never
  tilts; ``target_yaw_tilt`` lets the recorder check that).
* ``AbsoluteTargetWrapper``: a gymnasium wrapper whose ``step`` takes the 5-D absolute
  action and calls the env with the converted 7-D delta.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np

from sdp.envs.base_mujoco_env import quat_error, quat_from_rotvec, quat_mul

ENV_STATE_NAMES = ["cube_x", "cube_y", "cube_z", "cube_sin_yaw", "cube_cos_yaw"]
ABS_ACTION_NAMES = ["target_x", "target_y", "target_z", "target_yaw", "gripper"]


def wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def env_state(env) -> np.ndarray:
    pos, yaw = env.cube_pose()
    return np.array([pos[0], pos[1], pos[2], np.sin(yaw), np.cos(yaw)], dtype=np.float32)


def target_yaw_tilt(env, quat: np.ndarray) -> tuple[float, np.ndarray]:
    """Yaw about world z and residual tilt rotation vector, relative to home."""
    home = env._home_quat
    rel = quat_mul(quat, np.array([home[0], -home[1], -home[2], -home[3]]))
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, rel)
    mat = mat.reshape(3, 3)
    yaw = float(np.arctan2(mat[1, 0], mat[0, 0]))
    q_yaw = quat_from_rotvec(np.array([0.0, 0.0, yaw]))
    q_tilt = quat_mul(np.array([q_yaw[0], -q_yaw[1], -q_yaw[2], -q_yaw[3]]), rel)
    tilt = quat_error(q_tilt, np.array([1.0, 0.0, 0.0, 0.0]))
    return yaw, tilt


def abs_target(env, gripper: float) -> np.ndarray:
    pos, quat = env.target_pose()
    yaw, _ = target_yaw_tilt(env, quat)
    return np.array([pos[0], pos[1], pos[2], yaw, gripper], dtype=np.float32)


def abs_to_delta(env, a_abs) -> np.ndarray:
    a_abs = np.asarray(a_abs, dtype=np.float64)
    pos, quat = env.target_pose()
    yaw, _ = target_yaw_tilt(env, quat)
    d = np.zeros(7, dtype=np.float32)
    d[:3] = np.clip((a_abs[:3] - pos) / env.max_dpos, -1.0, 1.0)
    d[5] = np.clip(wrap_pi(a_abs[3] - yaw) / env.max_drot, -1.0, 1.0)
    d[6] = np.clip(a_abs[4], -1.0, 1.0)
    return d


class AbsoluteTargetWrapper(gym.Wrapper):
    """Env whose actions are absolute TCP targets [x, y, z, yaw, gripper]."""

    def __init__(self, env):
        super().__init__(env)
        lo = np.array([*env.workspace_lo, -env.max_yaw, -1.0], dtype=np.float32)
        hi = np.array([*env.workspace_hi, env.max_yaw, 1.0], dtype=np.float32)
        self.action_space = gym.spaces.Box(lo, hi, dtype=np.float32)

    def step(self, action):
        return self.env.step(abs_to_delta(self.env.unwrapped, action))
