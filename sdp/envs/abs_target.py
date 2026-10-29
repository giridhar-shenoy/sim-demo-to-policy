"""Absolute TCP-target actions for the arm envs.

The env's native action is a per-step delta of the TCP target (7-D, normalised). This
module adds an equivalent absolute form, ``[target_x, target_y, target_z, target_yaw,
gripper]``: the integrated TCP target the env should hold after the step, with yaw about
world z relative to the home orientation.

* ``abs_target(env, gripper)``: read the current integrated target as a 5-D action.
  Called after ``env.step(delta)``, it gives the absolute action equivalent to ``delta``.
* ``abs_to_delta(env, a_abs)``: the inverse, ``clip((target - current) / max_step, -1, 1)``
  with the yaw difference wrapped to [-pi, pi). Roll and pitch deltas are 0: the expert
  never tilts the hand, and ``target_yaw_tilt`` exposes the tilt so a recorder can check.
* ``AbsoluteTargetWrapper``: a gymnasium wrapper whose ``step`` takes the 5-D absolute
  action and steps the env with the converted 7-D delta.

Why: with per-step deltas, the cube's position shows up only in the ~13 approach frames
of each demo, mostly as saturated pulses, and ACT regressed them to their mean. A
privileged-state ACT with the same architecture scored 0/50 with delta actions and 50/50
with absolute targets (see README, "Delta vs absolute actions").

The logic is ported from ``experiments/privileged/priv_common.py`` (the privileged-state
experiment of 30 Oct 2026), with the yaw wrap range corrected to [-pi, pi).
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np

from sdp.envs.base_mujoco_env import quat_error, quat_from_rotvec, quat_mul

ABS_ACTION_NAMES = ["target_x", "target_y", "target_z", "target_yaw", "gripper"]
ACTION_MODES = ("delta", "abs_target")


def wrap_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def target_yaw_tilt(env, quat: np.ndarray) -> tuple[float, np.ndarray]:
    """Yaw about world z and residual tilt (rotation vector) of ``quat`` relative to home.

    Uses the same yaw-then-tilt decomposition as ``MujocoArmEnv._limit_orientation``.
    """
    rel = quat_mul(quat, _conj(env._home_quat))
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, rel)
    mat = mat.reshape(3, 3)
    yaw = float(np.arctan2(mat[1, 0], mat[0, 0]))
    q_yaw = quat_from_rotvec(np.array([0.0, 0.0, yaw]))
    tilt = quat_error(quat_mul(_conj(q_yaw), rel), np.array([1.0, 0.0, 0.0, 0.0]))
    return yaw, tilt


def abs_target(env, gripper: float) -> np.ndarray:
    """The env's current integrated TCP target as ``[x, y, z, yaw, gripper]``."""
    pos, quat = env.target_pose()
    yaw, _ = target_yaw_tilt(env, quat)
    return np.array([pos[0], pos[1], pos[2], yaw, gripper], dtype=np.float32)


def abs_to_delta(env, a_abs) -> np.ndarray:
    """The 7-D normalised delta that moves the env's target to ``a_abs`` (clipped)."""
    a_abs = np.asarray(a_abs, dtype=np.float64)
    pos, quat = env.target_pose()
    yaw, _ = target_yaw_tilt(env, quat)
    delta = np.zeros(7, dtype=np.float32)
    delta[:3] = np.clip((a_abs[:3] - pos) / env.max_dpos, -1.0, 1.0)
    delta[5] = np.clip(wrap_pi(a_abs[3] - yaw) / env.max_drot, -1.0, 1.0)
    delta[6] = np.clip(a_abs[4], -1.0, 1.0)
    return delta


class AbsoluteTargetWrapper(gym.Wrapper):
    """Env whose actions are absolute TCP targets ``[x, y, z, yaw, gripper]``.

    ``last_delta`` holds the 7-D delta the most recent ``step`` sent to the env.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base = env.unwrapped
        lo = np.array([*base.workspace_lo, -base.max_yaw, -1.0], dtype=np.float32)
        hi = np.array([*base.workspace_hi, base.max_yaw, 1.0], dtype=np.float32)
        self.action_space = gym.spaces.Box(lo, hi, dtype=np.float32)
        self.last_delta: np.ndarray | None = None

    def step(self, action):
        self.last_delta = abs_to_delta(self.env.unwrapped, action)
        return self.env.step(self.last_delta)
