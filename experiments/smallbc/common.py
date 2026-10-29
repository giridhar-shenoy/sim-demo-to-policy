"""Shared helpers for the small-CNN BC experiment (smallbc).

* Environment setup: MUJOCO_GL=egl and SDP_MENAGERIE_DIR default to the values the
  repo's CHECKPOINTS.md uses, unless they are already set.
* ``target_yaw``: yaw of an integrated TCP target about world z, relative to the home
  orientation, using the same decomposition the env uses to clamp the target
  (``MujocoArmEnv._limit_orientation``). Also returns the residual tilt rotation vector.
* ``target_xyzyaw(env)``: the env's current integrated target as [x, y, z, yaw].
* ``abs_to_delta(env, target)``: absolute target [x, y, z, yaw, gripper] -> the env's
  7-D normalised delta action: ``clip((target - current integrated target) / max_step,
  -1, 1)`` with the yaw difference wrapped to (-pi, pi]. Roll and pitch deltas are 0,
  exactly like the scripted expert, so the target never tilts.

Nothing here reads the cube state: the policy side only uses the robot's own
integrated target (its commanded pose), which the delta-action interface needs anyway.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
_REPO = Path(__file__).resolve().parents[2]
if "SDP_MENAGERIE_DIR" not in os.environ and not (_REPO / "third_party" / "mujoco_menagerie").is_dir():
    os.environ["SDP_MENAGERIE_DIR"] = str(Path("~/work/menagerie").expanduser())

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sdp.envs.base_mujoco_env import quat_error, quat_from_rotvec, quat_mul  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = _REPO
TARGET_NAMES = ("x", "y", "z", "yaw", "gripper")
# Eval seeds already spent on the ACT runs' dev/test evals. Never evaluate on them here.
USED_TEST_SEEDS = range(10000, 10500)
DEV_SEEDS = range(10500, 10550)


def wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def target_yaw(env, quat: np.ndarray) -> tuple[float, np.ndarray]:
    """(yaw about world z relative to home, residual tilt rotation vector)."""
    rel = quat_mul(quat, _conj(env._home_quat))
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, rel)
    mat = mat.reshape(3, 3)
    yaw = float(np.arctan2(mat[1, 0], mat[0, 0]))
    q_yaw = quat_from_rotvec(np.array([0.0, 0.0, yaw]))
    tilt = quat_error(quat_mul(_conj(q_yaw), rel), np.array([1.0, 0.0, 0.0, 0.0]))
    return yaw, tilt


def target_xyzyaw(env) -> np.ndarray:
    pos, quat = env.target_pose()
    yaw, _ = target_yaw(env, quat)
    return np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float64)


def target_after_action(env, action: np.ndarray) -> np.ndarray:
    """Integrated target [x, y, z, yaw] the env would hold after ``action``.

    Mirrors ``MujocoArmEnv._integrate_target`` (clip, workspace clamp, orientation
    limit) without stepping the env.
    """
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    pos, quat = env.target_pose()
    pos = np.clip(pos + a[:3] * env.max_dpos, env.workspace_lo, env.workspace_hi)
    quat = env._limit_orientation(quat_mul(quat_from_rotvec(a[3:6] * env.max_drot), quat))
    yaw, _ = target_yaw(env, quat)
    return np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float64)


def abs_to_delta(env, target, binarize_gripper: bool = True) -> np.ndarray:
    """Absolute target [x, y, z, yaw, gripper] -> 7-D delta action for ``env.step``."""
    target = np.asarray(target, dtype=np.float64)
    cur = target_xyzyaw(env)
    a = np.zeros(7, dtype=np.float32)
    a[:3] = np.clip((target[:3] - cur[:3]) / env.max_dpos, -1.0, 1.0)
    a[5] = np.clip(wrap_pi(target[3] - cur[3]) / env.max_drot, -1.0, 1.0)
    g = float(target[4])
    a[6] = (1.0 if g >= 0.0 else -1.0) if binarize_gripper else float(np.clip(g, -1.0, 1.0))
    return a


def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    return float("nan")
