"""Absolute TCP-target actions: conversion, yaw wrapping and replay of the expert."""

import numpy as np
import pytest

from sdp.envs.abs_target import (
    ABS_ACTION_NAMES,
    AbsoluteTargetWrapper,
    abs_target,
    abs_to_delta,
    target_yaw_tilt,
    wrap_pi,
)
from sdp.envs.base_mujoco_env import quat_from_rotvec
from sdp.expert import WaypointExpert


def test_wrap_pi_range():
    assert wrap_pi(0.0) == 0.0
    assert wrap_pi(1.5 * np.pi) == pytest.approx(-0.5 * np.pi)
    assert wrap_pi(-1.5 * np.pi) == pytest.approx(0.5 * np.pi)
    assert wrap_pi(np.pi) == pytest.approx(-np.pi)  # [-pi, pi): +pi maps to -pi
    for a in np.linspace(-10, 10, 41):
        w = wrap_pi(a)
        assert -np.pi <= w < np.pi
        assert np.isclose(np.cos(w), np.cos(a)) and np.isclose(np.sin(w), np.sin(a))


class _FakeEnv:
    """Just enough of the env for abs_to_delta: a target pose and the step limits."""

    max_dpos = 0.02
    max_drot = 0.1
    _home_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def __init__(self, pos, yaw):
        self._pos = np.asarray(pos, dtype=float)
        self._quat = quat_from_rotvec(np.array([0.0, 0.0, yaw]))

    def target_pose(self):
        return self._pos.copy(), self._quat.copy()


def test_abs_to_delta_takes_the_short_way_round_in_yaw():
    env = _FakeEnv([0.5, 0.0, 0.1], yaw=3.1)
    assert target_yaw_tilt(env, env._quat)[0] == pytest.approx(3.1)
    # From +3.1 rad to -3.1 rad is +0.083 rad across the +-pi seam, not -6.2 rad.
    delta = abs_to_delta(env, [0.5, 0.0, 0.1, -3.1, 1.0])
    assert delta[5] == pytest.approx((2 * np.pi - 6.2) / env.max_drot, abs=1e-5)
    np.testing.assert_allclose(delta[[0, 1, 2, 3, 4]], 0.0, atol=1e-7)
    assert delta[6] == 1.0


def test_abs_to_delta_scales_and_clips_position():
    env = _FakeEnv([0.5, 0.0, 0.1], yaw=0.0)
    delta = abs_to_delta(env, [0.51, -0.5, 0.1, 0.0, -1.0])
    assert delta[0] == pytest.approx(0.5)  # 1 cm of a 2 cm step
    assert delta[1] == -1.0  # clipped
    assert delta[6] == -1.0


def test_wrapper_action_space(make_env):
    env = AbsoluteTargetWrapper(make_env(render_obs=False))
    assert env.action_space.shape == (len(ABS_ACTION_NAMES),) == (5,)
    base = env.unwrapped
    np.testing.assert_allclose(env.action_space.low[:3], base.workspace_lo)
    assert env.action_space.high[3] == pytest.approx(base.max_yaw)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_absolute_targets_replay_the_expert(make_env, seed):
    # Record the expert's deltas and the target each one leaves behind.
    ref = make_env(render_obs=False)
    expert = WaypointExpert()
    ref.reset(seed=seed)
    expert.reset()
    deltas, targets = [], []
    while True:
        a = expert.act(ref)
        _, _, terminated, truncated, info = ref.step(a)
        deltas.append(a)
        targets.append(abs_target(ref, gripper=float(a[6])))
        if terminated or truncated:
            break
    assert info["success"]

    # Replay only the absolute targets through the wrapper.
    env = AbsoluteTargetWrapper(make_env(render_obs=False))
    env.reset(seed=seed)
    replayed = 0
    for a_ref, a_abs in zip(deltas, targets, strict=True):
        _, _, terminated, truncated, replay_info = env.step(a_abs)
        replayed += 1
        np.testing.assert_allclose(env.last_delta, a_ref, atol=1e-5)
        if terminated or truncated:
            break
    assert replayed == len(deltas)
    assert replay_info["success"]
