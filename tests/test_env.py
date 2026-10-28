import gymnasium as gym
import numpy as np
import pytest

import sdp.envs  # noqa: F401  (registers the env ids)
from sdp.envs.panda_pick_place import POSE_DISTRIBUTIONS
from sdp.expert import WaypointExpert


def rollout(env, actions, seed):
    obs, _ = env.reset(seed=seed)
    states, images = [obs["state"]], [obs["front"]]
    for a in actions:
        obs, *_ = env.step(a)
        states.append(obs["state"])
        images.append(obs["front"])
    return np.stack(states), np.stack(images)


def test_registered_and_spaces():
    env = gym.make("PandaPickPlace-v0")
    try:
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        space = env.action_space
        assert space.shape == (7,)
        np.testing.assert_array_equal(space.low, -np.ones(7))
        np.testing.assert_array_equal(space.high, np.ones(7))
        assert obs["front"].shape == obs["wrist"].shape == (96, 96, 3)
        assert obs["state"].shape == (15,)  # TCP pos 3 + quat 4 + gripper 1 + 7 joints
        assert not info["success"]
    finally:
        env.close()


def test_timing_constants(make_env):
    env = make_env(render_obs=False)
    assert env.model.opt.timestep == pytest.approx(1 / 500)
    assert env.n_substeps == 25  # 20 Hz control


def test_seed_determinism(make_env):
    rng = np.random.default_rng(123)
    actions = rng.uniform(-1, 1, size=(15, 7)).astype(np.float32)
    s1, i1 = rollout(make_env(), actions, seed=7)
    s2, i2 = rollout(make_env(), actions, seed=7)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(i1, i2)


def test_different_seeds_give_different_cube_poses(make_env):
    env = make_env(render_obs=False)
    poses = []
    for seed in range(5):
        env.reset(seed=seed)
        poses.append(env.cube_pose()[0][:2])
    assert len({tuple(np.round(p, 6)) for p in poses}) == 5


@pytest.mark.parametrize("dist", list(POSE_DISTRIBUTIONS))
def test_cube_pose_within_distribution(make_env, dist):
    env = make_env(render_obs=False, distribution=dist)
    box = POSE_DISTRIBUTIONS[dist]
    for seed in range(20):
        _, info = env.reset(seed=seed)
        x, y, yaw = info["episode"]["initial_cube_xy_yaw"]
        assert box["x"][0] <= x <= box["x"][1]
        assert box["y"][0] <= y <= box["y"][1]
        assert box["yaw"][0] <= yaw <= box["yaw"][1]


def test_actions_are_clipped_to_bounds(make_env):
    big = np.array([5.0, -5.0, 5.0, 0.0, 0.0, 5.0, 3.0])
    s_big, _ = rollout(make_env(render_obs=False), [big] * 5, seed=1)
    s_one, _ = rollout(make_env(render_obs=False), [np.clip(big, -1, 1)] * 5, seed=1)
    np.testing.assert_array_equal(s_big, s_one)


def test_tcp_target_stays_in_workspace(make_env):
    env = make_env(render_obs=False)
    env.reset(seed=0)
    down = np.array([0, 0, -1, 0, 0, 0, 1], dtype=np.float32)
    for _ in range(60):
        env.step(down)
    target, _ = env.target_pose()
    assert target[2] == pytest.approx(env.workspace_lo[2])
    assert env.tcp_pose()[0][2] > 0.0


def test_scripted_trajectory_succeeds_after_settling(make_env):
    env = make_env(render_obs=False)
    expert = WaypointExpert()
    env.reset(seed=0)
    first_candidate, info = None, {}
    for step in range(env.max_episode_steps):
        _, reward, terminated, truncated, info = env.step(expert.act(env))
        if info["is_success_candidate"] and first_candidate is None:
            first_candidate = step
        if terminated or truncated:
            break
    assert terminated and info["success"] and reward == 1.0
    # Success needs the placed state to hold for settle_steps consecutive steps (0.5 s).
    assert step - first_candidate + 1 >= env.settle_steps
    ep = info["episode"]
    assert ep["ever_reached"] and ep["ever_grasped"] and ep["ever_lifted"]
    assert not ep["dropped"] and not ep["collision"]


def test_no_success_if_gripper_never_opens(make_env):
    env = make_env(render_obs=False)
    expert = WaypointExpert()
    env.reset(seed=0)
    for _ in range(env.max_episode_steps):
        a = expert.act(env)
        if expert.phase_name in ("open", "retreat"):
            a[:6] = 0.0
            a[6] = -1.0  # keep holding the cube over the target
        _, _, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            break
    assert truncated and not info["success"]
    assert info["in_region"] or info["grasped"]


def test_render_rgb_array(make_env):
    env = make_env(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (240, 320, 3) and frame.dtype == np.uint8
    assert frame.std() > 5  # not a blank image
