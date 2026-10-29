"""Recorder round trip: write 2 tiny episodes, load them back with LeRobotDataset."""

import json

import numpy as np
import pytest

from sdp.envs import PandaPickPlaceEnv
from sdp.expert.policy import ExpertPolicy
from sdp.record.lerobot_writer import ACTION_NAMES, STATE_NAMES, read_sidecar
from sdp.record.recorder import record_dataset, record_episode

MAX_STEPS = 12


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds") / "tiny"
    records = record_dataset(
        root,
        "local/sdp_test_tiny",
        ExpertPolicy(),
        seeds=[3, 4],
        source="scripted",
        max_episode_steps=MAX_STEPS,
        log=None,
    )
    return root, records


def test_info_is_lerobot_v30(tiny_dataset):
    root, _ = tiny_dataset
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["fps"] == 20
    assert info["total_episodes"] == 2
    assert info["features"]["observation.images.front"]["dtype"] == "video"
    assert info["features"]["observation.state"]["names"] == STATE_NAMES
    assert info["features"]["action"]["names"] == ACTION_NAMES


def test_sidecar_flags_every_episode(tiny_dataset):
    root, records = tiny_dataset
    side = read_sidecar(root)
    assert [r["episode_index"] for r in side] == [0, 1]
    assert [r["seed"] for r in side] == [3, 4]
    # 12 steps is far too short to finish: the episodes are kept and flagged as failures.
    assert all(r["success"] is False and r["label"] != "success" for r in side)
    assert all(r["length"] == MAX_STEPS for r in side)
    assert records[0]["source"] == "scripted"


def test_load_back_matches_what_was_recorded(tiny_dataset):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root, _ = tiny_dataset
    ds = LeRobotDataset("local/sdp_test_tiny", root=root)
    assert ds.num_episodes == 2
    assert ds.num_frames == 2 * MAX_STEPS

    env = PandaPickPlaceEnv()
    try:
        frames, _ = record_episode(env, ExpertPolicy(), seed=3, distribution="nominal")
    finally:
        env.close()

    for i in (0, 5, MAX_STEPS - 1):
        item = ds[i]
        assert item["episode_index"].item() == 0 and item["frame_index"].item() == i
        np.testing.assert_allclose(item["action"].numpy(), frames[i]["action"], atol=1e-6)
        np.testing.assert_allclose(
            item["observation.state"].numpy(), frames[i]["observation.state"], atol=1e-6
        )
        assert item["task"].startswith("Pick up the red cube")
        # Video is lossy (AV1); check orientation/channel order, not exact pixels.
        img = item["observation.images.front"].numpy().transpose(1, 2, 0)
        ref = frames[i]["observation.images.front"].astype(np.float32) / 255.0
        assert img.shape == (96, 96, 3)
        assert np.abs(img - ref).mean() < 0.05


def test_abs_target_mode_records_5d_named_targets(tmp_path):
    from sdp.envs.abs_target import ABS_ACTION_NAMES

    root = tmp_path / "abs"
    record_dataset(
        root,
        "local/sdp_test_abs",
        ExpertPolicy(),
        seeds=[5],
        source="scripted",
        max_episode_steps=MAX_STEPS,
        action_mode="abs_target",
        log=None,
    )
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["features"]["action"]["shape"] == [5]
    assert info["features"]["action"]["names"] == ABS_ACTION_NAMES
    assert read_sidecar(root)[0]["action_mode"] == "abs_target"
    assert json.loads((root / "meta" / "sdp_recording.json").read_text())["action_mode"] == (
        "abs_target"
    )
