"""The ACT adapter infers the action mode from the checkpoint's action shape."""

import json

import pytest

from sdp.eval.act_policy import action_mode_from_config, action_mode_from_shape


def test_mode_from_shape():
    assert action_mode_from_shape(7) == "delta"
    assert action_mode_from_shape([7]) == "delta"
    assert action_mode_from_shape((5,)) == "abs_target"
    with pytest.raises(ValueError):
        action_mode_from_shape(6)


@pytest.mark.parametrize(("dim", "mode"), [(7, "delta"), (5, "abs_target")])
def test_mode_from_checkpoint_config(tmp_path, dim, mode):
    config = {"type": "act", "output_features": {"action": {"type": "ACTION", "shape": [dim]}}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert action_mode_from_config(tmp_path) == mode
