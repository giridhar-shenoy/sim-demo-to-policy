"""smallbc: model shapes, spatial softmax, chunk labels, policy act() and checkpoints."""

import numpy as np
import pytest
import torch

from sdp.policies.smallbc import is_smallbc_checkpoint
from sdp.policies.smallbc.common import wrap_pi
from sdp.policies.smallbc.data import PROPRIO_DIM, TARGET_DIM, make_chunks
from sdp.policies.smallbc.model import SmallBC, SpatialSoftmax, load_checkpoint
from sdp.policies.smallbc.train import lr_lambda


def _batch(n=2, size=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    front = torch.randint(0, 256, (n, size, size, 3), dtype=torch.uint8, generator=g)
    wrist = torch.randint(0, 256, (n, size, size, 3), dtype=torch.uint8, generator=g)
    proprio = torch.randn(n, PROPRIO_DIM, generator=g)
    return front, wrist, proprio


@pytest.mark.parametrize(("size", "map_size"), [(96, 12), (128, 16)])
def test_model_shapes(size, map_size):
    torch.manual_seed(0)
    model = SmallBC(image_size=size, chunk=16).eval()
    assert model.enc_front.map_size == map_size  # strides 2-2-2-1
    assert model.enc_front.out_dim == 2 * 32
    out = model(*_batch(size=size))
    assert out.shape == (2, 16, TARGET_DIM)
    assert model.predict(*_batch(size=size)).shape == (2, 16, TARGET_DIM)


def test_spatial_softmax_range_and_peak():
    torch.manual_seed(0)
    ssm = SpatialSoftmax(in_ch=4, h=12, w=12, num_kp=3)
    xy = ssm(torch.randn(5, 4, 12, 12))
    assert xy.shape == (5, 6)
    assert torch.all(xy >= -1) and torch.all(xy <= 1)

    # One sharp peak at column 11 (x = +1), row 0 (y = -1) for every keypoint.
    with torch.no_grad():
        ssm.kp.weight.zero_()
        ssm.kp.bias.zero_()
        ssm.kp.weight[:, 0, 0, 0] = 1.0
    feat = torch.zeros(1, 4, 12, 12)
    feat[0, 0, 0, 11] = 50.0
    xy = ssm(feat).reshape(3, 2)
    torch.testing.assert_close(xy, torch.tensor([[1.0, -1.0]] * 3), atol=1e-4, rtol=0)


def test_chunk_labels_abs_and_rel():
    # One episode of 4 frames; absolute targets [x, y, z, yaw, g] after each step.
    target_abs = np.array(
        [[0.1, 0.0, 0.2, 3.0, 1], [0.2, 0.0, 0.2, 3.1, 1], [0.3, 0.0, 0.2, -3.1, -1],
         [0.4, 0.0, 0.2, -3.0, -1]],
        dtype=np.float32,
    )  # fmt: skip
    target_obs = np.array(
        [[0.0, 0.0, 0.2, 2.9], [0.1, 0.0, 0.2, 3.0], [0.2, 0.0, 0.2, 3.1], [0.3, 0.0, 0.2, -3.1]],
        dtype=np.float32,
    )
    ep_start = np.zeros(4, dtype=int)
    ep_len = np.full(4, 4)
    t = np.arange(4)
    abs_chunks = make_chunks(target_abs, target_obs, ep_start, ep_len, t, chunk=3, mode="abs")
    assert abs_chunks.shape == (4, 3, 5)
    np.testing.assert_allclose(abs_chunks[3], target_abs[[3, 3, 3]])  # clamped at episode end
    np.testing.assert_allclose(abs_chunks[1], target_abs[[1, 2, 3]])

    rel = make_chunks(target_abs, target_obs, ep_start, ep_len, t, chunk=3, mode="rel")
    np.testing.assert_allclose(
        rel[1, :, :3], target_abs[[1, 2, 3], :3] - target_obs[1, :3], atol=1e-6
    )
    # Yaw goes across the +-pi seam the short way: 3.0 -> -3.1 is +0.183 rad, not -6.1.
    np.testing.assert_allclose(rel[1, 1, 3], wrap_pi(-3.1 - 3.0), atol=1e-5)
    assert abs(rel[1, 1, 3]) < 0.2
    np.testing.assert_allclose(rel[..., 4], abs_chunks[..., 4])  # gripper unchanged
    with pytest.raises(ValueError):
        make_chunks(target_abs, target_obs, ep_start, ep_len, t, chunk=3, mode="delta")


def _save_checkpoint(path, model, mode="abs", step=123):
    torch.save(
        {"model": model.state_dict(), "model_cfg": model.cfg, "target_mode": mode, "step": step},
        path,
    )


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = SmallBC(chunk=8).eval()
    model.set_stats(np.full((8, 5), 0.5), np.full((8, 5), 2.0), np.zeros(19), np.ones(19))
    path = tmp_path / "ckpt.pt"
    _save_checkpoint(path, model)
    assert is_smallbc_checkpoint(path)
    assert not is_smallbc_checkpoint(tmp_path)  # ACT checkpoints are directories

    loaded, ckpt = load_checkpoint(path)
    assert ckpt["target_mode"] == "abs" and ckpt["step"] == 123
    assert not loaded.training
    batch = _batch()
    torch.testing.assert_close(loaded.predict(*batch), model.predict(*batch))
    torch.testing.assert_close(loaded.y_std, torch.full((8, 5), 2.0))


def test_policy_act_on_env_observation(tmp_path, make_env):
    from sdp.policies.smallbc.policy import SmallBCPolicy

    torch.manual_seed(0)
    path = tmp_path / "ckpt.pt"
    _save_checkpoint(path, SmallBC(chunk=16).eval())
    policy = SmallBCPolicy(path, n_exec=8, threads=1)
    assert (policy.image_size, policy.chunk_size, policy.n_action_steps) == (96, 16, 8)
    assert policy.action_mode == "abs_target"

    env = make_env()
    assert policy.wrap_env(env) is env
    obs, _ = env.reset(seed=10500)
    action = policy.act(obs, env)
    assert action.shape == (7,) and action.dtype == np.float32
    assert np.all(np.abs(action) <= 1.0)
    assert action[3] == 0.0 and action[4] == 0.0  # no roll / pitch
    assert action[6] in (-1.0, 1.0)  # binarised gripper
    assert len(policy.queue) == 7  # one of 8 executed actions popped
    policy.reset()
    assert len(policy.queue) == 0 and policy.t == 0


def test_lr_schedule_warms_up_then_decays():
    f = lr_lambda(warmup=10, total=110)
    assert f(0) == pytest.approx(0.1)
    assert f(9) == pytest.approx(1.0)
    assert f(60) == pytest.approx(0.5)
    assert f(110) == pytest.approx(0.0, abs=1e-12)
