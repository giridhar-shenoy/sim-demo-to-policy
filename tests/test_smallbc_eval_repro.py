"""`sdp eval` on a smallbc checkpoint reproduces a stored result to 1e-6.

`sdp eval` is the one smallbc evaluation path: `scripts/eval_all.sh` calls it for the
official test seeds, and `experiments/smallbc/` re-exports the same package code. This
test runs it on 3 episodes (eval seeds 10500-10502, nominal) with a small seeded random
model saved inside the test, and compares the per-episode CSV row and the actions sent to
the env with `tests/data/smallbc_eval_expected.json`.

The model reads only the measured gripper opening (`obs["state"][7]`): its weights are
zero except one path from that input through each layer, so every output is a single
product, rounded exactly, and its 16-step chunk is a seeded random trajectory that moves
with the gripper. The output then depends neither on the rendered pixels (CI renders with
Mesa's llvmpipe) nor on the CPU's float32 summation order, so one expected file holds on
any machine. What it pins is the eval path around the network: checkpoint loading,
dispatch on the `.pt` suffix, the proprio input and its normalisation, the chunk queue
(8 of 16 executed), the abs->delta conversion with yaw wrapping and gripper
binarisation, the closed loop through the physics, the failure taxonomy and the CSV.
The targets stay 25 cm or more above the table, so no contact can amplify a last-bit
difference in the physics.

Regenerate the expected file (only after an intended change):
    SDP_UPDATE_EXPECTED=1 pytest tests/test_smallbc_eval_repro.py
"""

import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from sdp.cli import main as sdp_main
from sdp.policies.smallbc.model import SmallBC
from sdp.policies.smallbc.policy import SmallBCPolicy

EXPECTED = Path(__file__).resolve().parent / "data" / "smallbc_eval_expected.json"
TIMING = ("policy_ms_mean", "policy_ms_max", "wall_s")
EXACT = ("condition", "distribution", "seed", "success", "label", "steps")
GAP = 7  # obs["state"][7]: the measured gripper opening, 0 (closed) to 0.08 m (open)
ACTION_STRIDE = 10  # store every 10th action of each episode, plus the per-dim mean


def _seeded_checkpoint(path: Path) -> None:
    torch.manual_seed(0)
    model = SmallBC(chunk=16).eval()
    lin1, lin2, lin3 = model.head[0], model.head[2], model.head[4]
    gap = model.enc_front.out_dim + model.enc_wrist.out_dim + GAP  # keypoints come first
    with torch.no_grad():
        for lin in (lin1, lin2, lin3):
            lin.weight.zero_()
            lin.bias.zero_()
        lin1.weight[0, gap], lin1.weight[1, gap] = 1.0, -1.0  # relu(+g), relu(-g)
        lin2.weight[0, 0], lin2.weight[1, 1] = 1.0, 1.0
        gain = torch.rand(lin3.out_features) * 2 - 1
        lin3.weight[:, 0], lin3.weight[:, 1] = gain, -gain  # output = gain * g
        offset = torch.rand(16, 5).numpy() * 2 - 1
    # Targets [x, y, z, yaw, gripper] = base + y_std * (offset + gain * g), with offset,
    # gain and the normalised opening g all in [-1, 1]: z stays in 0.35 +- 0.06 m.
    base = np.array([0.55, 0.0, 0.35, 0.0, 0.0])
    y_std = np.tile([0.05, 0.05, 0.03, 0.3, 1.0], (16, 1))
    p_mean, p_std = np.zeros(19), np.ones(19)
    p_mean[GAP], p_std[GAP] = 0.04, 0.04
    model.set_stats(base + y_std * offset, y_std, p_mean, p_std)
    torch.save(
        {"model": model.state_dict(), "model_cfg": model.cfg, "target_mode": "abs", "step": 0},
        path,
    )


def _episodes(rows: list[dict], actions: list[np.ndarray]) -> list[dict]:
    out, start = [], 0
    for row in rows:
        n = int(row["steps"])
        acts = np.stack(actions[start : start + n])
        start += n
        ep = {k: v for k, v in row.items() if k not in TIMING}
        ep.update(
            seed=int(ep["seed"]),
            success=ep["success"] == "True",
            steps=n,
            **{k: float(ep[k]) for k in ep if k not in EXACT},
            action_mean=acts.mean(axis=0).round(9).tolist(),
            actions_every_10th_step=acts[::ACTION_STRIDE].round(9).tolist(),
        )
        out.append(ep)
    assert start == len(actions)
    return out


def test_sdp_eval_smallbc_matches_stored_result(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt_seeded.pt"
    _seeded_checkpoint(ckpt)

    actions: list[np.ndarray] = []
    act = SmallBCPolicy.act

    def recording_act(self, obs, env):
        a = act(self, obs, env)
        actions.append(np.asarray(a, dtype=np.float64))
        return a

    monkeypatch.setattr(SmallBCPolicy, "act", recording_act)
    prefix = tmp_path / "eval_seeded"
    argv = ["eval", "--checkpoint", str(ckpt), "--condition", "smallbc_seeded"]
    argv += ["--trials", "3", "--seed-start", "500", "--distributions", "nominal"]
    assert sdp_main([*argv, "--out", str(prefix)]) == 0
    rows = list(csv.DictReader(prefix.with_suffix(".csv").open()))
    got = _episodes(rows, actions)

    if os.environ.get("SDP_UPDATE_EXPECTED"):
        EXPECTED.parent.mkdir(exist_ok=True)
        text = json.dumps({"episodes": got}, indent=1)
        # one action (innermost list of numbers) per line
        text = re.sub(r"\[\s+([^\[\]{}]+?)\s+\]", lambda m: f"[{' '.join(m[1].split())}]", text)
        EXPECTED.write_text(text + "\n")
        pytest.skip(f"wrote {EXPECTED}")

    want = json.loads(EXPECTED.read_text())["episodes"]
    assert [int(r["seed"]) for r in rows] == [10500, 10501, 10502]
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert {k: g[k] for k in EXACT} == {k: w[k] for k in EXACT}
        for key in set(w) - set(EXACT):
            np.testing.assert_allclose(g[key], w[key], rtol=0, atol=1e-6, err_msg=key)
