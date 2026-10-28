"""End-to-end smoke: record 2 scripted episodes, train ACT for 50 steps, evaluate 3 episodes.

Checks the plumbing (recorder -> lerobot-train -> checkpoint -> harness), not the
policy: 50 steps is nowhere near a working policy.
"""

import csv
from pathlib import Path

import pytest

from sdp.cli import main as sdp_main
from sdp.eval.taxonomy import LABELS
from sdp.expert.policy import ExpertPolicy
from sdp.record.recorder import record_dataset
from sdp.train.train import main as train_main

CONFIG = Path(__file__).resolve().parents[1] / "sdp" / "train" / "configs" / "act_cpu.yaml"


@pytest.mark.slow
def test_train_50_steps_then_eval_3_episodes(tmp_path):
    data = tmp_path / "scripted_2ep"
    records = record_dataset(
        data, "local/scripted_2ep", ExpertPolicy(), [0, 1], "scripted", log=None
    )
    assert all(r["success"] for r in records)

    out = tmp_path / "train" / "smoke"
    code = train_main(
        [
            "--config",
            str(CONFIG),
            "--dataset",
            str(data),
            "--output-dir",
            str(out),
            "--steps",
            "50",
            "--set",
            "num_workers=0",
            "--set",
            "log_freq=10",
            "--set",
            "save_freq=50",
            # no ImageNet download in CI; the real run uses the pretrained backbone
            "--set",
            "policy.pretrained_backbone_weights=null",
        ]
    )
    assert code == 0
    assert (out / "checkpoints" / "last" / "pretrained_model" / "model.safetensors").is_file()
    rows = list(csv.DictReader(out.with_suffix(".csv").open()))
    assert [int(float(r["step"])) for r in rows] == [10, 20, 30, 40, 50]

    prefix = tmp_path / "eval_smoke"
    assert (
        sdp_main(
            [
                "eval",
                "--checkpoint",
                str(out),
                "--condition",
                "smoke",
                "--trials",
                "3",
                "--distributions",
                "nominal",
                "--out",
                str(prefix),
            ]
        )
        == 0
    )
    episodes = list(csv.DictReader(prefix.with_suffix(".csv").open()))
    assert [int(e["seed"]) for e in episodes] == [10000, 10001, 10002]
    assert all(e["label"] in LABELS for e in episodes)
