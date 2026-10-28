"""train.py builds the lerobot-train command line; no training runs here."""

from pathlib import Path

import yaml

from sdp.train.train import build_command, flatten, parse_log_line, rows_from_lines

CONFIG = Path(__file__).resolve().parents[1] / "sdp" / "train" / "configs" / "act_cpu.yaml"


def test_cpu_config_matches_the_spec():
    cfg = yaml.safe_load(CONFIG.read_text())
    policy = cfg["policy"]
    assert policy["type"] == "act" and policy["device"] == "cpu"
    assert policy["dim_model"] == 256
    assert (policy["n_encoder_layers"], policy["n_decoder_layers"]) == (2, 1)
    assert policy["chunk_size"] == 20
    assert cfg["batch_size"] == 8 and cfg["steps"] == 3000
    assert policy["push_to_hub"] is False and cfg["wandb"]["enable"] is False


def test_flatten_and_command_line():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert flatten({"a": {"b": 1, "c": {"d": True}}}) == {"a.b": 1, "a.c.d": True}
    cmd = build_command(
        cfg,
        "datasets/pickplace_scripted_50",
        "outputs/train/x",
        "x",
        episodes=[0, 2, 3],
        overrides={"steps": 50},
    )
    assert cmd[0] == "lerobot-train"
    assert "--policy.type=act" in cmd
    assert "--policy.push_to_hub=false" in cmd
    assert "--wandb.enable=false" in cmd
    assert "--dataset.repo_id=local/pickplace_scripted_50" in cmd
    assert "--dataset.root=datasets/pickplace_scripted_50" in cmd
    assert "--dataset.episodes=[0,2,3]" in cmd
    assert "--steps=50" in cmd and "--steps=3000" not in cmd
    assert all(a.startswith("--") and "=" in a for a in cmd[1:])


def test_parse_log_line():
    line = (
        "Training: 100%|xx| 40/40 [00:30<00:00, 2.00step/s]INFO 2026-10-29 03:23:26 "
        "ot_train.py:641 step:40 smpl:320 ep:3 epch:0.06 loss:6.809 grdn:144.230 "
        "lr:1.0e-05 updt_s:0.503 data_s:0.005 smp/s:16 l1_loss:0.340 kld_loss:0.647"
    )
    row = parse_log_line(line)
    assert row["step"] == 40 and row["loss"] == 6.809 and row["lr"] == 1e-5
    assert "s" not in row and "py" not in row and "03" not in row
    assert parse_log_line("step:1K smpl:8K loss:0.5")["step"] == 1000
    assert parse_log_line("Training: 8%| 236/3000") is None


def test_rows_rebuild_rounded_steps():
    lines = [
        "step:950 smpl:8K loss:0.3",
        "Training: 33%| 1000/3000",
        "step:1K smpl:8K loss:0.29",
        "step:1K smpl:8K loss:0.28",
    ]
    rows = rows_from_lines(lines, log_freq=50)
    assert [r["step"] for r in rows] == [50, 100, 150]
    assert [r["step_printed"] for r in rows] == [950, 1000, 1000]
