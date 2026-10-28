"""Thin wrapper around ``lerobot-train`` (LeRobot 0.6.1).

Builds the command line from a YAML config (keys are lerobot-train config paths,
flattened to ``--a.b=value``, the draccus syntax ``lerobot-train --help`` documents),
selects the successful episodes from the recorder's sidecar, runs it, tees the
output to ``<output_dir>.log``, parses the periodic log lines into
``<output_dir>.csv`` (no wandb), and writes the wall time to ``<output_dir>.json``.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

from sdp.record.lerobot_writer import read_sidecar

# lerobot-train log line, e.g.
# "INFO 2026-10-29 03:23:16 ot_train.py:641 step:20 smpl:160 ep:1 epch:0.03 loss:16.978 ...".
# Fields are parsed from "step:" onwards; keys must start with a letter and must not
# follow "/" (skips the timestamp, "ot_train.py:641" and "smp/s").
_LOG_FIELD = re.compile(r"(?<![\w/.])([a-z_][a-z0-9_]*):(-?[\d.]+(?:e[-+]?\d+)?[KMB]?)")
_SCALE = {"K": 1e3, "M": 1e6, "B": 1e9}


def flatten(cfg: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in cfg.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{path}."))
        else:
            out[path] = value
    return out


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | tuple):
        return "[" + ",".join(_fmt(v) for v in value) + "]"
    return str(value)


def build_command(
    config: dict,
    dataset_root: str | Path,
    output_dir: str | Path,
    job_name: str,
    episodes: list[int] | None = None,
    overrides: dict | None = None,
) -> list[str]:
    dataset_root = Path(dataset_root)
    flat = flatten(config)
    flat.update(
        {
            "dataset.repo_id": f"local/{dataset_root.name}",
            "dataset.root": str(dataset_root),
            "output_dir": str(output_dir),
            "job_name": job_name,
        }
    )
    if episodes is not None:
        flat["dataset.episodes"] = list(episodes)
    flat.update(overrides or {})
    return ["lerobot-train"] + [f"--{k}={_fmt(v)}" for k, v in flat.items()]


def successful_episodes(dataset_root: str | Path) -> list[int]:
    return [r["episode_index"] for r in read_sidecar(dataset_root) if r["success"]]


def parse_log_line(line: str) -> dict[str, float] | None:
    if "step:" not in line or "loss:" not in line:
        return None
    row = {}
    for key, raw in _LOG_FIELD.findall(line[line.index("step:") :]):
        scale = _SCALE.get(raw[-1], 1.0)
        row[key] = float(raw.rstrip("KMB")) * scale
    return row if "step" in row else None


def run(cmd: list[str], output_dir: Path) -> dict:
    log_path = output_dir.with_suffix(".log")
    csv_path = output_dir.with_suffix(".csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    with (
        log_path.open("w") as log,
        subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        ) as proc,
    ):
        log.write(" ".join(cmd) + "\n")
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            row = parse_log_line(line)
            if row:
                rows.append(row)
        code = proc.wait()
    wall = time.perf_counter() - t0
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "step", k))
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    meta = {"returncode": code, "wall_time_s": round(wall, 1), "log": str(log_path)}
    output_dir.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True, help="LeRobot dataset root")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--job-name", default=None)
    ap.add_argument("--steps", type=int, default=None, help="override steps (smoke runs)")
    ap.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="extra override"
    )
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = ap.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    overrides: dict[str, object] = dict(s.split("=", 1) for s in args.set)
    if args.steps is not None:
        overrides["steps"] = args.steps
    output_dir = Path(args.output_dir)
    cmd = build_command(
        config,
        args.dataset,
        output_dir,
        args.job_name or output_dir.name,
        episodes=successful_episodes(args.dataset),
        overrides=overrides,
    )
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    meta = run(cmd, output_dir)
    print(json.dumps(meta))
    return meta["returncode"]


if __name__ == "__main__":
    sys.exit(main())
