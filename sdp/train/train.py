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


def rows_from_lines(lines, log_freq: int) -> list[dict[str, float]]:
    """Parse lerobot-train log lines into rows.

    lerobot prints step counts of 1000 and above rounded ("step:2K" for 2350), so the
    step column is rebuilt from the row index: lerobot logs exactly every ``log_freq``
    steps. The printed value is kept as ``step_printed``.
    """
    rows = []
    for line in lines:
        row = parse_log_line(line)
        if row:
            row["step_printed"] = row["step"]
            row["step"] = float((len(rows) + 1) * log_freq)
            rows.append(row)
    return rows


def _log_freq(cmd: list[str]) -> int:
    for arg in cmd:
        if arg.startswith("--log_freq="):
            return int(arg.split("=", 1)[1])
    raise ValueError("--log_freq missing from the lerobot-train command")


def write_rows(rows: list[dict[str, float]], csv_path: Path) -> None:
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "step", k))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def reparse_log(output_dir: str | Path) -> Path:
    """Rebuild ``<output_dir>.csv`` from ``<output_dir>.log`` (first line = the command)."""
    output_dir = Path(output_dir)
    lines = output_dir.with_suffix(".log").read_text().replace("\r", "\n").splitlines()
    rows = rows_from_lines(lines[1:], _log_freq(lines[0].split()))
    write_rows(rows, output_dir.with_suffix(".csv"))
    return output_dir.with_suffix(".csv")


def run(cmd: list[str], output_dir: Path) -> dict:
    log_path = output_dir.with_suffix(".log")
    csv_path = output_dir.with_suffix(".csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
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
            lines.append(line)
        code = proc.wait()
    wall = time.perf_counter() - t0
    rows = rows_from_lines(lines, _log_freq(cmd))
    if rows:
        write_rows(rows, csv_path)
    meta = {"returncode": code, "wall_time_s": round(wall, 1), "log": str(log_path)}
    output_dir.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--dataset", help="LeRobot dataset root")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--reparse-log", action="store_true", help="only rebuild <output-dir>.csv from its .log"
    )
    ap.add_argument("--job-name", default=None)
    ap.add_argument("--steps", type=int, default=None, help="override steps (smoke runs)")
    ap.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="extra override"
    )
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = ap.parse_args(argv)
    if args.reparse_log:
        print(reparse_log(args.output_dir))
        return 0
    if not (args.config and args.dataset):
        ap.error("--config and --dataset are required unless --reparse-log")

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
