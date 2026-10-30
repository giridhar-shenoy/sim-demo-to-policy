"""Copy smallbc's training and dataset metadata into results/, in the form used for ACT.

smallbc trains outside lerobot-train, so its run directory has its own files:
``config.json`` (arguments and data split), ``train_log.csv`` (loss, val metrics and
elapsed time) and the recorder's ``record_*.json``. This script turns them into
``results/train_<run>.{json,csv}`` (wall time of the full run and up to the checkpoint
that is evaluated), ``results/config_<run>.json`` and ``results/dataset_<name>.json``.
It reads only metadata.

    python scripts/smallbc_results_from_run.py \\
        --run experiments/smallbc/runs/abs1000 --checkpoint-step 25000 \\
        --data experiments/smallbc/data/main1000 --name smallbc_abs1000_25k
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="experiments/smallbc/runs/abs1000")
    ap.add_argument("--checkpoint-step", type=int, default=25000)
    ap.add_argument("--data", default="experiments/smallbc/data/main1000")
    ap.add_argument("--name", default="smallbc_abs1000_25k", help="run key used in results/")
    ap.add_argument("--dataset-name", default="smallbc_main1000")
    args = ap.parse_args()

    run, data, results = Path(args.run), Path(args.data), Path("results")
    rows = list(csv.DictReader((run / "train_log.csv").open()))
    final = rows[-1]
    at_ckpt = next(r for r in rows if int(float(r["step"])) == args.checkpoint_step)
    meta = {
        "returncode": 0,
        "wall_time_s": float(final["elapsed_s"]),
        "checkpoint_step": args.checkpoint_step,
        "wall_time_to_checkpoint_s": float(at_ckpt["elapsed_s"]),
        "config": f"{run}/config.json",
        "machine": "i5-1135G7 (4C/8T), 14 GB RAM, CPU only, torch 4 threads",
        "python": platform.python_version(),
    }
    (results / f"train_{args.name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    shutil.copyfile(run / "train_log.csv", results / f"train_{args.name}.csv")
    shutil.copyfile(run / "config.json", results / f"config_{args.name}.json")

    record = json.loads(next(data.glob("record_*.json")).read_text())
    summary = {
        "dataset": args.dataset_name,
        "image_size": 96,
        "episodes": record["kept"],
        "successful_episodes": record["kept"],
        "attempted_episodes": record["attempted"],
        "frames": record["frames"],
        "mean_episode_length_steps": record["frames"] / record["kept"],
        "seeds": record["seeds"],
        "codebase_version": "smallbc memmap",
        "format": "uint8 memmap shards (no video)",
        "recording_wall_time_s": record["wall_s"],
    }
    (results / f"dataset_{args.dataset_name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"train": meta, "dataset": summary}, indent=2))


if __name__ == "__main__":
    main()
