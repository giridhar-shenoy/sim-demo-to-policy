"""Print and save episode counts and lengths for a recorded dataset (CHECKPOINT 2)."""

import json
import sys
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from sdp.record.lerobot_writer import read_sidecar


def main(root: str) -> None:
    root_p = Path(root)
    info = json.loads((root_p / "meta" / "info.json").read_text())
    side = read_sidecar(root_p)
    ds = LeRobotDataset(f"local/{root_p.name}", root=root_p)
    lengths = np.array([r["length"] for r in side])
    summary = {
        "dataset": root_p.name,
        "codebase_version": info["codebase_version"],
        "episodes": ds.num_episodes,
        "frames": ds.num_frames,
        "fps": ds.fps,
        "successful_episodes": int(sum(r["success"] for r in side)),
        "mean_episode_length_steps": float(lengths.mean()),
        "min_max_episode_length_steps": [int(lengths.min()), int(lengths.max())],
        "seeds": [min(r["seed"] for r in side), max(r["seed"] for r in side)],
        "features": sorted(info["features"]),
    }
    print(json.dumps(summary, indent=2))
    out = Path("results") / f"dataset_{root_p.name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
