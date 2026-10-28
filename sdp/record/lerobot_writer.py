"""LeRobot v3.0 dataset writer for this repo's envs.

A thin layer over ``LeRobotDataset.create / add_frame / save_episode / finalize`` that
fixes the feature schema and keeps a per-episode sidecar
(``meta/sdp_episodes.jsonl``: seed, pose distribution, success flag, failure label,
length, source). Failed episodes are written and flagged, never dropped; training
chooses which episodes to use from the sidecar.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

STATE_NAMES = [
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_qw",
    "tcp_qx",
    "tcp_qy",
    "tcp_qz",
    "gripper_gap",
] + [f"joint{i}" for i in range(1, 8)]
ACTION_NAMES = ["d_x", "d_y", "d_z", "d_roll", "d_pitch", "d_yaw", "gripper"]
SIDECAR = "meta/sdp_episodes.jsonl"

# SVT-AV1 (LeRobot's default video encoder) prints ~30 info lines per video; errors only.
os.environ.setdefault("SVT_LOG", "1")


def build_features(image_size: int = 96) -> dict:
    img = {
        "dtype": "video",
        "shape": (image_size, image_size, 3),
        "names": ["height", "width", "channels"],
    }
    return {
        "observation.images.front": dict(img),
        "observation.images.wrist": dict(img),
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
    }


def make_frame(obs: dict, action: np.ndarray, reward: float, done: bool, success: bool) -> dict:
    return {
        "observation.images.front": obs["front"],
        "observation.images.wrist": obs["wrist"],
        "observation.state": np.asarray(obs["state"], dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "next.reward": np.array([reward], dtype=np.float32),
        "next.done": np.array([done], dtype=bool),
        "next.success": np.array([success], dtype=bool),
    }


class LeRobotWriter:
    def __init__(
        self,
        root: str | Path,
        repo_id: str,
        fps: int,
        task: str,
        image_size: int = 96,
        robot_type: str = "franka_panda_sim",
        image_writer_threads: int = 4,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root)
        if self.root.exists():
            raise FileExistsError(f"{self.root} exists; refusing to overwrite a dataset")
        self.task = task
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=build_features(image_size),
            root=self.root,
            robot_type=robot_type,
            use_videos=True,
            image_writer_threads=image_writer_threads,
        )
        self._episodes: list[dict] = []

    def add_episode(self, frames: list[dict], meta: dict) -> int:
        """Write one episode; ``meta`` goes to the sidecar. Returns the episode index."""
        if not frames:
            raise ValueError("empty episode")
        for frame in frames:
            self.dataset.add_frame({**frame, "task": self.task})
        self.dataset.save_episode()
        index = len(self._episodes)
        record = {"episode_index": index, "length": len(frames), **meta}
        self._episodes.append(record)
        sidecar = self.root / SIDECAR
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with sidecar.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return index

    def finalize(self) -> None:
        self.dataset.finalize()


def read_sidecar(root: str | Path) -> list[dict]:
    path = Path(root) / SIDECAR
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
