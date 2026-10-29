"""Shared helpers for the integrity checks (no training; light CPU only)."""

from __future__ import annotations

import os
from pathlib import Path

# The repo's tests use ~/work/menagerie on this laptop (tests/conftest.py); same here.
os.environ.setdefault("SDP_MENAGERIE_DIR", str(Path("~/work/menagerie").expanduser()))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DS96 = REPO / "datasets" / "pickplace_scripted_50"
DS128 = REPO / "datasets" / "pickplace_scripted_50_128px"
CKPT_V5 = REPO / "outputs/train/act_cpu_v5_scripted50_128px/checkpoints/last/pretrained_model"
CKPT_V2 = REPO / "outputs/train/act_cpu_v2_scripted50/checkpoints/last/pretrained_model"


def check_mem(min_gb: float = 3.0) -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            gb = int(line.split()[1]) / 1024**2
            if gb < min_gb:
                raise SystemExit(f"MemAvailable {gb:.1f} GB < {min_gb} GB; aborting")
            return gb
    raise RuntimeError("no MemAvailable")


def load_dataset(root: Path, uint8: bool = True):
    """Open the dataset the way lerobot-train does (torchcodec, return_uint8=True)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        f"local/{root.name}", root=root, video_backend="torchcodec", return_uint8=uint8
    )


def episode_starts(root: Path) -> tuple[np.ndarray, np.ndarray]:
    from sdp.record.lerobot_writer import read_sidecar

    lengths = np.array([r["length"] for r in read_sidecar(root)])
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int)
    return starts, lengths


def hwc(t) -> np.ndarray:
    """CHW tensor (uint8, or float in [0,1]) -> HWC uint8."""
    a = t.permute(1, 2, 0).numpy()
    if a.dtype != np.uint8:
        a = (a * 255).round().clip(0, 255).astype(np.uint8)
    return a


# Red-cube mask. The cube's rgba is (0.85, 0.12, 0.10); under the flat lighting its raw
# rendered pixels have R >= ~110 with G, B < ~60, while the table checker is
# (~150, ~130, ~110) and the green target / grey arm have R < 2*G. Chosen from the raw
# renders (see check1 output "raw cube pixel stats").
def red_mask(img: np.ndarray) -> np.ndarray:
    r, g, b = (img[..., i].astype(np.int32) for i in range(3))
    return (r > 90) & (r > 2 * g) & (r > 2 * b)


def centroid(mask: np.ndarray) -> tuple[float, float, int]:
    vs, us = np.nonzero(mask)
    if len(us) == 0:
        return float("nan"), float("nan"), 0
    return float(us.mean()), float(vs.mean()), int(len(us))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def initial_cube_poses(seeds, distribution: str = "nominal") -> np.ndarray:
    """(x, y, yaw) per seed from a reset-only call (no steps, no rendering)."""
    from sdp.envs.panda_pick_place import PandaPickPlaceEnv

    env = PandaPickPlaceEnv(distribution=distribution, render_obs=False)
    out = []
    for s in seeds:
        env.reset(seed=int(s), options={"distribution": distribution})
        out.append(env._initial_cube.copy())
    env.close()
    return np.array(out)
