"""Check 2: is the cube's position recoverable from the decoded training pixels?

Frame 0 of all 50 episodes, decoded exactly as lerobot-train sees it (torchcodec,
uint8). Red-pixel centroid (u, v) per camera -> linear regression to the true initial
cube (x, y) from a reset-only env call per seed (no steps). R^2 in-sample and
leave-one-episode-out. Done for the 96 px and the 128 px dataset.
Writes check2_cube_in_pixels.json.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common  # noqa: E402,I001
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)


def fit_eval(X, Y):
    """Linear least squares with intercept; in-sample and LOO R^2 per target column."""
    A = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    ins = A @ coef
    loo = np.zeros_like(Y)
    for i in range(len(X)):
        m = np.arange(len(X)) != i
        c, *_ = np.linalg.lstsq(A[m], Y[m], rcond=None)
        loo[i] = A[i] @ c
    return {
        "r2_in_sample": [round(common.r2(Y[:, k], ins[:, k]), 4) for k in range(Y.shape[1])],
        "r2_loo": [round(common.r2(Y[:, k], loo[:, k]), 4) for k in range(Y.shape[1])],
        "rmse_loo_cm": [
            round(float(np.sqrt(np.mean((Y[:, k] - loo[:, k]) ** 2))) * 100, 3)
            for k in range(Y.shape[1])
        ],
    }


def main():
    print(f"MemAvailable {common.check_mem():.1f} GB")
    seeds = list(range(50))
    cube = common.initial_cube_poses(seeds)  # (x, y, yaw)
    out = {"targets": "initial cube (x, y) in metres", "cube_xy_std_cm": (cube[:, :2].std(0) * 100).round(2).tolist()}
    for name, root in (("96px", common.DS96), ("128px", common.DS128)):
        ds = common.load_dataset(root)
        starts, _ = common.episode_starts(root)
        res = {}
        for cam in ("front", "wrist"):
            uvn = []
            for s in starts:
                img = common.hwc(ds[int(s)][f"observation.images.{cam}"])
                uvn.append(common.centroid(common.red_mask(img)))
            uvn = np.array(uvn)
            ok = uvn[:, 2] > 0
            r = fit_eval(uvn[ok, :2], cube[ok, :2])
            r.update(
                {
                    "episodes_with_cube_visible": int(ok.sum()),
                    "red_px_min_mean_max": [int(uvn[ok, 2].min()), round(float(uvn[ok, 2].mean()), 1), int(uvn[ok, 2].max())],
                    "u_range": [round(float(uvn[ok, 0].min()), 1), round(float(uvn[ok, 0].max()), 1)],
                    "v_range": [round(float(uvn[ok, 1].min()), 1), round(float(uvn[ok, 1].max()), 1)],
                }
            )
            res[cam] = r
            print(name, cam, r)
        # Both cameras together.
        out[name] = res
        del ds
    (common.OUT / "check2_cube_in_pixels.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
