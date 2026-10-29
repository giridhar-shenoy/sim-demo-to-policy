"""Repo-style open-loop image-swap test for run 7 checkpoints (absolute-target actions).

For each checkpoint: feed frame ``--frame`` (default 2) of the first 20 training episodes
and compare the first predicted action with the recorded one (std across episodes);
then keep episode 0's state and swap in the other episodes' front + wrist images and
record the max change of the first action. x and yaw are reported both as the raw
absolute target and as the delta AbsoluteTargetWrapper would send (current target = the
dataset's absolute action at frame-1 of the episode whose state is used), which is on
the same normalised scale as the repo's delta-action runs. Black images are also tried.
Uses the repo's ACTPolicyAdapter (sdp/eval/act_policy.py) unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from priv_common import wrap_pi  # noqa: E402

from sdp.eval.act_policy import ACTPolicyAdapter  # noqa: E402
from sdp.record.lerobot_writer import read_sidecar  # noqa: E402

MAX_DPOS, MAX_DROT = 0.02, 0.10
DIMS = {"x": 0, "y": 1, "yaw": 3}


def to_obs(item) -> dict:
    def img(key):
        return (item[key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

    return {"front": img("observation.images.front"), "wrist": img("observation.images.wrist"),
            "state": item["observation.state"].numpy()}


def to_delta(a: np.ndarray, cur: np.ndarray) -> np.ndarray:
    return np.array([
        np.clip((a[0] - cur[0]) / MAX_DPOS, -1, 1),
        np.clip((a[1] - cur[1]) / MAX_DPOS, -1, 1),
        np.clip(wrap_pi(a[3] - cur[3]) / MAX_DROT, -1, 1),
    ])


def r4(d):
    return {k: round(float(v), 4) for k, v in d.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--dataset", default=str(HERE / "datasets" / "pickplace_abs_50"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--frame", type=int, default=2)
    ap.add_argument("--out", default=str(HERE / "results" / "swap_sensitivity.json"))
    args = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.dataset)
    ds = LeRobotDataset(f"local/{root.name}", root=root)
    lengths = [r["length"] for r in read_sidecar(root)]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int)[: args.episodes]
    f = args.frame
    obs = [to_obs(ds[int(s) + f]) for s in starts]
    rec = np.array([ds[int(s) + f]["action"].numpy() for s in starts])
    cur = np.array([ds[int(s) + f - 1]["action"].numpy() for s in starts])  # current target
    rec_d = np.array([to_delta(rec[i], cur[i]) for i in range(len(starts))])

    results = {}
    for ck in args.checkpoints:
        pol = ACTPolicyAdapter(ck)

        def first(o):
            pol.reset()
            return pol.act(o)

        pred = np.array([first(o) for o in obs])
        pred_d = np.array([to_delta(pred[i], cur[i]) for i in range(len(starts))])
        base = obs[0]
        a0 = first(base)
        d0 = to_delta(a0, cur[0])
        sw_raw, sw_d = [], []
        for o in obs[1:]:
            a = first({**base, "front": o["front"], "wrist": o["wrist"]})
            sw_raw.append(np.abs(a - a0))
            sw_d.append(np.abs(to_delta(a, cur[0]) - d0))
        sw_raw, sw_d = np.array(sw_raw), np.array(sw_d)
        a_blk = first({**base, "front": np.zeros_like(base["front"]),
                       "wrist": np.zeros_like(base["wrist"])})
        res = {
            "checkpoint": str(pol.path),
            "swap_max_change_raw": r4({k: sw_raw[:, i].max() for k, i in DIMS.items()}),
            "swap_max_change_delta": r4({k: sw_d[:, j].max() for j, k in enumerate(DIMS)}),
            "swap_mean_change_delta": r4({k: sw_d[:, j].mean() for j, k in enumerate(DIMS)}),
            "first_step_std_raw": {"pred": r4({k: pred[:, i].std() for k, i in DIMS.items()}),
                                   "data": r4({k: rec[:, i].std() for k, i in DIMS.items()})},
            "first_step_std_delta": {"pred": r4({k: pred_d[:, j].std() for j, k in enumerate(DIMS)}),
                                     "data": r4({k: rec_d[:, j].std() for j, k in enumerate(DIMS)})},
            "first_step_corr_raw": r4({k: np.corrcoef(pred[:, i], rec[:, i])[0, 1]
                                       if pred[:, i].std() > 1e-6 else float("nan")
                                       for k, i in DIMS.items()}),
            "black_images_change_delta": r4(dict(zip(DIMS, np.abs(to_delta(a_blk, cur[0]) - d0)))),
            "black_images_change_raw_z_grip": [round(float(abs(a_blk[2] - a0[2])), 4),
                                               round(float(abs(a_blk[4] - a0[4])), 4)],
        }
        results[Path(ck).name] = res
        print(Path(ck).name, json.dumps({k: v for k, v in res.items() if k != "checkpoint"}), flush=True)
    out = {"dataset": str(root), "frame": f, "episodes": len(starts), "checkpoints": results}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
