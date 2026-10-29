"""Open-loop check of a checkpoint on its own training data.

For the first 20 episodes of the dataset, feed frame ``--frame`` (default 2, inside the
approach phase) to the policy and compare its first action with the recorded one.
During the approach the expert's x, y and yaw actions depend on where the cube is, so a
policy that has learned to localise the cube must vary those outputs across episodes.
It also swaps the two images of episode 0's frame for those of other episodes (same
state) and records how much the first action moves: if the policy localised the cube
from the images, the approach actions would change.
Writes results/openloop_<run>.json: per-dimension std of data and prediction, their
correlation, the image-swap sensitivity, and the mean absolute error over frames
0, 10, ..., 90 of episode 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from sdp.eval.act_policy import ACTPolicyAdapter
from sdp.record.lerobot_writer import ACTION_NAMES, read_sidecar


def to_obs(item) -> dict:
    def img(key):
        return (item[key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

    return {
        "front": img("observation.images.front"),
        "wrist": img("observation.images.wrist"),
        "state": item["observation.state"].numpy(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="datasets/pickplace_scripted_50")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--frame", type=int, default=2)
    args = ap.parse_args()

    root = Path(args.dataset)
    ds = LeRobotDataset(f"local/{root.name}", root=root)
    policy = ACTPolicyAdapter(args.checkpoint)
    lengths = [r["length"] for r in read_sidecar(root)]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int)

    pred, data = [], []
    for start in starts[: args.episodes]:
        item = ds[int(start) + args.frame]
        policy.reset()
        pred.append(policy.act(to_obs(item)))
        data.append(item["action"].numpy())
    pred, data = np.array(pred), np.array(data)

    per_dim = {}
    for k, name in enumerate(ACTION_NAMES):
        p_std, d_std = float(pred[:, k].std()), float(data[:, k].std())
        varies = p_std > 1e-6 and d_std > 1e-6
        corr = float(np.corrcoef(pred[:, k], data[:, k])[0, 1]) if varies else None
        per_dim[name] = {"data_std": d_std, "pred_std": p_std, "corr": corr}

    base = to_obs(ds[int(starts[0]) + args.frame])
    policy.reset()
    a_base = policy.act(base)
    swaps = []
    for start in starts[1 : args.episodes]:
        other = to_obs(ds[int(start) + args.frame])
        policy.reset()
        swapped = {**base, "front": other["front"], "wrist": other["wrist"]}
        swaps.append(np.abs(policy.act(swapped) - a_base))
    blank = {**base, "front": np.zeros_like(base["front"]), "wrist": np.zeros_like(base["wrist"])}
    policy.reset()
    a_blank = policy.act(blank)

    err = []
    for i in range(0, min(100, lengths[0]), 10):
        item = ds[i]
        policy.reset()
        err.append(np.abs(policy.act(to_obs(item)) - item["action"].numpy()))
    out = {
        "checkpoint": str(policy.path),
        "dataset": str(root),
        "frame": args.frame,
        "episodes": args.episodes,
        "first_action_across_episodes": per_dim,
        "image_swap_max_abs_change": dict(
            zip(ACTION_NAMES, np.max(swaps, axis=0).round(4).tolist(), strict=True)
        ),
        "black_images_abs_change": dict(
            zip(ACTION_NAMES, np.abs(a_blank - a_base).round(4).tolist(), strict=True)
        ),
        "episode0_mean_abs_err": dict(
            zip(ACTION_NAMES, np.mean(err, axis=0).round(4).tolist(), strict=True)
        ),
    }
    ckpt = Path(args.checkpoint)
    # A run dir gives openloop_<run>.json; a checkpoints/<step> dir gives
    # openloop_<run>_step<N>.json, so intermediate checkpoints can be compared.
    if ckpt.parent.name == "checkpoints":
        run = f"{ckpt.parent.parent.name}_step{int(ckpt.name)}"
    else:
        run = ckpt.name
    path = Path("results") / f"openloop_{run}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    for name in ("d_x", "d_y", "d_yaw"):
        d = per_dim[name]
        print(
            f"{run} {name}: data std {d['data_std']:.2f}, pred std {d['pred_std']:.4f}, "
            f"max change when images are swapped {out['image_swap_max_abs_change'][name]:.4f}"
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
