"""Replay dataset (b)'s absolute actions through AbsoluteTargetWrapper from the same seeds.

Passes if every episode succeeds. Also reports the largest difference between the
converted delta and the expert's recorded delta (dataset (a)) over all frames.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from priv_common import AbsoluteTargetWrapper, abs_to_delta  # noqa: E402

from sdp.envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from sdp.eval.taxonomy import label_episode  # noqa: E402
from sdp.record.lerobot_writer import read_sidecar  # noqa: E402

HERE = Path(__file__).resolve().parent


def load_actions(root: Path) -> list[np.ndarray]:
    import pyarrow.parquet as pq

    tables = [pq.read_table(p) for p in sorted((root / "data").rglob("*.parquet"))]
    ep = np.concatenate([t.column("episode_index").to_numpy() for t in tables])
    fi = np.concatenate([t.column("frame_index").to_numpy() for t in tables])
    act = np.concatenate([np.stack(t.column("action").to_numpy(zero_copy_only=False)) for t in tables])
    out = []
    for e in sorted(set(ep.tolist())):
        idx = np.where(ep == e)[0]
        out.append(act[idx[np.argsort(fi[idx])]])
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--abs-root", default=str(HERE / "datasets" / "priv_abs_50"))
    ap.add_argument("--delta-root", default=str(HERE / "datasets" / "priv_delta_50"),
                    help="dataset with the expert's recorded deltas; 'none' to skip")
    ap.add_argument("--out", default=str(HERE / "results" / "replay_abs_wrapper.json"))
    args = ap.parse_args()
    abs_root = Path(args.abs_root)
    side = read_sidecar(abs_root)
    acts_abs = load_actions(abs_root)
    acts_delta = load_actions(Path(args.delta_root)) if args.delta_root != "none" else [None] * len(acts_abs)
    base = PandaPickPlaceEnv(render_obs=False)
    env = AbsoluteTargetWrapper(base)
    rows, max_diff = [], 0.0
    for rec, a_abs, a_delta in zip(side, acts_abs, acts_delta, strict=True):
        env.reset(seed=rec["seed"], options={"distribution": "nominal"})
        info = {}
        for t, a in enumerate(a_abs):
            d = abs_to_delta(base, a)
            if a_delta is not None:
                max_diff = max(max_diff, float(np.max(np.abs(d - a_delta[t]))))
            _, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
        rows.append(
            {
                "seed": rec["seed"],
                "success": bool(info["success"]),
                "label": label_episode(info["episode"], info),
                "steps": t + 1,
                "recorded_steps": len(a_abs),
            }
        )
    base.close()
    out = {
        "episodes": len(rows),
        "successes": sum(r["success"] for r in rows),
        "max_abs_diff_converted_vs_recorded_delta": max_diff,
        "steps_match": all(r["steps"] == r["recorded_steps"] for r in rows),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
