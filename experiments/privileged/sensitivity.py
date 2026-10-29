"""Open-loop sensitivity of a privileged-state checkpoint to the cube pose.

The repo's diagnose_openloop.py idea with environment_state in place of the images:
for the first 20 episodes, feed frame ``--frame`` (default 2, in the approach) and
compare the first predicted action across episodes with the recorded one; then keep
episode 0's robot state and swap in the other episodes' environment_state (cube pose)
and record how much the first action's x, y and yaw move.

For the absolute-target checkpoint the same numbers are also given as the delta the
wrapper would send (current target = the dataset's absolute action at frame-1), so they
are on the same normalised [-1, 1] scale as (a).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from priv_common import wrap_pi  # noqa: E402
from priv_policy import PrivilegedACTAdapter  # noqa: E402

HERE = Path(__file__).resolve().parent
MAX_DPOS, MAX_DROT = 0.02, 0.10


def load(root: Path) -> dict[int, dict[str, np.ndarray]]:
    import pyarrow.parquet as pq

    tables = [pq.read_table(p) for p in sorted((root / "data").rglob("*.parquet"))]

    def col(name):
        c = [t.column(name).to_numpy(zero_copy_only=False) for t in tables]
        c = np.concatenate(c)
        return np.stack(c) if c.dtype == object else c

    ep, fi = col("episode_index"), col("frame_index")
    st, es, act = col("observation.state"), col("observation.environment_state"), col("action")
    out = {}
    for e in sorted(set(ep.tolist())):
        idx = np.where(ep == e)[0]
        idx = idx[np.argsort(fi[idx])]
        out[e] = {"state": st[idx], "env": es[idx], "action": act[idx]}
    return out


def to_delta(a_abs: np.ndarray, cur_abs: np.ndarray) -> np.ndarray:
    """x, y, yaw of the delta the AbsoluteTargetWrapper would send."""
    dx = np.clip((a_abs[0] - cur_abs[0]) / MAX_DPOS, -1, 1)
    dy = np.clip((a_abs[1] - cur_abs[1]) / MAX_DPOS, -1, 1)
    dyaw = np.clip(wrap_pi(a_abs[3] - cur_abs[3]) / MAX_DROT, -1, 1)
    return np.array([dx, dy, dyaw])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", choices=["delta", "abs"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--frame", type=int, default=2)
    args = ap.parse_args()

    root = HERE / "datasets" / ("priv_delta_50" if args.mode == "delta" else "priv_abs_50")
    data = load(root)
    pol = PrivilegedACTAdapter(args.checkpoint)
    f = args.frame
    eps = list(data)[: args.episodes]
    dims = {"x": 0, "y": 1, "yaw": 5} if args.mode == "delta" else {"x": 0, "y": 1, "yaw": 3}

    def first(state, env_st):
        pol.reset()
        return pol.act_from_arrays(state, env_st)

    pred = np.array([first(data[e]["state"][f], data[e]["env"][f]) for e in eps])
    rec = np.array([data[e]["action"][f] for e in eps])
    per_dim = {}
    for name, k in dims.items():
        p, d = pred[:, k], rec[:, k]
        corr = float(np.corrcoef(p, d)[0, 1]) if p.std() > 1e-6 and d.std() > 1e-6 else None
        per_dim[name] = {
            "data_std": float(d.std()),
            "pred_std": float(p.std()),
            "corr": corr,
            "mean_abs_err": float(np.mean(np.abs(p - d))),
        }

    s0, e0 = data[eps[0]]["state"][f], data[eps[0]]["env"][f]
    a0 = first(s0, e0)
    swaps = np.array([np.abs(first(s0, data[e]["env"][f]) - a0) for e in eps[1:]])
    swap = {name: float(swaps[:, k].max()) for name, k in dims.items()}
    swap_mean = {name: float(swaps[:, k].mean()) for name, k in dims.items()}
    out = {
        "checkpoint": str(pol.path),
        "mode": args.mode,
        "frame": f,
        "episodes": len(eps),
        "first_action_across_episodes": per_dim,
        "env_state_swap_max_abs_change": swap,
        "env_state_swap_mean_abs_change": swap_mean,
    }
    if args.mode == "abs":
        cur0 = data[eps[0]]["action"][f - 1]
        d0 = to_delta(a0, cur0)
        sw = np.array([np.abs(to_delta(first(s0, data[e]["env"][f]), cur0) - d0) for e in eps[1:]])
        pd = np.array([to_delta(pred[i], data[e]["action"][f - 1]) for i, e in enumerate(eps)])
        rd = np.array([to_delta(rec[i], data[e]["action"][f - 1]) for i, e in enumerate(eps)])
        out["as_wrapper_delta"] = {
            "env_state_swap_max_abs_change": dict(zip(dims, sw.max(0).round(4).tolist(), strict=True)),
            "pred_std": dict(zip(dims, pd.std(0).round(4).tolist(), strict=True)),
            "data_std": dict(zip(dims, rd.std(0).round(4).tolist(), strict=True)),
        }
    path = HERE / "results" / f"sensitivity_{args.name}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
