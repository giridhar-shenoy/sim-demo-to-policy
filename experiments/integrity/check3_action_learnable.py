"""Check 3: are the approach actions a learnable function of the initial cube pose?

Inputs: initial cube (x, y, yaw) per episode (reset-only env call per seed). Targets: the
recorded d_x, d_y, d_yaw at frames 0-5. Models (sklearn is not installed): linear,
numpy polynomial (degree 2 and 3, ridge 1e-6), and numpy kNN (k = 3, standardised
inputs), plus kNN on the single matching coordinate and the frame-0 closed form. R^2 on held-out episodes (5-fold by episode, 10 episodes per fold).
Also: approach-phase length per episode and the fraction of saturated actions.

Approach length: the expert's above_cube phase ends after the first frame t at which
the measured TCP is within 8 mm (xy) and 0.04 rad (yaw) of the cube (waypoint.py
_advance). This is reconstructed from observation.state and the recorded actions, and
validated against the true phases of seeds 0-9 from check4's shadow expert.
Writes check3_action_learnable.json.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common  # noqa: E402,I001
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from sdp.expert.waypoint import wrap_quarter_turn  # noqa: E402

TARGETS = {"d_x": 0, "d_y": 1, "d_yaw": 5}
SAT = 0.6  # the expert's per-step limit (WaypointExpert.speed); actions never exceed it


def poly_features(X, deg):
    cols = [np.ones(len(X))]
    n = X.shape[1]
    from itertools import combinations_with_replacement

    for d in range(1, deg + 1):
        for idx in combinations_with_replacement(range(n), d):
            cols.append(np.prod(X[:, idx], axis=1))
    return np.column_stack(cols)


def fit_predict(kind, Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    if kind.startswith("poly") or kind == "linear":
        deg = 1 if kind == "linear" else int(kind[-1])
        A, B = poly_features(Ztr, deg), poly_features(Zte, deg)
        coef = np.linalg.solve(A.T @ A + 1e-6 * np.eye(A.shape[1]), A.T @ ytr)
        return B @ coef
    if kind == "knn3":
        d = ((Zte[:, None, :] - Ztr[None, :, :]) ** 2).sum(-1)
        nn = np.argsort(d, axis=1)[:, :3]
        return ytr[nn].mean(1)
    raise ValueError(kind)


def cv_r2(X, y, groups, kind, folds=5):
    pred = np.zeros_like(y)
    ug = np.unique(groups)
    for f in range(folds):
        test_g = ug[f::folds]
        te = np.isin(groups, test_g)
        pred[te] = fit_predict(kind, X[~te], y[~te], X[te])
    return common.r2(y, pred)


def tcp_yaw_from_quat(q):
    w, x, y, z = q.T
    # Rotation matrix column 1 (TCP y axis = finger closing direction): R[0,1], R[1,1].
    r01 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    return np.arctan2(r11, r01)


def approach_length(states, actions, cube, max_drot=0.10, pos_tol=0.008, yaw_tol=0.04):
    tcp_xy = states[:, :2]
    tcp_yaw = tcp_yaw_from_quat(states[:, 3:7])
    tgt_yaw = tcp_yaw[0] + np.concatenate([[0.0], np.cumsum(actions[:, 5] * max_drot)[:-1]])
    for t in range(len(states)):
        yaw_goal = tgt_yaw[t] + wrap_quarter_turn(cube[2] - tgt_yaw[t])
        if np.linalg.norm(cube[:2] - tcp_xy[t]) < pos_tol and abs(yaw_goal - tcp_yaw[t]) < yaw_tol:
            return t + 1
    return None


def main():
    df = pq.read_table(common.DS96 / "data/chunk-000/file-000.parquet").to_pandas()
    cube = common.initial_cube_poses(range(50))
    out = {}

    # --- approach length + saturation
    lengths, all_a, appr_a = [], [], []
    for ep in range(50):
        e = df[df.episode_index == ep]
        a = np.stack(e.action.values)
        s = np.stack(e["observation.state"].values)
        L = approach_length(s, a, cube[ep])
        lengths.append(L)
        all_a.append(a)
        appr_a.append(a[:L])
    lengths = np.array(lengths)
    truth = json.loads((common.OUT / "check4_alignment.json").read_text())["episodes"]
    val = {ep: (int(lengths[int(ep)]), r["approach_frames"]) for ep, r in truth.items()}
    A, P = np.concatenate(all_a), np.concatenate(appr_a)
    sat = lambda M, k: round(float(np.mean(np.abs(M[:, k]) >= SAT - 1e-6)), 3)  # noqa: E731
    out["approach"] = {
        "validation_reconstructed_vs_true_seeds0-9": val,
        "validation_exact_matches": sum(a == b for a, b in val.values()),
        "per_episode": lengths.tolist(),
        "mean": round(float(lengths.mean()), 2),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "approach_frames_total": int(lengths.sum()),
        "fraction_of_all_frames": round(float(lengths.sum() / len(df)), 4),
    }
    out["saturation_|a|>=0.6"] = {
        "all_frames": {n: sat(A, k) for n, k in TARGETS.items()} | {"d_z": sat(A, 2)},
        "approach_frames": {n: sat(P, k) for n, k in TARGETS.items()} | {"d_z": sat(P, 2)},
        "approach_x_or_yaw_values_saturated": round(
            float(np.mean(np.abs(P[:, [0, 5]]) >= SAT - 1e-6)), 3
        ),
        "max_abs_action_any_dim": float(np.abs(A[:, :6]).max()),
        # Cube-dependent targets in the whole dataset: frames where |d_x| or |d_yaw| > 0.05.
        "frames_with_|d_x|or|d_yaw|>0.05_fraction": round(
            float(np.mean((np.abs(A[:, 0]) > 0.05) | (np.abs(A[:, 5]) > 0.05))), 4
        ),
    }

    # --- regression, frames 0-5
    X_rows, Y_rows, G, T = [], [], [], []
    for ep in range(50):
        e = df[df.episode_index == ep]
        a = np.stack(e.action.values)
        for t in range(6):
            X_rows.append(cube[ep])
            Y_rows.append(a[t])
            G.append(ep)
            T.append(t)
    X, Y, G, T = np.array(X_rows), np.array(Y_rows), np.array(G), np.array(T)
    kinds = ["linear", "poly2", "poly3", "knn3"]
    reg = {}
    for name, k in TARGETS.items():
        y = Y[:, k]
        row = {"data_std_frames0-5": round(float(y.std()), 3)}
        for kind in kinds:
            # Per frame (inputs: cube pose) and pooled over frames 0-5 (inputs: pose + t).
            per_t = [round(cv_r2(X[T == t], y[T == t], G[T == t], kind), 3) for t in range(6)]
            pooled = round(cv_r2(np.column_stack([X, T]), y, G, kind), 3)
            row[kind] = {"per_frame_t0..t5": per_t, "pooled_with_t": pooled}
        # kNN on the one coordinate the expert's law uses for this output (x for d_x,
        # y for d_y, yaw for d_yaw; the x, y and yaw channels are decoupled in waypoint.py).
        c = {"d_x": 0, "d_y": 1, "d_yaw": 2}[name]
        per_t = [round(cv_r2(X[T == t][:, [c]], y[T == t], G[T == t], "knn3"), 3) for t in range(6)]
        pooled = round(cv_r2(np.column_stack([X[:, c], T]), y, G, "knn3"), 3)
        row["knn3_matching_coord_only"] = {"per_frame_t0..t5": per_t, "pooled_with_t": pooled}
        reg[name] = row
        print(name, json.dumps(row))
    out["regression_heldout_r2"] = reg
    # The expert's frame-0 law in closed form: clip((cube_x - tcp_x0) / max_dpos, +-0.6).
    tcp_x0 = float(np.stack(df[df.frame_index == 0]["observation.state"].values)[:, 0].mean())
    law = np.clip((cube[:, 0] - tcp_x0) / 0.02, -SAT, SAT)
    rec0 = np.stack(df[df.frame_index == 0].action.values)[:, 0]
    out["frame0_d_x_closed_form"] = {
        "r2": round(common.r2(rec0, law), 5),
        "max_abs_err": float(np.abs(rec0 - law).max()),
        "fraction_saturated": round(float(np.mean(np.abs(rec0) >= SAT - 1e-6)), 3),
    }
    out["frame0_action_std"] = {n: round(float(Y[T == 0, k].std()), 3) for n, k in TARGETS.items()}
    print(json.dumps(out["approach"], indent=0))
    print(json.dumps(out["saturation_|a|>=0.6"], indent=0))
    (common.OUT / "check3_action_learnable.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
