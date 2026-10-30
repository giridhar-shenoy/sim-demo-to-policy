"""Record scripted-expert demos for smallbc into uncompressed numpy shards.

One shard = up to ``--shard-size`` successful episodes, written as a directory of
``.npy`` files (memory-mappable, no codec):

    front.npy, wrist.npy   (N, S, S, 3) uint8   observation images at obs time
    state.npy              (N, 15) float32      env state at obs time
    cube.npy               (N, 4)  float32      cube [x, y, z, yaw] at obs time
    target_obs.npy         (N, 4)  float32      integrated TCP target [x, y, z, yaw] at obs time
    tilt.npy               (N,)    float32      tilt (rad) of that target from straight down
    action_exec.npy        (N, 7)  float32      delta action actually sent to env.step
    action_clean.npy       (N, 7)  float32      the expert's clean delta action (the label)
    target_abs.npy         (N, 5)  float32      target after the CLEAN action
                                                [x, y, z, yaw, gripper]
    phase.npy              (N,)    int8         expert phase index that produced the action
    frame_index.npy        (N,)    int16
    episodes.json          [{seed, start, length, noised, label}, ...]

DART-style noise: in a ``--noise-frac`` fraction of episodes (chosen per seed,
deterministically), Gaussian noise with std ``--noise-std`` (normalised action units)
is added to the delta dims in ``--noise-dims`` of the EXECUTED action; the stored
labels (``action_clean``, ``target_abs``) stay the clean expert output for the state
actually visited. Only successful episodes are kept; the success rate is logged.

Seeds come from the repo's demo range (0-999, sdp.seeds.check_demo).

    python -m sdp.policies.smallbc.record --out outputs/smallbc/data/demo5 --episodes 5
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from sdp.policies.smallbc.common import (
    apply_env_defaults,
    target_after_action,
    target_xyzyaw,
    target_yaw,
)

FIELDS = (
    "front",
    "wrist",
    "state",
    "cube",
    "target_obs",
    "tilt",
    "action_exec",
    "action_clean",
    "target_abs",
    "phase",
    "frame_index",
)
DTYPES = {
    "front": np.uint8,
    "wrist": np.uint8,
    "state": np.float32,
    "cube": np.float32,
    "target_obs": np.float32,
    "tilt": np.float32,
    "action_exec": np.float32,
    "action_clean": np.float32,
    "target_abs": np.float32,
    "phase": np.int8,
    "frame_index": np.int16,
}


def is_noised(seed: int, noise_std: float, noise_frac: float) -> bool:
    if noise_std <= 0.0 or noise_frac <= 0.0:
        return False
    return bool(np.random.default_rng([seed, 1]).random() < noise_frac)


def record_episode(env, expert, seed, distribution, noised, noise_std, noise_dims):
    from sdp.eval.taxonomy import label_episode

    obs, info = env.reset(seed=seed, options={"distribution": distribution})
    expert.reset()
    rng = np.random.default_rng([seed, 0xDA27])
    buf: dict[str, list] = {k: [] for k in FIELDS}
    max_label_err = 0.0
    t = 0
    while True:
        pos_quat = env.target_pose()
        t_obs = target_xyzyaw(env)
        _, tilt_vec = target_yaw(env, pos_quat[1])
        cube_pos, cube_yaw = env.cube_pose()
        phase = expert.phase
        clean = np.asarray(expert.act(env), dtype=np.float64)
        after = target_after_action(env, clean)
        exe = clean.copy()
        if noised:
            exe[noise_dims] += rng.normal(0.0, noise_std, size=len(noise_dims))
            exe = np.clip(exe, -1.0, 1.0)
        buf["front"].append(obs["front"])
        buf["wrist"].append(obs["wrist"])
        buf["state"].append(obs["state"])
        buf["cube"].append([cube_pos[0], cube_pos[1], cube_pos[2], cube_yaw])
        buf["target_obs"].append(t_obs)
        buf["tilt"].append(float(np.linalg.norm(tilt_vec)))
        buf["action_exec"].append(exe)
        buf["action_clean"].append(clean)
        buf["target_abs"].append([after[0], after[1], after[2], after[3], clean[6]])
        buf["phase"].append(phase)
        buf["frame_index"].append(t)
        obs, _, terminated, truncated, info = env.step(exe.astype(np.float32))
        if not noised:  # executed == clean, so the env's new target must equal the label
            err = np.abs(target_xyzyaw(env) - after)
            err[3] = abs(float(np.angle(np.exp(1j * err[3]))))
            max_label_err = max(max_label_err, float(err.max()))
        t += 1
        if terminated or truncated:
            break
    ep = {k: np.asarray(v, dtype=DTYPES[k]) for k, v in buf.items()}
    meta = {
        "seed": int(seed),
        "length": t,
        "noised": bool(noised),
        "success": bool(info["success"]),
        "label": label_episode(info["episode"], info),
        "max_tilt": float(ep["tilt"].max()),
        "label_check_err": max_label_err,
    }
    return ep, meta


def write_shard(out: Path, name: str, episodes: list[tuple[dict, dict]]) -> None:
    tmp = out / f".tmp_{name}"
    if tmp.exists():  # leftover from an interrupted run of this script
        shutil.rmtree(tmp)
    if (out / name).exists():
        raise FileExistsError(f"{out / name} already exists")
    tmp.mkdir(parents=True)
    rows, start = [], 0
    for _ep, meta in episodes:
        rows.append(
            {
                "seed": meta["seed"],
                "start": start,
                "length": meta["length"],
                "noised": meta["noised"],
                "label": meta["label"],
            }
        )
        start += meta["length"]
    for k in FIELDS:
        np.save(tmp / f"{k}.npy", np.concatenate([ep[k] for ep, _ in episodes], axis=0))
    (tmp / "episodes.json").write_text(json.dumps(rows) + "\n")
    final = out / name
    if final.exists():  # never overwrite recorded data; pick a new --out or seed range
        raise FileExistsError(f"{final} already exists")
    tmp.rename(final)


def record_seeds(
    worker: int,
    seeds: list[int],
    out: str,
    image_size: int,
    distribution: str,
    noise_std: float,
    noise_frac: float,
    noise_dims: list[int],
    shard_size: int,
    log_every: int,
) -> dict:
    from sdp.envs.panda_pick_place import PandaPickPlaceEnv
    from sdp.expert import WaypointExpert

    out_p = Path(out)
    env = PandaPickPlaceEnv(distribution=distribution, image_size=image_size)
    expert = WaypointExpert()
    stats: dict[str, Any] = {
        "attempted": 0,
        "kept": 0,
        "noised_attempted": 0,
        "noised_kept": 0,
        "frames": 0,
        "labels": Counter(),
        "max_tilt": 0.0,
        "label_check_err": 0.0,
    }
    pending: list[tuple[dict, dict]] = []
    shard_first = seeds[0] if seeds else 0
    t0 = time.perf_counter()

    def flush(last_seed: int) -> None:
        nonlocal pending, shard_first
        if pending:
            write_shard(out_p, f"shard_{shard_first:04d}_{last_seed:04d}", pending)
        pending = []
        shard_first = last_seed + 1

    try:
        for i, seed in enumerate(seeds):
            noised = is_noised(seed, noise_std, noise_frac)
            ep, meta = record_episode(
                env, expert, seed, distribution, noised, noise_std, noise_dims
            )
            stats["attempted"] += 1
            stats["noised_attempted"] += int(noised)
            stats["labels"][meta["label"]] += 1
            stats["max_tilt"] = max(stats["max_tilt"], meta["max_tilt"])
            stats["label_check_err"] = max(stats["label_check_err"], meta["label_check_err"])
            if meta["success"]:
                stats["kept"] += 1
                stats["noised_kept"] += int(noised)
                stats["frames"] += meta["length"]
                pending.append((ep, meta))
            if len(pending) >= shard_size:
                flush(seed)
            if log_every and (i + 1) % log_every == 0:
                el = time.perf_counter() - t0
                print(
                    f"[w{worker}] {i + 1}/{len(seeds)} seed {seed}: kept {stats['kept']}/"
                    f"{stats['attempted']}, {60 * (i + 1) / el:.1f} ep/min",
                    flush=True,
                )
        if seeds:
            flush(seeds[-1])
    finally:
        env.close()
    stats["wall_s"] = time.perf_counter() - t0
    stats["labels"] = dict(stats["labels"])
    return stats


def main() -> None:
    from sdp.seeds import demo_seeds

    apply_env_defaults()

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--distribution", default="nominal")
    p.add_argument("--noise-std", type=float, default=0.0, help="std in normalised action units")
    p.add_argument("--noise-frac", type=float, default=0.0, help="fraction of episodes noised")
    p.add_argument(
        "--noise-dims",
        default="0,1,2,3,4,5",
        help="delta dims that get noise (3,4 = roll/pitch, which the expert never corrects)",
    )
    p.add_argument("--shard-size", type=int, default=25)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    seeds = demo_seeds(args.episodes, start=args.seed_start)  # raises outside 0-999
    noise_dims = [int(d) for d in args.noise_dims.split(",") if d.strip()]
    assert all(0 <= d < 6 for d in noise_dims), "noise dims must be delta dims 0-5"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_w = max(1, min(args.workers, len(seeds)))
    chunks = [list(c) for c in np.array_split(np.array(seeds), n_w)]
    jobs = [
        (
            w,
            [int(s) for s in c],
            str(out),
            args.image_size,
            args.distribution,
            args.noise_std,
            args.noise_frac,
            noise_dims,
            args.shard_size,
            args.log_every,
        )
        for w, c in enumerate(chunks)
    ]
    t0 = time.perf_counter()
    if n_w == 1:
        results = [record_seeds(*jobs[0])]
    else:
        with mp.get_context("spawn").Pool(n_w) as pool:
            results = pool.starmap(record_seeds, jobs)
    wall = time.perf_counter() - t0

    total = {
        k: sum(r[k] for r in results)
        for k in ("attempted", "kept", "noised_attempted", "noised_kept", "frames")
    }
    labels: Counter[str] = Counter()
    for r in results:
        labels.update(r["labels"])
    clean_att = total["attempted"] - total["noised_attempted"]
    clean_kept = total["kept"] - total["noised_kept"]
    summary = {
        "args": vars(args),
        "seeds": [seeds[0], seeds[-1]],
        **total,
        "success_rate": total["kept"] / max(1, total["attempted"]),
        "success_rate_clean": clean_kept / clean_att if clean_att else None,
        "success_rate_noised": (
            total["noised_kept"] / total["noised_attempted"] if total["noised_attempted"] else None
        ),
        "labels": dict(labels),
        "max_tilt_rad": max(r["max_tilt"] for r in results),
        "clean_label_check_max_err": max(r["label_check_err"] for r in results),
        "wall_s": round(wall, 2),
        "episodes_per_min": round(60 * total["attempted"] / wall, 2),
        "frames_per_s": round(sum(r["frames"] for r in results) / wall, 1),
        "workers": n_w,
    }
    (out / f"record_{seeds[0]:04d}_{seeds[-1]:04d}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=2))


if __name__ == "__main__":
    main()
