"""Run 8 dataset: 200 scripted demos (demo seeds 0-199), 96 px front + wrist video,
15-D state, 5-D absolute-target action (same features as run 7), with DART noise on
every odd seed.

DART: on noisy episodes, Gaussian noise (std 0.3, normalised action units) is added to
the EXECUTED delta on dims x, y, z, yaw only (indices 0, 1, 2, 5; no roll/pitch, the
gripper is untouched), then clipped to [-1, 1]. The expert keeps reading the env and
corrects. The recorded label is always the CLEAN expert intent: the absolute target the
expert's own action would have produced from the current integrated target,
    pos = clip(cur_pos + a[:3] * max_dpos, workspace_lo, workspace_hi)
    yaw = clip(wrap(cur_yaw + a[5] * max_drot), -max_yaw, max_yaw)
plus its gripper command. That mirrors env._integrate_target. On clean episodes the
label must equal env.target_pose() after the step, and this is checked.

Only successful episodes are written (repo LeRobotWriter, video). Each attempted episode
goes to attempts.jsonl and per-group success rates go to recording_summary.json.
Noise RNG: np.random.default_rng(1_000_000 + seed).
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from priv_common import ABS_ACTION_NAMES, abs_target, target_yaw_tilt, wrap_pi  # noqa: E402

import sdp.record.lerobot_writer as lw  # noqa: E402
from sdp.envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from sdp.eval.taxonomy import label_episode  # noqa: E402
from sdp.expert.policy import ExpertPolicy  # noqa: E402
from sdp.seeds import demo_seeds  # noqa: E402

NOISE_STD = 0.3
NOISE_DIMS = [0, 1, 2, 5]
_orig_build_features = lw.build_features


def build_features_abs(image_size: int = 96) -> dict:
    feats = _orig_build_features(image_size)
    feats["action"] = {"dtype": "float32", "shape": (5,), "names": ABS_ACTION_NAMES}
    return feats


lw.build_features = build_features_abs


def clean_label(env, a: np.ndarray) -> np.ndarray:
    pos, quat = env.target_pose()
    yaw, _ = target_yaw_tilt(env, quat)
    a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
    p = np.clip(pos + a[:3] * env.max_dpos, env.workspace_lo, env.workspace_hi)
    y = float(np.clip(wrap_pi(yaw + a[5] * env.max_drot), -env.max_yaw, env.max_yaw))
    return np.array([p[0], p[1], p[2], y, a[6]], dtype=np.float32)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--out", default=str(HERE / "datasets" / "pickplace_abs_dart_200"))
    args = ap.parse_args()
    root = Path(args.out)
    env = PandaPickPlaceEnv(image_size=96)
    writer = lw.LeRobotWriter(root, f"local/{root.name}", fps=env.control_hz,
                              task=env.task_description, image_size=96)
    policy = ExpertPolicy()
    t0 = time.perf_counter()
    attempts = []
    max_label_err_clean = 0.0
    try:
        for seed in demo_seeds(args.episodes):
            dart = seed % 2 == 1
            rng = np.random.default_rng(1_000_000 + seed)
            obs, info = env.reset(seed=seed, options={"distribution": "nominal"})
            policy.reset()
            frames = []
            while True:
                a = np.asarray(policy.act(obs, env), dtype=np.float32)
                label = clean_label(env, a)
                executed = a.copy()
                if dart:
                    executed[NOISE_DIMS] += rng.normal(0.0, NOISE_STD, size=len(NOISE_DIMS))
                    executed = np.clip(executed, -1.0, 1.0)
                next_obs, reward, term, trunc, info = env.step(executed)
                if not dart:
                    err = np.abs(abs_target(env, float(a[6])) - label)
                    err[3] = abs(wrap_pi(float(err[3])))
                    max_label_err_clean = max(max_label_err_clean, float(err.max()))
                frames.append(lw.make_frame(obs, label, reward, term or trunc, info["success"]))
                obs = next_obs
                if term or trunc:
                    break
            meta = {"seed": seed, "distribution": "nominal", "success": bool(info["success"]),
                    "label": label_episode(info["episode"], info), "source": "scripted",
                    "dart": dart, "noise_std": NOISE_STD if dart else 0.0}
            attempts.append({"length": len(frames), **meta})
            if meta["success"]:
                writer.add_episode(frames, meta)
            print(f"seed {seed} dart={int(dart)}: {meta['label']} ({len(frames)} frames, "
                  f"{time.perf_counter() - t0:.0f} s)", flush=True)
    finally:
        writer.finalize()
        env.close()
    groups = {}
    for name, flag in (("clean", False), ("dart", True)):
        g = [r for r in attempts if r["dart"] == flag]
        groups[name] = {"attempted": len(g), "successes": sum(r["success"] for r in g),
                        "labels": dict(Counter(r["label"] for r in g)),
                        "frames_kept": sum(r["length"] for r in g if r["success"])}
    summary = {"groups": groups,
               "episodes_kept": sum(r["success"] for r in attempts),
               "frames_kept": sum(r["length"] for r in attempts if r["success"]),
               "max_label_err_clean_vs_env_target": max_label_err_clean,
               "wall_time_s": round(time.perf_counter() - t0, 1)}
    (HERE / "results").mkdir(exist_ok=True)
    with (HERE / "results" / "attempts.jsonl").open("w") as f:
        for r in attempts:
            f.write(json.dumps(r) + "\n")
    (HERE / "results" / "recording_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
