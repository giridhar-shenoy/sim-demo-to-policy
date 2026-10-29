"""Run 7 dataset: run 2's inputs (96 px front + wrist video, 15-D state) with the 5-D
absolute-target action [target_x, target_y, target_z, target_yaw, gripper].

Reuses the repo's LeRobotWriter (same video encoding, same sidecar) and make_frame; only
the action feature is swapped, by replacing lerobot_writer.build_features in this
process (the tracked file is not modified). The absolute target is read from
env.target_pose() after the expert's step, exactly as in the privileged dataset (b).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from priv_common import ABS_ACTION_NAMES, abs_target  # noqa: E402

import sdp.record.lerobot_writer as lw  # noqa: E402
from sdp.envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from sdp.eval.taxonomy import label_episode  # noqa: E402
from sdp.expert.policy import ExpertPolicy  # noqa: E402
from sdp.seeds import demo_seeds  # noqa: E402

_orig_build_features = lw.build_features


def build_features_abs(image_size: int = 96) -> dict:
    feats = _orig_build_features(image_size)
    feats["action"] = {"dtype": "float32", "shape": (5,), "names": ABS_ACTION_NAMES}
    return feats


lw.build_features = build_features_abs


def main() -> None:
    root = HERE / "datasets" / "pickplace_abs_50"
    env = PandaPickPlaceEnv(image_size=96)
    writer = lw.LeRobotWriter(root, f"local/{root.name}", fps=env.control_hz,
                              task=env.task_description, image_size=96)
    policy = ExpertPolicy()
    t0 = time.perf_counter()
    records = []
    try:
        for seed in demo_seeds(50):
            obs, info = env.reset(seed=seed, options={"distribution": "nominal"})
            policy.reset()
            frames = []
            while True:
                action = np.asarray(policy.act(obs, env), dtype=np.float32)
                next_obs, reward, term, trunc, info = env.step(action)
                a_abs = abs_target(env, float(action[6]))
                frames.append(lw.make_frame(obs, a_abs, reward, term or trunc, info["success"]))
                obs = next_obs
                if term or trunc:
                    break
            meta = {"seed": seed, "distribution": "nominal", "success": bool(info["success"]),
                    "label": label_episode(info["episode"], info), "source": "scripted"}
            writer.add_episode(frames, meta)
            records.append({"length": len(frames), **meta})
            print(f"seed {seed}: {meta['label']} ({len(frames)} frames, "
                  f"{time.perf_counter() - t0:.0f} s)", flush=True)
    finally:
        writer.finalize()
        env.close()
    summary = {"episodes": len(records), "successes": sum(r["success"] for r in records),
               "frames": sum(r["length"] for r in records),
               "wall_time_s": round(time.perf_counter() - t0, 1)}
    (root / "meta" / "sdp_recording.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
