"""Record the two state-only datasets (no cameras) from one expert pass per seed.

For each demo seed the scripted expert runs once with rendering off. Every frame stores
observation.state (the env's 15-D robot state), observation.environment_state (cube
pose at observation time) and two actions: (a) the expert's 7-D delta, (b) the absolute
target [x, y, z, yaw, gripper] read from env.target_pose() after the step. Both datasets
get the repo's sidecar (meta/sdp_episodes.jsonl) so sdp.train.train can select the
successful episodes.

Also reports the largest roll/pitch (tilt) of the expert's target over all frames.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from priv_common import ABS_ACTION_NAMES, ENV_STATE_NAMES, abs_target, env_state, target_yaw_tilt  # noqa: E402

from sdp.envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from sdp.eval.taxonomy import label_episode  # noqa: E402
from sdp.expert.policy import ExpertPolicy  # noqa: E402
from sdp.record.lerobot_writer import ACTION_NAMES, SIDECAR, STATE_NAMES  # noqa: E402
from sdp.seeds import demo_seeds  # noqa: E402

os.environ.setdefault("SVT_LOG", "1")


def features(action_names: list[str]) -> dict:
    return {
        "observation.state": {"dtype": "float32", "shape": (15,), "names": STATE_NAMES},
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (len(ENV_STATE_NAMES),),
            "names": ENV_STATE_NAMES,
        },
        "action": {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="experiments/privileged/datasets")
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(args.out_dir)
    roots = {"delta": out / "priv_delta_50", "abs": out / "priv_abs_50"}
    names = {"delta": ACTION_NAMES, "abs": ABS_ACTION_NAMES}
    ds = {}
    for k, root in roots.items():
        if root.exists():
            raise FileExistsError(root)
        ds[k] = LeRobotDataset.create(
            repo_id=f"local/{root.name}",
            fps=20,
            features=features(names[k]),
            root=root,
            robot_type="franka_panda_sim",
            use_videos=False,
            image_writer_threads=0,
        )

    env = PandaPickPlaceEnv(render_obs=False)
    policy = ExpertPolicy()
    task = env.task_description
    max_tilt = 0.0
    t0 = time.perf_counter()
    records = []
    for i, seed in enumerate(demo_seeds(args.episodes)):
        obs, info = env.reset(seed=seed, options={"distribution": "nominal"})
        policy.reset()
        n = 0
        while True:
            es = env_state(env)
            action = np.asarray(policy.act(obs, env), dtype=np.float32)
            next_obs, reward, term, trunc, info = env.step(action)
            a_abs = abs_target(env, float(action[6]))
            _, tilt = target_yaw_tilt(env, env.target_pose()[1])
            max_tilt = max(max_tilt, float(np.linalg.norm(tilt)))
            common = {
                "observation.state": np.asarray(obs["state"], np.float32),
                "observation.environment_state": es,
                "next.reward": np.array([reward], np.float32),
                "next.done": np.array([term or trunc], bool),
                "next.success": np.array([info["success"]], bool),
                "task": task,
            }
            ds["delta"].add_frame({**common, "action": action})
            ds["abs"].add_frame({**common, "action": a_abs})
            n += 1
            obs = next_obs
            if term or trunc:
                break
        meta = {
            "seed": seed,
            "distribution": "nominal",
            "success": bool(info["success"]),
            "label": label_episode(info["episode"], info),
            "source": "scripted",
        }
        for k in ds:
            ds[k].save_episode()
            rec = {"episode_index": i, "length": n, **meta}
            p = roots[k] / SIDECAR
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        records.append({"length": n, **meta})
        print(f"seed {seed}: {meta['label']} ({n} frames)", flush=True)
    for k in ds:
        ds[k].finalize()
    env.close()
    summary = {
        "episodes": len(records),
        "successes": sum(r["success"] for r in records),
        "frames": sum(r["length"] for r in records),
        "max_target_tilt_rad": max_tilt,
        "wall_time_s": round(time.perf_counter() - t0, 1),
    }
    print(json.dumps(summary))
    (out / "recording_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
