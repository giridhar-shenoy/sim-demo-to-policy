"""Record demonstrations from a policy (the scripted expert for v0.1) into LeRobot v3.0."""

from __future__ import annotations

import json
import time
from pathlib import Path

from sdp.envs.abs_target import abs_target
from sdp.envs.panda_pick_place import PandaPickPlaceEnv
from sdp.eval.harness import Policy
from sdp.eval.taxonomy import label_episode
from sdp.record.lerobot_writer import LeRobotWriter, make_frame
from sdp.seeds import check_demo


def record_episode(
    env: PandaPickPlaceEnv,
    policy: Policy,
    seed: int,
    distribution: str,
    action_mode: str = "delta",
):
    """Run ``policy`` (which acts in the env's 7-D delta space) for one episode.

    ``action_mode="abs_target"`` records, for each step, the integrated TCP target the env
    holds after the policy's delta (``sdp.envs.abs_target.abs_target``), so replaying the
    recorded targets through ``AbsoluteTargetWrapper`` reproduces the episode.
    """
    obs, info = env.reset(seed=seed, options={"distribution": distribution})
    policy.reset()
    frames = []
    while True:
        action = policy.act(obs, env)
        next_obs, reward, terminated, truncated, info = env.step(action)
        if action_mode == "abs_target":
            recorded = abs_target(env, gripper=float(action[6]))
        else:
            recorded = action
        frames.append(make_frame(obs, recorded, reward, terminated or truncated, info["success"]))
        obs = next_obs
        if terminated or truncated:
            break
    meta = {
        "seed": int(seed),
        "distribution": distribution,
        "success": bool(info["success"]),
        "label": label_episode(info["episode"], info),
        "action_mode": action_mode,
    }
    return frames, meta


def record_dataset(
    root: str | Path,
    repo_id: str,
    policy: Policy,
    seeds: list[int],
    source: str,
    distribution: str = "nominal",
    max_episode_steps: int | None = None,
    image_size: int = 96,
    action_mode: str = "delta",
    log=print,
) -> list[dict]:
    """Record one episode per seed (demo seeds only) and finalize the dataset.

    The total wall time goes to ``meta/sdp_recording.json`` next to the sidecar.
    """
    check_demo(seeds)
    env = PandaPickPlaceEnv(distribution=distribution, image_size=image_size)
    if max_episode_steps is not None:
        env.max_episode_steps = max_episode_steps
    writer = LeRobotWriter(
        root,
        repo_id,
        fps=env.control_hz,
        task=env.task_description,
        image_size=image_size,
        action_mode=action_mode,
    )
    records = []
    t0 = time.perf_counter()
    try:
        for i, seed in enumerate(seeds):
            frames, meta = record_episode(env, policy, seed, distribution, action_mode)
            meta["source"] = source
            writer.add_episode(frames, meta)
            records.append(meta)
            if log is not None:
                log(
                    f"episode {i + 1}/{len(seeds)} seed {seed}: {meta['label']} "
                    f"({len(frames)} frames, {time.perf_counter() - t0:.0f} s elapsed)"
                )
    finally:
        writer.finalize()
        env.close()
    timing = {
        "wall_time_s": round(time.perf_counter() - t0, 1),
        "episodes": len(records),
        "image_size": image_size,
        "action_mode": action_mode,
        "source": source,
    }
    (Path(root) / "meta" / "sdp_recording.json").write_text(json.dumps(timing, indent=2) + "\n")
    return records
