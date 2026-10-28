"""Record demonstrations from a policy (the scripted expert for v0.1) into LeRobot v3.0."""

from __future__ import annotations

import time
from pathlib import Path

from sdp.envs.panda_pick_place import PandaPickPlaceEnv
from sdp.eval.harness import Policy
from sdp.eval.taxonomy import label_episode
from sdp.record.lerobot_writer import LeRobotWriter, make_frame
from sdp.seeds import check_demo


def record_episode(env: PandaPickPlaceEnv, policy: Policy, seed: int, distribution: str):
    obs, info = env.reset(seed=seed, options={"distribution": distribution})
    policy.reset()
    frames = []
    while True:
        action = policy.act(obs, env)
        next_obs, reward, terminated, truncated, info = env.step(action)
        frames.append(make_frame(obs, action, reward, terminated or truncated, info["success"]))
        obs = next_obs
        if terminated or truncated:
            break
    meta = {
        "seed": int(seed),
        "distribution": distribution,
        "success": bool(info["success"]),
        "label": label_episode(info["episode"], info),
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
    log=print,
) -> list[dict]:
    """Record one episode per seed (demo seeds only) and finalize the dataset."""
    check_demo(seeds)
    env = PandaPickPlaceEnv(distribution=distribution)
    if max_episode_steps is not None:
        env.max_episode_steps = max_episode_steps
    writer = LeRobotWriter(root, repo_id, fps=env.control_hz, task=env.task_description)
    records = []
    t0 = time.perf_counter()
    try:
        for i, seed in enumerate(seeds):
            frames, meta = record_episode(env, policy, seed, distribution)
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
    return records
