"""Evaluation harness: run a policy on listed seeds and write per-episode rows.

Every published success rate traces back to a CSV written here, with one row per
episode (seed, distribution, success, length, failure label, initial cube pose).
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from sdp.envs.panda_pick_place import PandaPickPlaceEnv
from sdp.eval.stats import wilson_interval
from sdp.eval.taxonomy import LABELS, label_episode


class Policy(Protocol):
    """Anything that maps an observation to a 7-D action. ``env`` is passed so the
    scripted expert can read privileged state; learned policies ignore it."""

    def reset(self) -> None: ...

    def act(self, obs: dict, env: PandaPickPlaceEnv) -> np.ndarray: ...


@dataclass
class EpisodeResult:
    condition: str
    distribution: str
    seed: int
    success: bool
    label: str
    steps: int
    init_x: float
    init_y: float
    init_yaw: float
    final_dx: float
    final_dy: float
    policy_ms_mean: float
    policy_ms_max: float
    wall_s: float


@dataclass
class Rollout:
    result: EpisodeResult
    frames: list[np.ndarray] = field(default_factory=list)


def run_episode(
    env: PandaPickPlaceEnv,
    policy: Policy,
    seed: int,
    distribution: str,
    condition: str,
    record_frames: bool = False,
    on_step: Callable[[dict, np.ndarray, dict], None] | None = None,
) -> Rollout:
    t0 = time.perf_counter()
    obs, info = env.reset(seed=seed, options={"distribution": distribution})
    policy.reset()
    frames = [env.render()] if record_frames else []
    policy_s = []
    while True:
        tp = time.perf_counter()
        action = np.asarray(policy.act(obs, env), dtype=np.float32)
        policy_s.append(time.perf_counter() - tp)
        if on_step is not None:
            on_step(obs, action, info)
        obs, _, terminated, truncated, info = env.step(action)
        if record_frames:
            frames.append(env.render())
        if terminated or truncated:
            break
    ep = info["episode"]
    x, y, yaw = ep["initial_cube_xy_yaw"]
    result = EpisodeResult(
        condition=condition,
        distribution=distribution,
        seed=int(seed),
        success=bool(info["success"]),
        label=label_episode(ep, info),
        steps=int(ep["steps"]),
        init_x=round(float(x), 5),
        init_y=round(float(y), 5),
        init_yaw=round(float(yaw), 5),
        final_dx=round(float(info["cube_target_offset"][0]), 5),
        final_dy=round(float(info["cube_target_offset"][1]), 5),
        policy_ms_mean=round(1e3 * float(np.mean(policy_s)), 3),
        policy_ms_max=round(1e3 * float(np.max(policy_s)), 3),
        wall_s=round(time.perf_counter() - t0, 3),
    )
    return Rollout(result=result, frames=frames)


def run_condition(
    env: PandaPickPlaceEnv,
    policy: Policy,
    seeds: Iterable[int],
    distribution: str,
    condition: str,
    log: Callable[[str], None] | None = print,
) -> list[EpisodeResult]:
    results = []
    for i, seed in enumerate(seeds):
        r = run_episode(env, policy, seed, distribution, condition).result
        results.append(r)
        if log is not None and ((i + 1) % 10 == 0 or not r.success):
            n_ok = sum(x.success for x in results)
            log(f"[{condition}/{distribution}] {i + 1} episodes, {n_ok} ok; seed {seed}: {r.label}")
    return results


def summarize(results: list[EpisodeResult]) -> dict:
    n = len(results)
    k = sum(r.success for r in results)
    lo, hi = wilson_interval(k, n)
    lengths = [r.steps for r in results]
    ok_lengths = [r.steps for r in results if r.success]
    labels = Counter(r.label for r in results)
    seeds = [r.seed for r in results]
    return {
        "condition": results[0].condition,
        "distribution": results[0].distribution,
        "trials": n,
        "successes": k,
        "success_rate": k / n,
        "wilson95": [lo, hi],
        "mean_steps": float(np.mean(lengths)),
        "mean_steps_success": float(np.mean(ok_lengths)) if ok_lengths else None,
        "labels": {label: labels.get(label, 0) for label in LABELS},
        "seed_min": min(seeds),
        "seed_max": max(seeds),
        "policy_ms_mean": float(np.mean([r.policy_ms_mean for r in results])),
        "policy_ms_max": float(np.max([r.policy_ms_max for r in results])),
    }


def write_csv(results: list[EpisodeResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[EpisodeResult]:
    out = []
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            out.append(
                EpisodeResult(
                    condition=row["condition"],
                    distribution=row["distribution"],
                    seed=int(row["seed"]),
                    success=row["success"] == "True",
                    label=row["label"],
                    steps=int(row["steps"]),
                    init_x=float(row["init_x"]),
                    init_y=float(row["init_y"]),
                    init_yaw=float(row["init_yaw"]),
                    final_dx=float(row["final_dx"]),
                    final_dy=float(row["final_dy"]),
                    policy_ms_mean=float(row.get("policy_ms_mean") or "nan"),
                    policy_ms_max=float(row.get("policy_ms_max") or "nan"),
                    wall_s=float(row["wall_s"]),
                )
            )
    return out


def write_summary(summaries: list[dict], path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta or {}, "conditions": summaries}, indent=2) + "\n")
