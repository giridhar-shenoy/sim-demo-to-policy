"""Env throughput on this machine: control steps per second, headless.

Measures PandaPickPlace-v0 with the two 96x96 image observations rendered (what
recording and eval use) and with image rendering off (physics + IK only), using
random actions for --steps control steps after a warm-up. Writes results/env_speed.json.
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from sdp.envs import PandaPickPlaceEnv


def bench(render_obs: bool, steps: int, seed: int = 0) -> float:
    env = PandaPickPlaceEnv(render_obs=render_obs)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    for _ in range(20):  # warm-up: GL context, caches
        env.step(rng.uniform(-0.3, 0.3, 7))
    done_steps, t0 = 0, time.perf_counter()
    while done_steps < steps:
        _, _, term, trunc, _ = env.step(rng.uniform(-0.3, 0.3, 7))
        done_steps += 1
        if term or trunc:
            env.reset(seed=seed + done_steps)
    rate = steps / (time.perf_counter() - t0)
    env.close()
    return rate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default="results/env_speed.json")
    args = ap.parse_args()
    out = {
        "env": "PandaPickPlace-v0",
        "steps": args.steps,
        "steps_per_s_with_images": round(bench(True, args.steps), 1),
        "steps_per_s_without_images": round(bench(False, args.steps), 1),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "cpu": platform.processor() or platform.machine(),
        "load_avg_1min_at_end": round(os.getloadavg()[0], 2),
    }
    print(json.dumps(out, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
