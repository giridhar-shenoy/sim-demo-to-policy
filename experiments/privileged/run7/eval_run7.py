"""Evaluate a run 7 checkpoint (pixels + state, absolute-target actions).

The repo's ACTPolicyAdapter (unchanged) drives a PandaPickPlaceEnv(image_size=96)
wrapped in AbsoluteTargetWrapper; episodes go through the repo harness and taxonomy.
Seed sets: ``dev`` = 10500-10549 (the coordinator's dev seeds), ``demo`` = 0-9.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from priv_common import AbsoluteTargetWrapper  # noqa: E402

from sdp.envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from sdp.eval.act_policy import ACTPolicyAdapter  # noqa: E402
from sdp.eval.harness import run_condition, summarize, write_csv, write_summary  # noqa: E402
from sdp.seeds import demo_seeds, eval_seeds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", choices=["dev", "demo"], required=True)
    ap.add_argument("--distribution", choices=["nominal", "shifted"], default="nominal")
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    seeds = eval_seeds(50, start=500) if args.seeds == "dev" else demo_seeds(10)
    policy = ACTPolicyAdapter(args.checkpoint)
    base = PandaPickPlaceEnv(image_size=policy.image_size)
    env = AbsoluteTargetWrapper(base)
    cond = f"{args.name}_{args.seeds}{seeds[0]}"
    try:
        res = run_condition(env, policy, seeds, args.distribution, cond)
    finally:
        base.close()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.results_dir) / f"eval_{cond}_{args.distribution}"
    write_csv(res, f"{out}.csv")
    s = summarize(res)
    meta = {"command": " ".join(sys.argv), "checkpoint": str(policy.path),
            "wall_time_s": round(time.perf_counter() - t0, 1)}
    write_summary([s], f"{out}.json", meta=meta)
    lo, hi = s["wilson95"]
    print(f"{cond} {args.distribution}: {s['successes']}/{s['trials']} "
          f"({100 * s['success_rate']:.0f} %, Wilson 95 % {100 * lo:.1f}-{100 * hi:.1f} %) "
          f"labels {s['labels']} wall {meta['wall_time_s']} s", flush=True)


if __name__ == "__main__":
    main()
