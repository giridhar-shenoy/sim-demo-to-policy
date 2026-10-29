"""Evaluate a privileged-state ACT checkpoint with the repo's harness and taxonomy.

Runs 50 nominal episodes on fresh eval seeds 10500-10549 and 10 on demo seeds 0-9
(diagnostic). ``--mode abs`` wraps the env in AbsoluteTargetWrapper.
Writes results/eval_<name>_{s10500,demo0}.csv and one JSON summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from priv_common import AbsoluteTargetWrapper  # noqa: E402
from priv_policy import PrivilegedACTAdapter  # noqa: E402

from sdp.envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from sdp.eval.harness import run_condition, summarize, write_csv, write_summary  # noqa: E402
from sdp.seeds import demo_seeds, eval_seeds  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", choices=["delta", "abs"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--eval-trials", type=int, default=50)
    ap.add_argument("--demo-trials", type=int, default=10)
    args = ap.parse_args()

    t0 = time.perf_counter()
    policy = PrivilegedACTAdapter(args.checkpoint)
    base = PandaPickPlaceEnv(render_obs=False)
    env = AbsoluteTargetWrapper(base) if args.mode == "abs" else base
    sets = {
        "s10500": eval_seeds(args.eval_trials, start=500),  # 10500-10549, fresh
        "demo0": demo_seeds(args.demo_trials),  # 0-9, training seeds (diagnostic)
    }
    summaries = []
    try:
        for tag, seeds in sets.items():
            cond = f"{args.name}_{tag}"
            res = run_condition(env, policy, seeds, "nominal", cond)
            write_csv(res, HERE / "results" / f"eval_{cond}.csv")
            s = summarize(res)
            summaries.append(s)
            lo, hi = s["wilson95"]
            print(
                f"{cond}: {s['successes']}/{s['trials']} ({100 * s['success_rate']:.0f} %, "
                f"Wilson 95 % {100 * lo:.0f}-{100 * hi:.0f} %) labels {s['labels']}",
                flush=True,
            )
    finally:
        base.close()
    meta = {
        "command": " ".join(sys.argv),
        "checkpoint": str(policy.path),
        "mode": args.mode,
        "n_action_steps": policy.policy.config.n_action_steps,
        "chunk_size": policy.policy.config.chunk_size,
        "wall_time_s": round(time.perf_counter() - t0, 1),
    }
    write_summary(summaries, HERE / "results" / f"eval_{args.name}.json", meta=meta)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
