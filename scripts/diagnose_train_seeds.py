"""Diagnostic, not an eval: roll a checkpoint out on its own demonstration seeds.

If a policy also fails from the exact initial states it was trained on, the problem is
fitting, not generalisation to new cube poses. Uses demo seeds 0-9 on purpose, so the
output goes to results/trainseeds_<run>.json and never into the eval tables.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sdp.envs import PandaPickPlaceEnv
from sdp.eval.act_policy import ACTPolicyAdapter
from sdp.eval.harness import run_episode
from sdp.seeds import demo_seeds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    policy = ACTPolicyAdapter(args.checkpoint)
    env = PandaPickPlaceEnv()
    try:
        results = [
            run_episode(env, policy, s, "nominal", "train_seed_diagnostic").result
            for s in demo_seeds(args.n)
        ]
    finally:
        env.close()
    run = Path(args.checkpoint).name
    out = {
        "checkpoint": str(policy.path),
        "note": "diagnostic on training (demo) seeds; not an evaluation",
        "seeds": [r.seed for r in results],
        "successes": sum(r.success for r in results),
        "labels": dict(Counter(r.label for r in results)),
    }
    path = Path("results") / f"trainseeds_{run}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{run}: {out['successes']}/{len(results)} on demo seeds, labels {out['labels']}")


if __name__ == "__main__":
    main()
