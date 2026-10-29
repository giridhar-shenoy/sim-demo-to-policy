"""Scripted-expert success rate on PandaPickPlace-v0 (CHECKPOINT 1 number).

Runs the waypoint expert on demo seeds 0-99 (the population the scripted demos are
drawn from) under the nominal pose distribution, and on eval seeds 10000-10099 under
both distributions. Writes results/expert_pickplace.csv and .json.
"""

import argparse
import platform

from sdp.envs import PandaPickPlaceEnv
from sdp.eval.harness import run_condition, summarize, write_csv, write_summary
from sdp.expert.policy import ExpertPolicy
from sdp.seeds import demo_seeds, eval_seeds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="results/expert_pickplace")
    ap.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="also render both cameras at this size (the expert itself never reads them)",
    )
    args = ap.parse_args()

    if args.image_size is None:
        env = PandaPickPlaceEnv(render_obs=False)
    else:
        env = PandaPickPlaceEnv(render_obs=True, image_size=args.image_size)
    policy = ExpertPolicy()
    runs = [
        ("expert_demo_seeds", "nominal", demo_seeds(args.n)),
        ("expert_eval_seeds", "nominal", eval_seeds(args.n)),
        ("expert_eval_seeds", "shifted", eval_seeds(args.n)),
    ]
    rows, summaries = [], []
    for condition, dist, seeds in runs:
        res = run_condition(env, policy, seeds, dist, condition, log=None)
        rows += res
        s = summarize(res)
        summaries.append(s)
        lo, hi = s["wilson95"]
        print(
            f"{condition:18s} {dist:8s} {s['successes']}/{s['trials']} "
            f"({100 * s['success_rate']:.0f} %, Wilson 95 % {100 * lo:.0f}-{100 * hi:.0f} %), "
            f"mean length {s['mean_steps']:.1f} steps, labels {s['labels']}"
        )
    write_csv(rows, f"{args.out}.csv")
    meta = {"python": platform.python_version(), "image_size": args.image_size}
    write_summary(summaries, f"{args.out}.json", meta=meta)
    env.close()


if __name__ == "__main__":
    main()
