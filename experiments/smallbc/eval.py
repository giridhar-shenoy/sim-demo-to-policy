"""Evaluate a smallbc checkpoint with the repo's harness (same taxonomy, Wilson CI).

Writes ``<out>.csv`` (one row per episode) and ``<out>.json`` (success rate, Wilson
95 % interval, failure labels, seeds). Default seeds are the smallbc dev range
10500-10549. Seeds 10000-10499 were spent on the ACT runs and are refused.

    python experiments/smallbc/eval.py --ckpt experiments/smallbc/runs/main/last.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEV_SEEDS, HERE, USED_TEST_SEEDS  # noqa: E402
from policy import SmallBCPolicy  # noqa: E402


def main() -> None:
    from sdp.envs.panda_pick_place import PandaPickPlaceEnv
    from sdp.eval.harness import run_condition, summarize, write_csv, write_summary
    from sdp.seeds import check_eval

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seed-start", type=int, default=DEV_SEEDS.start)
    p.add_argument("--n", type=int, default=len(DEV_SEEDS))
    p.add_argument("--distribution", default="nominal", choices=("nominal", "shifted"))
    p.add_argument("--n-exec", type=int, default=8)
    p.add_argument("--temporal-ensemble", action="store_true")
    p.add_argument("--te-m", type=float, default=0.01)
    p.add_argument("--no-binarize-gripper", action="store_true")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--condition", default=None, help="label in the CSV/JSON (default: run dir name)")
    p.add_argument("--out", default=None, help="output path without extension")
    args = p.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n))
    check_eval(seeds)
    spent = [s for s in seeds if s in USED_TEST_SEEDS]
    if spent:
        raise SystemExit(f"seeds {spent[0]}..{spent[-1]} are already-used test seeds (10000-10499)")

    ckpt = Path(args.ckpt)
    cond = args.condition or f"smallbc_{ckpt.parent.name}"
    policy = SmallBCPolicy(
        ckpt,
        n_exec=args.n_exec,
        temporal_ensemble=args.temporal_ensemble,
        te_m=args.te_m,
        binarize_gripper=not args.no_binarize_gripper,
        threads=args.threads,
    )
    tag = "te" if args.temporal_ensemble else f"x{args.n_exec}"
    out = Path(args.out) if args.out else (
        HERE / "results" / f"eval_{ckpt.parent.name}_{ckpt.stem}_{tag}_{args.distribution}_s{seeds[0]}_n{len(seeds)}"
    )
    env = PandaPickPlaceEnv(distribution=args.distribution, image_size=policy.image_size)
    t0 = time.perf_counter()
    try:
        results = run_condition(env, policy, seeds, args.distribution, cond)
    finally:
        env.close()
    write_csv(results, out.with_suffix(".csv"))
    summary = summarize(results)
    meta = {
        "checkpoint": str(ckpt),
        "checkpoint_step": policy.step_trained,
        "target_mode": policy.mode,
        "chunk": policy.chunk,
        "n_exec": args.n_exec,
        "temporal_ensemble": args.temporal_ensemble,
        "te_m": args.te_m,
        "binarize_gripper": not args.no_binarize_gripper,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    write_summary([summary], out.with_suffix(".json"), meta=meta)
    lo, hi = summary["wilson95"]
    print(
        f"{cond} {args.distribution} seeds {seeds[0]}-{seeds[-1]}: {summary['successes']}/{summary['trials']} "
        f"({100 * summary['success_rate']:.0f} %, Wilson 95 % {100 * lo:.0f}-{100 * hi:.0f} %) "
        f"labels {summary['labels']} -> {out.with_suffix('.json')}"
    )


if __name__ == "__main__":
    main()
