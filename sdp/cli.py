"""Command-line entry point: ``sdp record | train | eval | report``."""

from __future__ import annotations

import argparse
import sys

from sdp import __version__


def _record(args: argparse.Namespace) -> int:
    from sdp.expert.policy import ExpertPolicy
    from sdp.record.recorder import record_dataset
    from sdp.seeds import demo_seeds

    if args.source != "scripted":
        print("only --source scripted is implemented in v0.1 (teleop is deferred)")
        return 2
    records = record_dataset(
        args.out,
        args.repo_id,
        ExpertPolicy(),
        demo_seeds(args.episodes, start=args.seed_start),
        source=args.source,
        image_size=args.image_size,
        action_mode=args.action_mode,
    )
    ok = sum(r["success"] for r in records)
    print(f"wrote {len(records)} episodes to {args.out} ({ok} successful)")
    return 0


def _eval(args: argparse.Namespace, argv: list[str] | None = None) -> int:
    import platform
    import time

    from sdp.envs import PandaPickPlaceEnv
    from sdp.eval.act_policy import ACTPolicyAdapter
    from sdp.eval.harness import run_condition, summarize, write_csv, write_summary
    from sdp.seeds import eval_seeds

    t0 = time.perf_counter()
    policy = ACTPolicyAdapter(args.checkpoint, n_action_steps=args.n_action_steps)
    env = policy.wrap_env(PandaPickPlaceEnv(image_size=policy.image_size))
    seeds = eval_seeds(args.trials, start=args.seed_start)
    rows, summaries = [], []
    try:
        for dist in args.distributions:
            res = run_condition(env, policy, seeds, dist, args.condition)
            rows += res
            summaries.append(summarize(res))
            s = summaries[-1]
            lo, hi = s["wilson95"]
            print(
                f"{args.condition} {dist}: {s['successes']}/{s['trials']} "
                f"({100 * s['success_rate']:.0f} %, Wilson 95 % {100 * lo:.0f}-{100 * hi:.0f} %)"
            )
    finally:
        env.close()
    out = args.out or f"results/eval_{args.condition}_s{seeds[0]}"
    write_csv(rows, f"{out}.csv")
    meta = {
        "command": "sdp " + " ".join(sys.argv[1:] if argv is None else argv),
        "checkpoint": str(policy.path),
        "n_action_steps": policy.policy.config.n_action_steps,
        "chunk_size": policy.policy.config.chunk_size,
        "image_size": policy.image_size,
        "action_mode": policy.action_mode,
        "wall_time_s": round(time.perf_counter() - t0, 1),
        "python": platform.python_version(),
    }
    write_summary(summaries, f"{out}.json", meta=meta)
    return 0


def _report(args: argparse.Namespace) -> int:
    from sdp.eval.report import update_docs

    update_docs(args.results, args.targets)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sdp", description=__doc__)
    parser.add_argument("--version", action="version", version=f"sdp {__version__}")
    sub = parser.add_subparsers(dest="command")

    rec = sub.add_parser("record", help="record demonstrations as a LeRobot v3.0 dataset")
    rec.add_argument("--env", default="PandaPickPlace-v0", choices=["PandaPickPlace-v0"])
    rec.add_argument("--episodes", type=int, default=50)
    rec.add_argument("--seed-start", type=int, default=0, help="offset into demo seeds 0-999")
    rec.add_argument("--source", default="scripted", choices=["scripted", "teleop"])
    rec.add_argument("--out", required=True, help="dataset root directory (must not exist)")
    rec.add_argument("--repo-id", default="local/sdp_pickplace_scripted")
    rec.add_argument("--image-size", type=int, default=96, help="camera resolution (square)")
    rec.add_argument(
        "--action-mode",
        default="delta",
        choices=["delta", "abs_target"],
        help="record the 7-D delta (env-native) or the 5-D absolute TCP target",
    )
    rec.set_defaults(func=_record)

    ev = sub.add_parser("eval", help="evaluate an ACT checkpoint on held-out eval seeds")
    ev.add_argument("--checkpoint", required=True, help="training output dir or pretrained_model")
    ev.add_argument("--condition", required=True, help="row name, e.g. act_cpu_scripted50")
    ev.add_argument("--trials", type=int, default=100, help="episodes per distribution")
    ev.add_argument("--seed-start", type=int, default=0, help="offset into eval seeds 10000-19999")
    ev.add_argument("--distributions", nargs="+", default=["nominal", "shifted"])
    ev.add_argument("--n-action-steps", type=int, default=None, help="override the checkpoint's")
    ev.add_argument(
        "--out", default=None, help="output prefix (default results/eval_<condition>_s<seed>)"
    )
    ev.set_defaults(func=_eval)

    rep = sub.add_parser("report", help="regenerate the result tables in the docs")
    rep.add_argument("--results", default="results")
    rep.add_argument("--targets", nargs="+", default=["docs/results.md", "README.md"])
    rep.set_defaults(func=_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    if args.func is _eval:
        return _eval(args, argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
