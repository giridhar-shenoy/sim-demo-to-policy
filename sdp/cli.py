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
    )
    ok = sum(r["success"] for r in records)
    print(f"wrote {len(records)} episodes to {args.out} ({ok} successful)")
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
    rec.set_defaults(func=_record)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
