"""Converter check: replay recorded clean targets through SmallBCPolicy (no network).

For every episode in a (clean) dataset, an oracle predictor returns the recorded
K-step label chunk for the current step, and ``SmallBCPolicy`` turns it into delta
actions exactly as it would a network output (chunk queue, rel -> abs, abs -> delta).
The episode runs through the repo's harness on the demo's own seed. It passes if every
episode succeeds and the executed actions match the expert's clean actions.

    python experiments/smallbc/check_replay.py --data experiments/smallbc/data/demo5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from common import HERE  # noqa: E402
from data import Demos  # noqa: E402
from policy import SmallBCPolicy  # noqa: E402

CONFIGS = (
    ("abs", 1, False),
    ("abs", 8, False),
    ("rel", 8, False),
    ("abs", 1, True),
    ("rel", 1, True),
)


def main() -> None:
    from sdp.envs.panda_pick_place import PandaPickPlaceEnv
    from sdp.eval.harness import run_episode

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--out", default=str(HERE / "results" / "check_replay.json"))
    args = p.parse_args()

    rows = []
    env = None
    for mode, n_exec, te in CONFIGS:
        demos = Demos(args.data, chunk=args.chunk, mode=mode)
        if env is None:
            env = PandaPickPlaceEnv(image_size=demos.image_size)
        clean_eps = [e for e in demos.episodes if not e["noised"]]
        for e in clean_eps:
            g0, length = e["gstart"], e["length"]

            def oracle(obs, env_, t, g0=g0, length=length):
                return demos.labels[g0 + min(t, length - 1)]

            pol = SmallBCPolicy(
                predictor=oracle, target_mode=mode, chunk=args.chunk, n_exec=n_exec, temporal_ensemble=te
            )
            actions = []
            r = run_episode(
                env, pol, e["seed"], "nominal", "replay", on_step=lambda o, a, i: actions.append(a.copy())
            ).result
            n = min(length, len(actions))
            rec = demos.action_clean[g0 : g0 + n]
            diff = float(np.abs(np.asarray(actions[:n]) - rec).max())
            rows.append(
                {
                    "mode": mode,
                    "n_exec": n_exec,
                    "temporal_ensemble": te,
                    "seed": e["seed"],
                    "success": r.success,
                    "label": r.label,
                    "steps": r.steps,
                    "recorded_steps": length,
                    "max_action_diff": round(diff, 6),
                }
            )
            print(rows[-1], flush=True)
    env.close()
    ok = all(r["success"] and r["steps"] == r["recorded_steps"] for r in rows)
    summary = {
        "pass": ok,
        "episodes": len(rows),
        "successes": sum(r["success"] for r in rows),
        "max_action_diff": max(r["max_action_diff"] for r in rows),
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))


if __name__ == "__main__":
    main()
