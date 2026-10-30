"""What does a policy collide with? Replays the episodes an eval CSV labelled 'collision'.

The env flags a collision when an arm link or the hand touches the table, or when the
fingers press into it with more than 20 N (``PandaPickPlaceEnv._on_substep``). This
script hooks that check, and at the physics substep where the flag first turns on it
records which robot body touched the table, the contact normal force, the TCP height,
the finger gap, and the cube's horizontal distance from the TCP. Replays are
deterministic; each must reproduce its CSV label, or the script stops.

    python scripts/diagnose_collisions.py --checkpoint <ckpt> \\
        --csv results/eval_smallbc_abs1000_25k_s10800_shifted.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from sdp.envs import PandaPickPlaceEnv
from sdp.eval.harness import run_episode
from sdp.eval.policies import load_policy


def _table_contacts(env) -> list[dict]:
    m, d = env.model, env.data
    out = []
    force = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if env._table_gid not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == env._table_gid else c.geom1
        body = m.geom_bodyid[other]
        if body in env._robot_body_ids and m.body(body).name != "link0":
            mujoco.mj_contactForce(m, d, i, force)
            out.append({"body": m.body(body).name, "normal_force_n": round(float(force[0]), 2)})
    return out


def _x_range(rows: list[dict], label: str) -> list[float] | None:
    xs = [float(r["init_x"]) for r in rows if r["label"] == label]
    return [min(xs), max(xs)] if xs else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--csv", required=True, help="eval CSV with per-episode labels")
    ap.add_argument("--out", default=None, help="default: results/collisions_<csv stem>.json")
    args = ap.parse_args()

    all_rows = list(csv.DictReader(open(args.csv)))
    rows = [r for r in all_rows if r["label"] == "collision"]
    policy, policy_type = load_policy(args.checkpoint)
    base = PandaPickPlaceEnv(image_size=policy.image_size)
    env = policy.wrap_env(base)
    episodes = []
    try:
        for row in rows:
            first: dict = {}

            def hooked(orig=base._on_substep, first=first) -> None:
                before = base._episode.get("collision", False)
                orig()
                if not before and base._episode["collision"] and not first:
                    tcp = base.data.site_xpos[base._tcp_sid]
                    cube, _ = base.cube_pose()
                    first.update(
                        step=base._step_count,
                        contacts=_table_contacts(base),
                        tcp_z_m=round(float(tcp[2]), 4),
                        finger_gap_m=round(base.gripper_opening(), 4),
                        cube_xy_dist_m=round(float(np.linalg.norm(tcp[:2] - cube[:2])), 4),
                    )

            base._on_substep = hooked  # type: ignore[method-assign]
            res = run_episode(env, policy, int(row["seed"]), row["distribution"], "diag").result
            if res.label != row["label"] or res.steps != int(row["steps"]):
                raise SystemExit(f"seed {row['seed']}: replay gave {res.label}/{res.steps}")
            episodes.append(
                {
                    "seed": res.seed,
                    "init_x": float(row["init_x"]),
                    "init_y": float(row["init_y"]),
                    **first,
                }
            )
    finally:
        env.close()

    bodies = Counter(c["body"] for e in episodes for c in e["contacts"][:1])
    summary = {
        "checkpoint": str(args.checkpoint),
        "policy_type": policy_type,
        "csv": args.csv,
        "collision_episodes": len(episodes),
        "first_contact_body_counts": dict(bodies),
        "finger_press_episodes": sum(
            1 for e in episodes if all(c["body"].endswith("finger") for c in e["contacts"])
        ),
        "median_step_of_first_collision": float(np.median([e["step"] for e in episodes]))
        if episodes
        else None,
        "median_cube_xy_dist_m": float(np.median([e["cube_xy_dist_m"] for e in episodes]))
        if episodes
        else None,
        "collision_init_x_range_m": [
            min(e["init_x"] for e in episodes),
            max(e["init_x"] for e in episodes),
        ]
        if episodes
        else None,
        "success_init_x_range_m": _x_range(all_rows, "success"),
        "nominal_box_x_max_m": 0.60,
        "episodes": episodes,
    }
    out = Path(args.out or f"results/collisions_{Path(args.csv).stem.removeprefix('eval_')}.json")
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
