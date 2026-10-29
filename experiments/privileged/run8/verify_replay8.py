"""Replay run 8's clean labels through AbsoluteTargetWrapper on the un-noised (even) seeds.

Every clean episode in the dataset must succeed with the same length. Noisy episodes
are skipped: their labels are relative to the noisy trajectory's targets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from priv_common import AbsoluteTargetWrapper  # noqa: E402
from verify_replay import load_actions  # noqa: E402

from sdp.envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from sdp.eval.taxonomy import label_episode  # noqa: E402
from sdp.record.lerobot_writer import read_sidecar  # noqa: E402


def main() -> None:
    root = HERE / "datasets" / "pickplace_abs_dart_200"
    side = read_sidecar(root)
    acts = load_actions(root)
    assert len(side) == len(acts)
    base = PandaPickPlaceEnv(render_obs=False)
    env = AbsoluteTargetWrapper(base)
    rows = []
    for rec, a_abs in zip(side, acts, strict=True):
        if rec["dart"]:
            continue
        env.reset(seed=rec["seed"], options={"distribution": "nominal"})
        for t, a in enumerate(a_abs):
            _, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
        rows.append({"seed": rec["seed"], "success": bool(info["success"]),
                     "label": label_episode(info["episode"], info),
                     "steps": t + 1, "recorded_steps": len(a_abs)})
    base.close()
    out = {"clean_episodes_replayed": len(rows), "successes": sum(r["success"] for r in rows),
           "steps_match": all(r["steps"] == r["recorded_steps"] for r in rows), "rows": rows}
    (HERE / "results" / "replay_clean_labels.json").write_text(json.dumps(out, indent=2) + "\n")
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
