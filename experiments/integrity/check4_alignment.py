"""Check 4: recorder alignment by open-loop replay of recorded actions.

For demo seeds 0-9: fresh env (no rendering; physics does not depend on it), reset with
the episode's seed, apply the recorded action sequence open-loop, and compare every
observation.state with the dataset's. A shadow scripted expert is queried before each
step (it only reads env state): if action[t] was chosen from obs_t, the expert's action
equals the recorded one at every t. Off-by-one controls (seeds 0-1): replay action[t+1]
or action[t-1] at step t. Env episodes: 10 + 4 = 14.
Also checks frame_index / timestamp / next.done / next.success bookkeeping.
Writes check4_alignment.json (incl. the expert phase per frame for seeds 0-9).
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common  # noqa: E402,I001
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from sdp.envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from sdp.expert.waypoint import WaypointExpert  # noqa: E402
from sdp.record.lerobot_writer import read_sidecar  # noqa: E402


def load_table(root):
    df = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pandas()
    return df


def replay(seed, actions, states, shadow=True):
    env = PandaPickPlaceEnv(image_size=96, render_obs=False)
    obs, _ = env.reset(seed=seed, options={"distribution": "nominal"})
    exp = WaypointExpert()
    exp.reset()
    div = [float(np.abs(obs["state"] - states[0]).max())]
    exp_err, phases = [], []
    info = {}
    n = len(actions)
    for t in range(n):
        if shadow:
            phases.append(exp.phase_name)
            exp_err.append(float(np.abs(exp.act(env) - actions[t]).max()))
        obs, _, term, trunc, info = env.step(actions[t])
        if t + 1 < n:
            div.append(float(np.abs(obs["state"] - states[t + 1]).max()))
        if (term or trunc) and t + 1 < n:
            break
    env.close()
    return {
        "success": bool(info.get("success", False)),
        "terminated_at_step": int(env._step_count),
        "label_episode": info.get("episode", {}),
        "max_state_div": max(div),
        "state_div_per_frame": div,
        "max_shadow_expert_action_err": max(exp_err) if exp_err else None,
        "phases": phases,
    }


def main():
    print(f"MemAvailable {common.check_mem():.1f} GB")
    df = load_table(common.DS96)
    side = read_sidecar(common.DS96)
    out = {"episodes": {}, "off_by_one": {}, "bookkeeping": {}}

    # Bookkeeping over all 50 episodes.
    bad_fi = bad_ts = bad_done = bad_succ = 0
    for rec in side:
        e = df[df.episode_index == rec["episode_index"]]
        n = len(e)
        assert n == rec["length"]
        bad_fi += int((e.frame_index.values != np.arange(n)).any())
        bad_ts += int(np.abs(e.timestamp.values - np.arange(n) / 20.0).max() > 1e-4)
        done = e["next.done"].values
        succ = e["next.success"].values
        bad_done += int(not (done[-1] and not done[:-1].any()))
        bad_succ += int(not (succ[-1] and not succ[:-1].any()))
    out["bookkeeping"] = {
        "episodes": len(side),
        "frame_index_mismatch_eps": bad_fi,
        "timestamp_mismatch_eps": bad_ts,
        "done_not_only_last_eps": bad_done,
        "success_not_only_last_eps": bad_succ,
        "seeds_equal_episode_index": all(r["seed"] == r["episode_index"] for r in side),
    }
    print(out["bookkeeping"])

    for ep in range(10):
        e = df[df.episode_index == ep]
        acts = np.stack(e.action.values).astype(np.float32)
        sts = np.stack(e["observation.state"].values).astype(np.float32)
        r = replay(side[ep]["seed"], acts, sts)
        out["episodes"][str(ep)] = {k: v for k, v in r.items() if k != "state_div_per_frame"}
        out["episodes"][str(ep)]["length"] = len(acts)
        out["episodes"][str(ep)]["approach_frames"] = r["phases"].count("above_cube")
        print(
            f"ep {ep}: success {r['success']} steps {r['terminated_at_step']}/{len(acts)} "
            f"max state div {r['max_state_div']:.2e} shadow-expert err "
            f"{r['max_shadow_expert_action_err']:.2e} approach {r['phases'].count('above_cube')}"
        )

    for ep in (0, 1):
        e = df[df.episode_index == ep]
        acts = np.stack(e.action.values).astype(np.float32)
        sts = np.stack(e["observation.state"].values).astype(np.float32)
        early = np.concatenate([acts[1:], acts[-1:]])  # action[t+1] applied at t
        late = np.concatenate([acts[:1], acts[:-1]])  # action[t-1] applied at t
        for name, a in (("action_t_plus_1", early), ("action_t_minus_1", late)):
            r = replay(side[ep]["seed"], a, sts, shadow=False)
            div = np.array(r["state_div_per_frame"])
            out["off_by_one"][f"ep{ep}_{name}"] = {
                "success": r["success"],
                "max_state_div": round(r["max_state_div"], 4),
                "state_div_frame10": round(float(div[min(10, len(div) - 1)]), 4),
                "label_episode": r["label_episode"],
            }
            print(ep, name, out["off_by_one"][f"ep{ep}_{name}"]["success"], r["max_state_div"])

    eps = out["episodes"].values()
    out["summary"] = {
        "success": f"{sum(r['success'] for r in eps)}/10",
        "max_state_div_all": max(r["max_state_div"] for r in eps),
        "max_shadow_expert_action_err_all": max(r["max_shadow_expert_action_err"] for r in eps),
        "terminated_exactly_at_length": all(r["terminated_at_step"] == r["length"] for r in eps),
    }
    print(out["summary"])
    (common.OUT / "check4_alignment.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
