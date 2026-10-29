"""Check 6: does the codec alone move the policy's output?

Run 5 (128 px, cameras only): first action (fresh chunk after reset) on the raw MuJoCo
render vs the decoded dataset frame, seed 0, frames 0-10 (raw frames saved by check 1;
the state is bit-identical, check 4). Scale references: the change when episode 0's
decoded images are swapped for episodes 1-9's (cube elsewhere), and the recorded action.
Same for run 2 (96 px, state + cameras). No env episodes.
Writes check6_policy_sensitivity.json.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common  # noqa: E402,I001
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

from sdp.record.lerobot_writer import ACTION_NAMES  # noqa: E402


def first_action(adapter, obs):
    adapter.reset()
    return adapter.act(obs)


def chunk(adapter, obs):
    """Full 20-step chunk, unnormalised."""
    with torch.inference_mode():
        b = adapter.preprocessor(adapter._batch(obs))
        a = adapter.policy.predict_action_chunk(b)  # (1, 20, 7), normalised
        return adapter.postprocessor(a).squeeze(0).numpy()


def main():
    print(f"MemAvailable {common.check_mem():.1f} GB")
    from sdp.eval.act_policy import ACTPolicyAdapter

    raw = np.load(common.OUT / "raw_seed0_frames.npz")
    out = {}
    for run, ckpt, root, size in (
        ("run5", common.CKPT_V5, common.DS128, "128"),
        ("run2", common.CKPT_V2, common.DS96, "96"),
    ):
        ad = ACTPolicyAdapter(ckpt)
        ds = common.load_dataset(root)
        starts, _ = common.episode_starts(root)
        per_frame = []
        chunk_diff = []
        for t in range(11):
            item = ds[t]
            st = item["observation.state"].numpy()
            o_raw = {"front": raw[f"{size}_front"][t], "wrist": raw[f"{size}_wrist"][t], "state": st}
            o_dec = {"front": common.hwc(item["observation.images.front"]),
                     "wrist": common.hwc(item["observation.images.wrist"]), "state": st}
            assert np.array_equal(o_dec["front"], raw[f"{size}_dec_front"][t])
            a_raw, a_dec = first_action(ad, o_raw), first_action(ad, o_dec)
            per_frame.append(np.abs(a_raw - a_dec))
            chunk_diff.append(np.abs(chunk(ad, o_raw) - chunk(ad, o_dec)).max(0))
            if t == 0:
                f0 = {"raw": a_raw.round(4).tolist(), "decoded": a_dec.round(4).tolist(),
                      "recorded": item["action"].numpy().round(4).tolist()}
        base = {"front": common.hwc(ds[0]["observation.images.front"]),
                "wrist": common.hwc(ds[0]["observation.images.wrist"]),
                "state": ds[0]["observation.state"].numpy()}
        a_base = first_action(ad, base)
        swaps, recs = [], []
        for s in starts[1:10]:
            it = ds[int(s)]
            sw = {**base, "front": common.hwc(it["observation.images.front"]),
                  "wrist": common.hwc(it["observation.images.wrist"])}
            swaps.append(np.abs(first_action(ad, sw) - a_base))
            recs.append(it["action"].numpy())
        per_frame, chunk_diff = np.array(per_frame), np.array(chunk_diff)
        out[run] = {
            "frame0_first_action": f0,
            "frame0_abs_change_raw_vs_decoded": dict(zip(ACTION_NAMES, per_frame[0].round(5).tolist(), strict=True)),
            "frames0-10_max_abs_change_first_action": dict(zip(ACTION_NAMES, per_frame.max(0).round(5).tolist(), strict=True)),
            "frames0-10_max_abs_change_whole_chunk": dict(zip(ACTION_NAMES, chunk_diff.max(0).round(5).tolist(), strict=True)),
            "reference_image_swap_max_abs_change_eps1-9": dict(zip(ACTION_NAMES, np.max(swaps, 0).round(5).tolist(), strict=True)),
            "reference_recorded_frame0_action_std_eps0-9": dict(
                zip(ACTION_NAMES, np.std([ds[0]["action"].numpy()] + recs, 0).round(3).tolist(), strict=True)
            ),
        }
        print(run, json.dumps(out[run], indent=1))
        del ad, ds
    (common.OUT / "check6_policy_sensitivity.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
