"""Check 1: raw MuJoCo renders vs AV1-decoded dataset frames (train vs eval pixels).

Re-runs env + scripted expert for demo seeds 0-2 at 96 px exactly like
sdp/record/recorder.py:record_episode (obs_t is stored with the action chosen from it),
plus seed 0 at 128 px against the 128 px dataset. Env episodes: 4.
Writes check1_codec.json, check1_seed0_f{0,5}.png and raw_seed0_frames.npz (for check 6).
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common  # noqa: E402,I001
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

torch.set_num_threads(2)

from sdp.envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from sdp.expert.policy import ExpertPolicy  # noqa: E402

CAMS = ("front", "wrist")


def rollout(env, seed):
    pol = ExpertPolicy()
    obs, _ = env.reset(seed=seed, options={"distribution": "nominal"})
    pol.reset()
    imgs = {c: [] for c in CAMS}
    states, actions, phases = [], [], []
    while True:
        phases.append(pol.expert.phase_name)
        a = pol.act(obs, env)
        for c in CAMS:
            imgs[c].append(obs[c].copy())
        states.append(obs["state"].copy())
        actions.append(np.asarray(a, np.float32))
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    return {c: np.stack(v) for c, v in imgs.items()}, np.stack(states), np.stack(actions), phases, info


def decoded_episode(ds, start, length):
    out = {c: [] for c in CAMS}
    st, ac = [], []
    for i in range(start, start + length):
        it = ds[i]
        for c in CAMS:
            out[c].append(common.hwc(it[f"observation.images.{c}"]))
        st.append(it["observation.state"].numpy())
        ac.append(it["action"].numpy())
    return {c: np.stack(v) for c, v in out.items()}, np.stack(st), np.stack(ac)


def cube_stats(raw, dec):
    cr, cd = common.centroid(common.red_mask(raw)), common.centroid(common.red_mask(dec))
    shift = float(np.hypot(cr[0] - cd[0], cr[1] - cd[1])) if cr[2] and cd[2] else None
    # Looser threshold, to show sensitivity: R - max(G, B) > 50.
    loose = lambda im: int(  # noqa: E731
        ((im[..., 0].astype(int) - np.maximum(im[..., 1], im[..., 2]).astype(int)) > 50).sum()
    )
    return {
        "raw_px": cr[2],
        "dec_px": cd[2],
        "raw_px_loose": loose(raw),
        "dec_px_loose": loose(dec),
        "raw_centroid_uv": [round(cr[0], 2), round(cr[1], 2)],
        "dec_centroid_uv": [round(cd[0], 2), round(cd[1], 2)],
        "centroid_shift_px": None if shift is None else round(shift, 3),
        # Mean RGB of the cube's raw pixels, raw vs decoded at the same pixels.
        "raw_cube_rgb": raw[common.red_mask(raw)].mean(0).round(1).tolist() if cr[2] else None,
        "dec_rgb_at_raw_cube_px": dec[common.red_mask(raw)].mean(0).round(1).tolist()
        if cr[2]
        else None,
    }


def compare(raw_imgs, dec_imgs):
    res = {}
    for c in CAMS:
        r, d = raw_imgs[c].astype(np.int16), dec_imgs[c].astype(np.int16)
        diff = np.abs(r - d)
        n = min(len(r), len(d))
        # Frame alignment: which raw frame index best matches each decoded frame.
        offsets = []
        for t in range(n):
            cands = {k: np.abs(r[t + k] - d[t]).mean() for k in (-2, -1, 0, 1, 2) if 0 <= t + k < n}
            offsets.append(min(cands, key=cands.get))
        # Offsets only where the scene actually moves (raw MAE vs previous frame >= 1.0);
        # in static stretches every neighbour matches equally and the argmin is noise.
        motion = [np.abs(r[t] - r[t - 1]).mean() if t else 0.0 for t in range(n)]
        moving = [offsets[t] for t in range(n) if motion[t] >= 1.0]
        per_frame = diff.reshape(len(diff), -1).mean(1)
        res[c] = {
            "mean_abs_diff": round(float(diff.mean()), 3),
            "max_abs_diff": int(diff.max()),
            "p99_abs_diff": float(np.percentile(diff, 99)),
            "psnr_db": round(common.psnr(raw_imgs[c], dec_imgs[c]), 2),
            "best_match_offset_counts": {str(k): offsets.count(k) for k in sorted(set(offsets))},
            "best_match_offset_counts_moving": {str(k): moving.count(k) for k in sorted(set(moving))},
            # GOP is 2 (video.g = 2): even frames are keyframes, odd frames predicted.
            "mean_abs_diff_even_frames": round(float(per_frame[0::2].mean()), 3),
            "mean_abs_diff_odd_frames": round(float(per_frame[1::2].mean()), 3),
        }
    return res


def save_png(raw_imgs, dec_imgs, t, path, scale=4, amp=4):
    rows = []
    for c in CAMS:
        r, d = raw_imgs[c][t], dec_imgs[c][t]
        diff = np.clip(np.abs(r.astype(int) - d.astype(int)) * amp, 0, 255).astype(np.uint8)
        rows.append(np.concatenate([r, d, diff], axis=1))
    img = np.concatenate(rows, axis=0)
    Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST).save(
        path
    )


def main():
    print(f"MemAvailable {common.check_mem():.1f} GB")
    out = {"seeds_96px": {}, "seed0_128px": None, "notes": []}
    ds = common.load_dataset(common.DS96)
    starts, lengths = common.episode_starts(common.DS96)
    env = PandaPickPlaceEnv(image_size=96)
    raw_seed0 = {}
    all_raw = {c: [] for c in CAMS}
    all_dec = {c: [] for c in CAMS}
    for seed in (0, 1, 2):
        raw, st, ac, phases, info = rollout(env, seed)
        dec, dst, dac = decoded_episode(ds, int(starts[seed]), int(lengths[seed]))
        assert len(raw["front"]) == lengths[seed], (len(raw["front"]), lengths[seed])
        for c in CAMS:
            all_raw[c].append(raw[c])
            all_dec[c].append(dec[c])
        rec = {
            "length_rerun": len(st),
            "length_dataset": int(lengths[seed]),
            "success_rerun": bool(info["success"]),
            "state_max_abs_diff_vs_dataset": float(np.abs(st - dst).max()),
            "action_max_abs_diff_vs_dataset": float(np.abs(ac - dac).max()),
            "approach_frames": phases.count("above_cube"),
            "pixels": compare(raw, dec),
            "cube": {
                f"{c}_f{t}": cube_stats(raw[c][t], dec[c][t]) for c in CAMS for t in (0, 5, 10)
            },
        }
        # Over every frame in which the raw image shows the cube (>= 5 px).
        for c in CAMS:
            shifts, ratios = [], []
            for t in range(len(st)):
                m_r, m_d = common.red_mask(raw[c][t]), common.red_mask(dec[c][t])
                if m_r.sum() >= 5:
                    cr, cd = common.centroid(m_r), common.centroid(m_d)
                    ratios.append(cd[2] / cr[2])
                    if cd[2]:
                        shifts.append(np.hypot(cr[0] - cd[0], cr[1] - cd[1]))
            rec[f"cube_all_frames_{c}"] = {
                "frames_with_cube": len(ratios),
                "dec_over_raw_px_ratio_mean": round(float(np.mean(ratios)), 3),
                "dec_over_raw_px_ratio_min": round(float(np.min(ratios)), 3),
                "centroid_shift_px_mean": round(float(np.mean(shifts)), 3),
                "centroid_shift_px_max": round(float(np.max(shifts)), 3),
            }
        out["seeds_96px"][str(seed)] = rec
        if seed == 0:
            for t in (0, 5):
                save_png(raw, dec, t, common.OUT / f"check1_seed0_f{t}.png")
            raw_seed0["96"] = {c: raw[c][:11] for c in CAMS}
            raw_seed0["96_dec"] = {c: dec[c][:11] for c in CAMS}
        print(f"seed {seed} done: {json.dumps(rec['pixels'])}")
    env.close()
    out["pooled_96px_seeds0-2"] = compare(
        {c: np.concatenate(all_raw[c]) for c in CAMS}, {c: np.concatenate(all_dec[c]) for c in CAMS}
    )
    del ds

    # Seed 0 at 128 px against the 128 px dataset (run 5's input).
    ds128 = common.load_dataset(common.DS128)
    s128, l128 = common.episode_starts(common.DS128)
    env = PandaPickPlaceEnv(image_size=128)
    raw, st, ac, phases, info = rollout(env, 0)
    env.close()
    dec, dst, dac = decoded_episode(ds128, int(s128[0]), int(l128[0]))
    out["seed0_128px"] = {
        "state_max_abs_diff_vs_dataset": float(np.abs(st - dst).max()),
        "pixels": compare(raw, dec),
        "cube": {f"{c}_f{t}": cube_stats(raw[c][t], dec[c][t]) for c in CAMS for t in (0, 5, 10)},
    }
    raw_seed0["128"] = {c: raw[c][:11] for c in CAMS}
    raw_seed0["128_dec"] = {c: dec[c][:11] for c in CAMS}
    np.savez_compressed(
        common.OUT / "raw_seed0_frames.npz",
        **{f"{k}_{c}": d[c] for k, d in raw_seed0.items() for c in CAMS},
    )
    (common.OUT / "check1_codec.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["pooled_96px_seeds0-2"], indent=1))
    print(json.dumps(out["seed0_128px"]["pixels"], indent=1))


if __name__ == "__main__":
    main()
