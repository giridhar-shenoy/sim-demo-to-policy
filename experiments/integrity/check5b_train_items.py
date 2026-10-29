"""Check 5b: do training items pair the right images with the right action chunk?

Builds the dataset exactly as lerobot-train did for run 5 / run 2 (episodes=[0..49],
ACT delta_timestamps -> 20-step action chunk, torchcodec, return_uint8=True) and, for a
spread of indices, checks: (1) images equal the plain decoded frame at the same index
(which check 1 matched to the raw re-render for seeds 0-2); (2) the action chunk equals
the parquet actions t..t+19 of the same episode; (3) action_is_pad marks exactly the
frames past the episode end. Also prints run 5's predicted chunk at ep 0 frame 0 next
to the recorded chunk. Writes check5b_train_items.json.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common  # noqa: E402,I001
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)


def main():
    print(f"MemAvailable {common.check_mem():.1f} GB")
    import lerobot.policies.act.configuration_act  # noqa: F401  (registers "act")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    out = {}
    for run, ckpt, root in (("run5", common.CKPT_V5, common.DS128), ("run2", common.CKPT_V2, common.DS96)):
        cfg = PreTrainedConfig.from_pretrained(str(ckpt))
        meta = LeRobotDatasetMetadata(f"local/{root.name}", root=root)
        dts = resolve_delta_timestamps(cfg, meta)
        train_ds = LeRobotDataset(f"local/{root.name}", root=root, episodes=list(range(50)),
                                  delta_timestamps=dts, video_backend="torchcodec", return_uint8=True)
        plain = common.load_dataset(root)
        df = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pandas()
        acts = np.stack(df.action.values)
        starts, lengths = common.episode_starts(root)
        rows, ok = [], True
        idxs = [(e, f) for e in (0, 1, 25, 49) for f in (0, 5)]
        idxs += [(0, int(lengths[0]) - 1), (49, int(lengths[49]) - 10)]
        for e, f in idxs:
            i = int(starts[e]) + f
            it, pl = train_ds[i], plain[i]
            img_eq = all(torch.equal(it[k], pl[k]) for k in ("observation.images.front", "observation.images.wrist"))
            L = int(lengths[e])
            want = acts[int(starts[e]) + np.minimum(np.arange(f, f + 20), L - 1)]
            chunk_eq = bool(np.allclose(it["action"].numpy(), want))
            pad = it["action_is_pad"].numpy()
            pad_ok = bool((pad == (np.arange(f, f + 20) >= L)).all())
            ep_ok = int(it["episode_index"]) == e and int(it["frame_index"]) == f
            ok &= img_eq and chunk_eq and pad_ok and ep_ok
            rows.append({"episode": e, "frame": f, "images_equal_plain": img_eq,
                         "chunk_equals_parquet": chunk_eq, "pad_flags_ok": pad_ok,
                         "episode_frame_index_ok": ep_ok, "n_pad": int(pad.sum())})
        out[run] = {"all_ok": bool(ok), "rows": rows, "action_chunk_len": len(dts["action"])}
        print(run, "all_ok", ok)
        if run == "run5":
            from sdp.eval.act_policy import ACTPolicyAdapter

            ad = ACTPolicyAdapter(ckpt)
            it = plain[0]
            obs = {"front": common.hwc(it["observation.images.front"]),
                   "wrist": common.hwc(it["observation.images.wrist"]),
                   "state": it["observation.state"].numpy()}
            with torch.inference_mode():
                b = ad.preprocessor(ad._batch(obs))
                pred = ad.postprocessor(ad.policy.predict_action_chunk(b)).squeeze(0).numpy()
            out[run]["ep0_f0_chunk"] = {
                name: {"pred": pred[:, k].round(3).tolist(), "recorded": acts[:20, k].round(3).tolist()}
                for name, k in (("d_x", 0), ("d_y", 1), ("d_z", 2), ("d_yaw", 5), ("gripper", 6))
            }
            for name, v in out[run]["ep0_f0_chunk"].items():
                print(name, "pred", v["pred"][:16]); print(name, "rec ", v["recorded"][:16])
            # Frame-0 d_y is -0.6 in every demo: what does the policy say for eps 0-9?
            fy = []
            for e in range(10):
                it = plain[int(starts[e])]
                o = {"front": common.hwc(it["observation.images.front"]),
                     "wrist": common.hwc(it["observation.images.wrist"]), "state": obs["state"]}
                ad.reset()
                fy.append(float(ad.act(o)[1]))
            out[run]["frame0_d_y_pred_eps0-9"] = np.round(fy, 4).tolist()
            out[run]["frame0_d_y_recorded_all50"] = sorted(set(np.round(acts[starts, 1], 4).tolist()))
            del ad
        del train_ds, plain
    (common.OUT / "check5b_train_items.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
