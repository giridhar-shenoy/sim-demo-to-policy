"""Extra probe (follows from check 5b): does run 5's predicted 20-step chunk vary along
the chunk, and does its d_y track the episode-invariant d_y = -0.6 of the approach?

Decoded training frames of episode 0 (frames 0, 7, 14, 30, 40, 44, 60, 80) -> full chunk.
Per frame: std of the prediction along the 20 chunk positions vs the recorded chunk's.
Writes check7_chunk_probe.json. No env episodes.
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

DIMS = (("d_x", 0), ("d_y", 1), ("d_z", 2), ("d_yaw", 5), ("gripper", 6))


def main():
    print(f"MemAvailable {common.check_mem():.1f} GB")
    from sdp.eval.act_policy import ACTPolicyAdapter

    out = {}
    for run, ckpt, root in (("run5", common.CKPT_V5, common.DS128), ("run2", common.CKPT_V2, common.DS96)):
        ad = ACTPolicyAdapter(ckpt)
        ds = common.load_dataset(root)
        acts = np.stack(pq.read_table(root / "data/chunk-000/file-000.parquet").to_pandas().action.values)
        L = 113  # episode 0
        rows = {}
        for f in (0, 7, 14, 30, 40, 44, 60, 80):
            it = ds[f]
            obs = {"front": common.hwc(it["observation.images.front"]),
                   "wrist": common.hwc(it["observation.images.wrist"]), "state": it["observation.state"].numpy()}
            with torch.inference_mode():
                b = ad.preprocessor(ad._batch(obs))
                pred = ad.postprocessor(ad.policy.predict_action_chunk(b)).squeeze(0).numpy()
            rec = acts[np.minimum(np.arange(f, f + 20), L - 1)]
            rows[str(f)] = {
                n: {"pred_first": round(float(pred[0, k]), 3), "rec_first": round(float(rec[0, k]), 3),
                    "pred_std_along_chunk": round(float(pred[:, k].std()), 4),
                    "rec_std_along_chunk": round(float(rec[:, k].std()), 4)}
                for n, k in DIMS
            }
        out[run] = rows
        for f, r in rows.items():
            print(run, f, {n: (v["pred_first"], v["rec_first"], v["pred_std_along_chunk"], v["rec_std_along_chunk"]) for n, v in r.items()})
        del ad, ds
    (common.OUT / "check7_chunk_probe.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
