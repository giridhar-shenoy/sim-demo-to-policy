"""Check 5: normalisation and input format, train vs eval (runs 5 and 2).

1. Reads policy_preprocessor.json / policy_postprocessor.json and the normalizer
   safetensors: norm mode and stats for images, state and action.
2. Feeds one decoded training frame through (a) the training-time path, replicated from
   lerobot_train.py (LeRobotDataset(return_uint8=True, torchcodec, ACT delta_timestamps)
   -> collate -> /255 -> preprocessor built from dataset stats with the ImageNet override,
   as make_dataset does for use_imagenet_stats=True), and (b) sdp/eval/act_policy.py's
   adapter (HWC uint8 -> _batch -> the checkpoint's saved preprocessor). Compares tensors.
Writes check5_normalisation.json.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common  # noqa: E402,I001
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

from safetensors.torch import load_file  # noqa: E402

from sdp.record.lerobot_writer import ACTION_NAMES, STATE_NAMES  # noqa: E402

RUNS = {"run5": (common.CKPT_V5, common.DS128), "run2": (common.CKPT_V2, common.DS96)}


def r(x, n=4):
    return np.round(np.asarray(x, dtype=np.float64).reshape(-1), n).tolist()


def stats_report(ckpt):
    pre = json.loads((ckpt / "policy_preprocessor.json").read_text())
    post = json.loads((ckpt / "policy_postprocessor.json").read_text())
    norm = next(s for s in pre["steps"] if s["registry_name"] == "normalizer_processor")
    unnorm = next(s for s in post["steps"] if s["registry_name"] == "unnormalizer_processor")
    st = load_file(str(ckpt / norm["state_file"]))
    st_post = load_file(str(ckpt / unnorm["state_file"]))
    feats = norm["config"]["features"]
    rep = {
        "preprocessor_steps": [s["registry_name"] for s in pre["steps"]],
        "postprocessor_steps": [s["registry_name"] for s in post["steps"]],
        "norm_map": norm["config"]["norm_map"],
        "normalised_features": {k: v["type"] for k, v in feats.items()},
        "image_mean": {k: r(st[f"{k}.mean"]) for k in feats if "images" in k},
        "image_std": {k: r(st[f"{k}.std"]) for k in feats if "images" in k},
        "action_mean": dict(zip(ACTION_NAMES, r(st["action.mean"]), strict=True)),
        "action_std": dict(zip(ACTION_NAMES, r(st["action.std"]), strict=True)),
        "post_action_stats_equal_pre": bool(
            torch.equal(st["action.mean"], st_post["action.mean"])
            and torch.equal(st["action.std"], st_post["action.std"])
        ),
    }
    if "observation.state" in feats:
        rep["state_mean"] = dict(zip(STATE_NAMES, r(st["observation.state.mean"]), strict=True))
        rep["state_std"] = dict(zip(STATE_NAMES, r(st["observation.state.std"], 6), strict=True))
    else:
        rep["state"] = "not a policy input (not normalised; ACT ignores the key)"
    return rep, st


def training_path(ckpt, root, idx):
    """Replicates lerobot_train.py: dataset item -> collate -> /255 -> fresh preprocessor."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import IMAGENET_STATS

    cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    meta = LeRobotDatasetMetadata(f"local/{root.name}", root=root)
    dts = resolve_delta_timestamps(cfg, meta)
    ds = LeRobotDataset(
        f"local/{root.name}", root=root, delta_timestamps=dts, video_backend="torchcodec",
        return_uint8=True,
    )
    for key in ds.meta.camera_keys:  # make_dataset with use_imagenet_stats=True
        for stype, s in IMAGENET_STATS.items():
            ds.meta.stats[key][stype] = torch.tensor(s, dtype=torch.float32)
    pre, _ = make_pre_post_processors(cfg, dataset_stats=ds.meta.stats)
    item = ds[idx]
    batch = torch.utils.data.default_collate([item])
    raw_info = {
        k: {"dtype": str(batch[k].dtype), "shape": list(batch[k].shape)}
        for k in ("observation.images.front", "observation.images.wrist", "observation.state")
    }
    for key in ds.meta.camera_keys:
        if batch[key].dtype == torch.uint8:
            batch[key] = batch[key].to(dtype=torch.float32) / 255.0
    pre_in = {k: batch[k].clone() for k in ds.meta.camera_keys}
    out = pre(batch)
    return item, out, pre_in, raw_info, dts


def main():
    print(f"MemAvailable {common.check_mem():.1f} GB")
    from sdp.eval.act_policy import ACTPolicyAdapter

    res = {}
    for run, (ckpt, root) in RUNS.items():
        rep, _ = stats_report(ckpt)
        ds_stats = json.loads((root / "meta/stats.json").read_text())
        rep["action_stats_equal_dataset_stats_json"] = bool(
            np.allclose(rep_mean := np.array(list(rep["action_mean"].values())), np.round(ds_stats["action"]["mean"], 4))
            and np.allclose(np.array(list(rep["action_std"].values())), np.round(ds_stats["action"]["std"], 4))
        )
        del rep_mean
        rep["dataset_image_stats_not_used_example_front_mean"] = r(
            np.array(ds_stats["observation.images.front"]["mean"]).reshape(-1)
        )
        adapter = ACTPolicyAdapter(ckpt)
        cmp = {}
        for idx in (0, 57):  # ep 0 frame 0 and a mid-episode frame
            item, train_out, pre_in, raw_info, dts = training_path(ckpt, root, idx)
            obs = {
                "front": common.hwc(item["observation.images.front"]),
                "wrist": common.hwc(item["observation.images.wrist"]),
                "state": item["observation.state"].numpy(),
            }
            b = adapter._batch(obs)
            adapter_in = {k: b[k].clone() for k in pre_in}
            ev = adapter.preprocessor(b)
            keys = [k for k in train_out if k.startswith("observation.")]
            cmp[f"index_{idx}"] = {
                "pre_norm_input_max_abs_diff": {
                    k: float((adapter_in[k] - pre_in[k]).abs().max()) for k in pre_in
                },
                "adapter_input_dtype_shape_range": {
                    k: [str(adapter_in[k].dtype), list(adapter_in[k].shape),
                        round(float(adapter_in[k].min()), 3), round(float(adapter_in[k].max()), 3)]
                    for k in adapter_in
                },
                "training_raw_batch": raw_info,
                "preprocessed_max_abs_diff": {
                    k: float((ev[k].float() - train_out[k].float()).abs().max()) for k in keys if k in ev
                },
                "keys_only_in_one_path": sorted(set(keys) ^ {k for k in ev if k.startswith("observation.")}),
            }
        rep["train_vs_adapter"] = cmp
        rep["delta_timestamps"] = {k: (len(v) if v else v) for k, v in (dts or {}).items()}
        rep["checkpoint_image_shape"] = adapter.image_size
        res[run] = rep
        print(run, json.dumps(rep, indent=1)[:4000])
        del adapter
    (common.OUT / "check5_normalisation.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
