"""Load smallbc shards, build action-chunk labels, split episodes, normalise.

Images stay on disk as memory-mapped ``.npy`` files; the small per-frame arrays
(state, targets) are concatenated in RAM.

Labels for the frame at step t of an episode of length L are the K targets recorded
at steps t, t+1, ..., t+K-1 (the last one repeated past the end):

* ``abs``: [x, y, z, yaw, gripper] of the integrated TCP target after each clean action;
* ``rel``: the same with [x, y, z, yaw] taken relative to the target at observation
  time t (yaw difference wrapped), gripper unchanged.

The proprio input is the 15-D env state plus the integrated target at observation
time [x, y, z, yaw] (19-D).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TARGET_DIM = 5
PROPRIO_DIM = 19


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def make_chunks(target_abs, target_obs, ep_start, ep_len, t_in_ep, chunk, mode):
    """(N, K, 5) label chunks for frames with the given episode bookkeeping."""
    k = np.arange(chunk)[None, :]
    rows = ep_start[:, None] + np.minimum(t_in_ep[:, None] + k, ep_len[:, None] - 1)
    out = target_abs[rows].astype(np.float32)  # (N, K, 5)
    if mode == "rel":
        out[..., :4] -= target_obs[ep_start + t_in_ep][:, None, :4]
        out[..., 3] = wrap_pi(out[..., 3])
    elif mode != "abs":
        raise ValueError(f"unknown target mode {mode!r}")
    return out


class Demos:
    # Per-frame arrays, concatenated over shards in __init__ (set with setattr).
    state: np.ndarray
    target_obs: np.ndarray
    target_abs: np.ndarray
    action_clean: np.ndarray
    phase: np.ndarray

    def __init__(self, roots, chunk: int = 16, mode: str = "abs"):
        roots = [Path(r) for r in ([roots] if isinstance(roots, (str, Path)) else roots)]
        shards = []
        for r in roots:  # a record.py output dir, or a single shard dir
            cand = [r] if (r / "episodes.json").is_file() else sorted(r.glob("shard_*"))
            shards += [d for d in cand if (d / "episodes.json").is_file()]
        if not shards:
            raise FileNotFoundError(f"no shards under {roots}")
        self.front, self.wrist = [], []
        small: dict[str, list] = {
            k: [] for k in ("state", "target_obs", "target_abs", "action_clean", "phase")
        }
        eps = []
        shard_of, local_of = [], []
        g = 0
        for si, d in enumerate(shards):
            self.front.append(np.load(d / "front.npy", mmap_mode="r"))
            self.wrist.append(np.load(d / "wrist.npy", mmap_mode="r"))
            for k in small:
                small[k].append(np.load(d / f"{k}.npy"))
            n = len(small["state"][-1])
            shard_of.append(np.full(n, si, np.int32))
            local_of.append(np.arange(n, dtype=np.int64))
            for e in json.loads((d / "episodes.json").read_text()):
                eps.append({**e, "gstart": g + e["start"], "shard": si})
            g += n
        for k, v in small.items():
            setattr(self, k, np.concatenate(v))
        self.shard_of = np.concatenate(shard_of)
        self.local_of = np.concatenate(local_of)
        self.episodes = eps
        self.n_frames = g
        self.image_size = int(self.front[0].shape[1])
        self.chunk, self.mode = chunk, mode

        ep_start = np.zeros(g, np.int64)
        ep_len = np.zeros(g, np.int64)
        t_in = np.zeros(g, np.int64)
        self.ep_id = np.zeros(g, np.int64)
        for i, e in enumerate(eps):
            sl = slice(e["gstart"], e["gstart"] + e["length"])
            ep_start[sl], ep_len[sl], t_in[sl] = e["gstart"], e["length"], np.arange(e["length"])
            self.ep_id[sl] = i
        self.labels = make_chunks(
            self.target_abs, self.target_obs, ep_start, ep_len, t_in, chunk, mode
        )
        self.proprio = np.concatenate([self.state, self.target_obs], axis=1).astype(np.float32)

    def split(
        self, val_frac: float = 0.1, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Frame indices (train, val) and the val episodes' seeds. Split is by episode."""
        n = len(self.episodes)
        n_val = int(round(val_frac * n)) if n > 1 else 0
        if val_frac > 0 and n > 1:
            n_val = max(1, n_val)
        perm = np.random.default_rng(seed).permutation(n)
        val_eps = np.sort(perm[:n_val])
        is_val = np.isin(self.ep_id, val_eps)
        return (
            np.flatnonzero(~is_val),
            np.flatnonzero(is_val),
            [int(self.episodes[i]["seed"]) for i in val_eps],
        )

    def images(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(B, S, S, 3) uint8 front and wrist images for global frame indices."""
        f = np.empty((len(idx), self.image_size, self.image_size, 3), np.uint8)
        w = np.empty_like(f)
        for j, gi in enumerate(idx):
            s, li = self.shard_of[gi], self.local_of[gi]
            f[j] = self.front[s][li]
            w[j] = self.wrist[s][li]
        return f, w


def norm_stats(x: np.ndarray, axis=0, min_std: float = 1e-3) -> dict[str, np.ndarray]:
    mean = x.mean(axis=axis).astype(np.float32)
    std = np.maximum(x.std(axis=axis), min_std).astype(np.float32)
    return {"mean": mean, "std": std}
