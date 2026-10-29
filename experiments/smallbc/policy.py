"""Harness-compatible policy for smallbc checkpoints (sdp.eval.harness.Policy).

``act(obs, env)``:

* every ``n_exec`` steps (default 8) the network predicts a K-step chunk of targets
  from the two images and the proprio vector (15-D state + the integrated TCP target
  at observation time); ``rel`` chunks are turned into absolute targets by adding that
  target (yaw wrapped);
* each step pops one absolute target [x, y, z, yaw, gripper] and converts it to the
  env's delta action with ``common.abs_to_delta``:
  ``clip((target - current integrated target) / max_step, -1, 1)``, yaw wrapped,
  roll/pitch 0, gripper binarised (sign) by default.

With ``temporal_ensemble=True`` it predicts every step and averages all chunks that
cover the current step with ACT's weights w_i = exp(-m * i), i = 0 for the oldest.

The policy reads ``env.target_pose()`` (the robot's own commanded pose, needed by the
delta interface) and never the cube state. ``predictor`` replaces the network with a
callable ``(obs, env, t) -> (K, 5)`` raw chunk, used by ``check_replay.py``.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from common import abs_to_delta, target_xyzyaw, wrap_pi  # noqa: E402


class SmallBCPolicy:
    def __init__(
        self,
        checkpoint: str | Path | None = None,
        n_exec: int = 8,
        temporal_ensemble: bool = False,
        te_m: float = 0.01,
        binarize_gripper: bool = True,
        threads: int = 1,
        predictor=None,
        target_mode: str | None = None,
        chunk: int | None = None,
    ):
        self.model = None
        if checkpoint is not None:
            import torch

            from model import load_checkpoint

            torch.set_num_threads(threads)
            self._torch = torch
            self.model, ckpt = load_checkpoint(checkpoint)
            self.mode = ckpt["target_mode"]
            self.chunk = int(self.model.cfg["chunk"])
            self.image_size = int(self.model.cfg["image_size"])
            self.step_trained = int(ckpt["step"])
        else:
            if predictor is None or target_mode is None or chunk is None:
                raise ValueError("need a checkpoint, or predictor + target_mode + chunk")
            self.mode, self.chunk, self.image_size = target_mode, int(chunk), None
        self.predictor = predictor
        self.n_exec = int(n_exec)
        if not 1 <= self.n_exec <= self.chunk:
            raise ValueError(f"n_exec must be in [1, {self.chunk}]")
        self.temporal_ensemble = temporal_ensemble
        self.te_m = te_m
        self.binarize_gripper = binarize_gripper
        self.reset()
        if self.model is not None:
            self._warm_up()

    def reset(self) -> None:
        self.queue: deque = deque()
        self.chunks: list[tuple[int, np.ndarray]] = []
        self.t = 0

    def _warm_up(self) -> None:
        s = self.image_size
        blank = np.zeros((s, s, 3), np.uint8)
        obs = {"front": blank, "wrist": blank, "state": np.zeros(15, np.float32)}
        self._network(obs, np.zeros(4))

    def _network(self, obs: dict, t_obs: np.ndarray) -> np.ndarray:
        torch = self._torch
        proprio = np.concatenate([obs["state"], t_obs]).astype(np.float32)[None]
        front = torch.from_numpy(np.ascontiguousarray(obs["front"]))[None]
        wrist = torch.from_numpy(np.ascontiguousarray(obs["wrist"]))[None]
        with torch.inference_mode():
            out = self.model.predict(front, wrist, torch.from_numpy(proprio))
        return out[0].numpy().astype(np.float64)

    def predict_chunk(self, obs: dict, env) -> np.ndarray:
        """Absolute (K, 5) target chunk for the current observation."""
        t_obs = target_xyzyaw(env)
        if self.model is not None:
            chunk = self._network(obs, t_obs)
        else:
            chunk = np.array(self.predictor(obs, env, self.t), dtype=np.float64)
        if self.mode == "rel":
            chunk[:, :3] += t_obs[:3]
            chunk[:, 3] = wrap_pi(chunk[:, 3] + t_obs[3])
        return chunk

    def act(self, obs: dict, env) -> np.ndarray:
        if self.temporal_ensemble:
            self.chunks.append((self.t, self.predict_chunk(obs, env)))
            self.chunks = [(s, c) for s, c in self.chunks if self.t - s < self.chunk]
            preds = np.stack([c[self.t - s] for s, c in self.chunks])  # oldest first
            w = np.exp(-self.te_m * np.arange(len(preds)))
            w /= w.sum()
            ref = preds[0, 3]
            preds[:, 3] = ref + wrap_pi(preds[:, 3] - ref)
            target = (w[:, None] * preds).sum(axis=0)
        else:
            if not self.queue:
                self.queue.extend(self.predict_chunk(obs, env)[: self.n_exec])
            target = self.queue.popleft()
        self.t += 1
        return abs_to_delta(env, target, self.binarize_gripper)
