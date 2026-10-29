"""Harness ``Policy`` for the privileged-state ACT checkpoints (no cameras).

Modelled on sdp/eval/act_policy.py: observation -> preprocessor -> select_action ->
postprocessor. The batch carries observation.state (from the env's obs) and
observation.environment_state (the cube pose, read from the env, see priv_common).
For the absolute-target checkpoint the returned action is the 5-D absolute target;
run it in an ``AbsoluteTargetWrapper`` env, which converts it to the env's delta.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from priv_common import env_state  # noqa: E402

from sdp.eval.act_policy import resolve_checkpoint  # noqa: E402


class PrivilegedACTAdapter:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self._torch = torch
        self.path = resolve_checkpoint(checkpoint)
        self.policy = ACTPolicy.from_pretrained(str(self.path))
        self.policy.to(device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.path),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.act_from_arrays(np.zeros(15, np.float32), np.zeros(5, np.float32))  # warm-up
        self.reset()

    def reset(self) -> None:
        self.policy.reset()

    def _batch(self, state: np.ndarray, env_st: np.ndarray) -> dict:
        t = self._torch
        return {
            "observation.state": t.from_numpy(np.asarray(state, np.float32)).unsqueeze(0),
            "observation.environment_state": t.from_numpy(np.asarray(env_st, np.float32)).unsqueeze(0),
        }

    def act_from_arrays(self, state: np.ndarray, env_st: np.ndarray) -> np.ndarray:
        with self._torch.inference_mode():
            batch = self.preprocessor(self._batch(state, env_st))
            action = self.postprocessor(self.policy.select_action(batch))
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def chunk_from_arrays(self, state: np.ndarray, env_st: np.ndarray) -> np.ndarray:
        """Full (chunk_size, action_dim) chunk, unnormalised."""
        with self._torch.inference_mode():
            batch = self.preprocessor(self._batch(state, env_st))
            chunk = self.policy.predict_action_chunk(batch)  # (1, chunk, dim), normalised
            chunk = self.postprocessor(chunk)
        return chunk.squeeze(0).cpu().numpy()

    def act(self, obs: dict, env) -> np.ndarray:
        return self.act_from_arrays(obs["state"], env_state(env.unwrapped))
