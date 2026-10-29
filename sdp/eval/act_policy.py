"""Load a LeRobot ACT checkpoint and expose it as a harness ``Policy``.

LeRobot 0.6.1 keeps normalisation outside the model, in processor pipelines saved
next to the weights (``policy_preprocessor.json`` / ``policy_postprocessor.json``).
The adapter mirrors ``lerobot_eval``: observation -> preprocessor -> ``select_action``
-> postprocessor. ``select_action`` keeps an internal action queue (``n_action_steps``
from the checkpoint config), cleared by ``reset()`` at the start of every episode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sdp.record.lerobot_writer import STATE_NAMES


def resolve_checkpoint(path: str | Path) -> Path:
    """Accept a run dir, a checkpoints/<step> dir, or the pretrained_model dir itself."""
    path = Path(path)
    for candidate in (
        path,
        path / "pretrained_model",
        path / "checkpoints" / "last" / "pretrained_model",
    ):
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return candidate
    raise FileNotFoundError(f"no LeRobot pretrained_model under {path}")


class ACTPolicyAdapter:
    def __init__(
        self, checkpoint: str | Path, device: str = "cpu", n_action_steps: int | None = None
    ):
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self._torch = torch
        self.path = resolve_checkpoint(checkpoint)
        self.policy = ACTPolicy.from_pretrained(str(self.path))
        if n_action_steps is not None:
            self.policy.config.n_action_steps = n_action_steps
        self.policy.to(device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.path),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self._warm_up()

    @property
    def image_size(self) -> int:
        """Camera resolution the checkpoint was trained on (square images)."""
        return int(self.policy.config.image_features["observation.images.front"].shape[-1])

    def _warm_up(self) -> None:
        """One forward pass on a blank observation, so the first episode's timing does
        not include torch's lazy initialisation (~0.6 s on the laptop)."""
        blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        state = np.zeros(len(STATE_NAMES), np.float32)  # ignored if the policy has no state input
        self.act({"front": blank, "wrist": blank, "state": state})
        self.reset()

    def reset(self) -> None:
        self.policy.reset()

    def _batch(self, obs: dict) -> dict:
        torch = self._torch

        def image(img: np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
            return t.unsqueeze(0)

        return {
            "observation.images.front": image(obs["front"]),
            "observation.images.wrist": image(obs["wrist"]),
            "observation.state": torch.from_numpy(obs["state"]).float().unsqueeze(0),
        }

    def act(self, obs: dict, env=None) -> np.ndarray:
        with self._torch.inference_mode():
            batch = self.preprocessor(self._batch(obs))
            action = self.policy.select_action(batch)
            action = self.postprocessor(action)
        return action.squeeze(0).cpu().numpy().astype(np.float32)
