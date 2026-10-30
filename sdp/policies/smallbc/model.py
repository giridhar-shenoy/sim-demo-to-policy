"""Small from-scratch visuomotor BC policy (robomimic / Diffusion-Policy-style encoder).

Per camera: 4 conv layers (GroupNorm + ReLU), strides 2-2-2-1, so a 96 px image gives a
12x12 map (128 px -> 16x16), then a spatial-softmax head: a 1x1 conv to ``num_kp``
heatmaps, a softmax over each heatmap with a learned temperature, and the expected
(x, y) of each heatmap in [-1, 1]. Each camera yields 2 * num_kp coordinates.

Head: [front keypoints, wrist keypoints, (normalised proprio)] -> MLP -> K x 5 chunk of
normalised targets. Target / proprio mean-std live in buffers, so a checkpoint is
self-contained.

Training-only augmentation: DrQ random shift (replicate-pad by ``shift_pad`` px, random
crop back to the original size), independently per sample and per camera.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def conv_gn(cin: int, cout: int, k: int, s: int, groups: int = 8) -> list[nn.Module]:
    return [
        nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
        nn.GroupNorm(groups, cout),
        nn.ReLU(inplace=True),
    ]


class SpatialSoftmax(nn.Module):
    pos: torch.Tensor  # registered buffer

    def __init__(self, in_ch: int, h: int, w: int, num_kp: int = 32):
        super().__init__()
        self.kp = nn.Conv2d(in_ch, num_kp, 1)
        self.log_temp = nn.Parameter(torch.zeros(1))  # temperature = exp(log_temp), init 1
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        self.register_buffer("pos", torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1))  # (HW, 2)
        self.num_kp = num_kp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.kp(x)
        b, k, h, w = x.shape
        att = torch.softmax(x.reshape(b, k, h * w) / self.log_temp.exp(), dim=-1)
        return (att @ self.pos).reshape(b, 2 * k)  # (B, 2K): x0, y0, x1, y1, ...


class Encoder(nn.Module):
    def __init__(self, image_size: int = 96, width: int = 32, num_kp: int = 32):
        super().__init__()
        c = width
        self.conv = nn.Sequential(
            *conv_gn(3, c, 5, 2),
            *conv_gn(c, 2 * c, 3, 2),
            *conv_gn(2 * c, 2 * c, 3, 2),
            *conv_gn(2 * c, 2 * c, 3, 1),
        )
        with torch.no_grad():
            h = self.conv(torch.zeros(1, 3, image_size, image_size)).shape[-1]
        self.map_size = int(h)
        self.ssm = SpatialSoftmax(2 * c, h, h, num_kp)
        self.out_dim = 2 * num_kp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ssm(self.conv(x))


def random_shift(x: torch.Tensor, pad: int) -> torch.Tensor:
    """DrQ augmentation: replicate-pad, random integer crop back to (H, W)."""
    if pad <= 0:
        return x
    n, _, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (n,)).tolist()
    oy = torch.randint(0, 2 * pad + 1, (n,)).tolist()
    return torch.stack([xp[i, :, oy[i] : oy[i] + h, ox[i] : ox[i] + w] for i in range(n)])


class SmallBC(nn.Module):
    # registered buffers (normalisation stats saved with the checkpoint)
    y_mean: torch.Tensor
    y_std: torch.Tensor
    p_mean: torch.Tensor
    p_std: torch.Tensor

    def __init__(
        self,
        image_size: int = 96,
        width: int = 32,
        num_kp: int = 32,
        use_proprio: bool = True,
        proprio_dim: int = 19,
        chunk: int = 16,
        target_dim: int = 5,
        hidden: int = 512,
        shift_pad: int = 4,
    ):
        super().__init__()
        self.cfg = dict(
            image_size=image_size,
            width=width,
            num_kp=num_kp,
            use_proprio=use_proprio,
            proprio_dim=proprio_dim,
            chunk=chunk,
            target_dim=target_dim,
            hidden=hidden,
            shift_pad=shift_pad,
        )
        self.enc_front = Encoder(image_size, width, num_kp)
        self.enc_wrist = Encoder(image_size, width, num_kp)
        in_dim = (
            self.enc_front.out_dim + self.enc_wrist.out_dim + (proprio_dim if use_proprio else 0)
        )
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, chunk * target_dim),
        )
        self.register_buffer("y_mean", torch.zeros(chunk, target_dim))
        self.register_buffer("y_std", torch.ones(chunk, target_dim))
        self.register_buffer("p_mean", torch.zeros(proprio_dim))
        self.register_buffer("p_std", torch.ones(proprio_dim))

    def set_stats(self, y_mean, y_std, p_mean, p_std) -> None:
        for name, v in (("y_mean", y_mean), ("y_std", y_std), ("p_mean", p_mean), ("p_std", p_std)):
            getattr(self, name).copy_(torch.as_tensor(v, dtype=torch.float32))

    def _img(self, x: torch.Tensor) -> torch.Tensor:
        """uint8 (B, H, W, 3) -> float (B, 3, H, W) in [-1, 1], random-shifted in training."""
        x = x.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
        if self.training:
            x = random_shift(x, self.cfg["shift_pad"])
        return x

    def forward(
        self, front: torch.Tensor, wrist: torch.Tensor, proprio: torch.Tensor
    ) -> torch.Tensor:
        """Normalised (B, K, D) chunk. ``proprio`` is raw (un-normalised)."""
        feats = [self.enc_front(self._img(front)), self.enc_wrist(self._img(wrist))]
        if self.cfg["use_proprio"]:
            feats.append((proprio - self.p_mean) / self.p_std)
        out = self.head(torch.cat(feats, dim=1))
        return out.reshape(-1, self.cfg["chunk"], self.cfg["target_dim"])

    def loss(self, front, wrist, proprio, y_raw) -> torch.Tensor:
        """L1 in normalised target space."""
        y = (y_raw - self.y_mean) / self.y_std
        return (self(front, wrist, proprio) - y).abs().mean()

    @torch.no_grad()
    def predict(self, front, wrist, proprio) -> torch.Tensor:
        """Raw (B, K, D) targets."""
        return self(front, wrist, proprio) * self.y_std + self.y_mean


def load_checkpoint(path, map_location="cpu") -> tuple[SmallBC, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model = SmallBC(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
