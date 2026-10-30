"""smallbc: a small from-scratch visuomotor BC policy with absolute-target actions.

Ported from ``experiments/smallbc/`` (the overnight experiment of 30 Oct 2026, which is
kept unchanged as the record): a per-camera CNN with a spatial-softmax keypoint head,
an MLP over [keypoints, proprio], and a 16-step chunk of absolute TCP targets.

* ``python -m sdp.policies.smallbc.record``: scripted demos into uint8 memmap shards;
* ``python -m sdp.policies.smallbc.train``: L1 training with random-shift augmentation;
* ``sdp eval --checkpoint <ckpt.pt>``: the harness eval, dispatched by checkpoint type.
"""

from pathlib import Path

CHECKPOINT_SUFFIX = ".pt"


def is_smallbc_checkpoint(path: str | Path) -> bool:
    """smallbc checkpoints are single ``.pt`` files; ACT checkpoints are directories."""
    p = Path(path)
    return p.is_file() and p.suffix == CHECKPOINT_SUFFIX
