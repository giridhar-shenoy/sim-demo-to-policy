"""Shared helpers for the small-CNN BC experiment (smallbc): a thin re-export.

The code lives in ``sdp/policies/smallbc/`` (one implementation, the one ``sdp eval``
and ``scripts/eval_all.sh`` run); the experiment's original copy is in git history
(tag v0.1.0). Kept here: the experiment's environment setup (MUJOCO_GL=egl and
SDP_MENAGERIE_DIR default to the values the repo's CHECKPOINTS.md uses, unless they are
already set) and its seed ranges.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
_REPO = Path(__file__).resolve().parents[2]
if "SDP_MENAGERIE_DIR" not in os.environ and not (_REPO / "third_party" / "mujoco_menagerie").is_dir():
    os.environ["SDP_MENAGERIE_DIR"] = str(Path("~/work/menagerie").expanduser())

from sdp.policies.smallbc.common import *  # noqa: E402,F401,F403

HERE = Path(__file__).resolve().parent
REPO = _REPO
# Eval seeds already spent on the ACT runs' dev/test evals. Never evaluate on them here.
USED_TEST_SEEDS = range(10000, 10500)
DEV_SEEDS = range(10500, 10550)
