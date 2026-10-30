"""Load an evaluation policy from a checkpoint path, ACT or smallbc."""

from __future__ import annotations

from pathlib import Path


def load_policy(checkpoint: str | Path, n_action_steps: int | None = None):
    """(policy, policy_type). A ``.pt`` file is smallbc; a directory is a LeRobot ACT run.

    smallbc runs with 1 torch thread, as in ``experiments/smallbc/eval.py``, so episodes
    reproduce exactly.
    """
    from sdp.policies.smallbc import is_smallbc_checkpoint

    if is_smallbc_checkpoint(checkpoint):
        from sdp.policies.smallbc.policy import SmallBCPolicy

        return SmallBCPolicy(checkpoint, n_exec=n_action_steps or 8, threads=1), "smallbc"
    from sdp.eval.act_policy import ACTPolicyAdapter

    return ACTPolicyAdapter(checkpoint, n_action_steps=n_action_steps), "act"
