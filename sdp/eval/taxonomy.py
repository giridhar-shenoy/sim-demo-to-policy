"""Failure taxonomy for PandaPickPlace-v0 episodes.

``label_episode`` maps the per-episode flags the env accumulates (``info["episode"]``)
plus the final step's info to exactly one label. Checks run in this order, so each
failed episode gets the earliest cause in the list:

``success``        the env's success predicate held for 0.5 s (episode terminated).
``collision``      an arm link or the hand touched the table, or the fingers pressed
                   into it with more than 20 N.
``never_reached``  the TCP never came within 3 cm of the cube centre.
``grasp_missed``   reached the cube but never lifted it 2 cm while both fingers touched it.
``dropped``        carried the cube, then it landed out of the fingers while the gripper
                   was still commanded closed.
``placed_outside`` the cube ended on the table, released, outside the target square.
``timeout``        anything else at the step limit, e.g. still holding the cube, or the
                   cube in the square but the gripper never opened.
"""

from __future__ import annotations

LABELS = (
    "success",
    "collision",
    "never_reached",
    "grasp_missed",
    "dropped",
    "placed_outside",
    "timeout",
)
FAILURE_LABELS = LABELS[1:]


def label_episode(episode: dict, final: dict) -> str:
    """``episode``: ``info["episode"]`` flags; ``final``: the last step's info dict."""
    if final.get("success"):
        return "success"
    if episode.get("collision"):
        return "collision"
    if not episode.get("ever_reached"):
        return "never_reached"
    if not episode.get("ever_lifted"):
        return "grasp_missed"
    if episode.get("dropped"):
        return "dropped"
    if final.get("on_table") and not final.get("grasped") and not final.get("in_region"):
        return "placed_outside"
    return "timeout"
