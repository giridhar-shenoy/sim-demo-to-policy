import pytest

from sdp.eval.taxonomy import FAILURE_LABELS, LABELS, label_episode

BASE = {
    "ever_reached": True,
    "ever_grasped": True,
    "ever_lifted": True,
    "dropped": False,
    "collision": False,
}
FINAL = {"success": False, "on_table": True, "grasped": False, "in_region": False}


def ep(**kw):
    return {**BASE, **kw}


def final(**kw):
    return {**FINAL, **kw}


def test_label_set_matches_spec():
    assert set(FAILURE_LABELS) == {
        "never_reached",
        "grasp_missed",
        "dropped",
        "placed_outside",
        "collision",
        "timeout",
    }
    assert LABELS[0] == "success"


@pytest.mark.parametrize(
    ("episode", "last", "expected"),
    [
        (ep(), final(success=True), "success"),
        (ep(collision=True), final(success=True), "success"),
        (ep(collision=True, ever_reached=False), final(), "collision"),
        (ep(ever_reached=False, ever_grasped=False, ever_lifted=False), final(), "never_reached"),
        (ep(ever_lifted=False), final(), "grasp_missed"),
        (ep(dropped=True), final(), "dropped"),
        (ep(), final(), "placed_outside"),
        (ep(), final(grasped=True, on_table=False), "timeout"),
        (ep(), final(in_region=True), "timeout"),
    ],
)
def test_label_priority(episode, last, expected):
    assert label_episode(episode, last) == expected
