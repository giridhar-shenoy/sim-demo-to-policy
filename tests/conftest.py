import os
from pathlib import Path

import pytest

# Use the shared checkout on the laptop when nothing else is configured; CI sets
# SDP_MENAGERIE_DIR in the image.
_LAPTOP_MENAGERIE = Path("~/work/menagerie").expanduser()
if "SDP_MENAGERIE_DIR" not in os.environ and _LAPTOP_MENAGERIE.is_dir():
    os.environ["SDP_MENAGERIE_DIR"] = str(_LAPTOP_MENAGERIE)
os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.fixture
def make_env():
    from sdp.envs import PandaPickPlaceEnv

    envs = []

    def _make(**kwargs):
        env = PandaPickPlaceEnv(**kwargs)
        envs.append(env)
        return env

    yield _make
    for env in envs:
        env.close()
