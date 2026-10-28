"""Adapter so the scripted expert plugs into the eval harness and the recorder."""

from __future__ import annotations

import numpy as np

from sdp.envs.panda_pick_place import PandaPickPlaceEnv
from sdp.expert.waypoint import WaypointExpert


class ExpertPolicy:
    def __init__(self, expert: WaypointExpert | None = None):
        self.expert = expert or WaypointExpert()

    def reset(self) -> None:
        self.expert.reset()

    def act(self, obs: dict, env: PandaPickPlaceEnv) -> np.ndarray:
        return self.expert.act(env)
