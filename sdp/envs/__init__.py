"""Gymnasium environments. Importing this module registers them."""

import gymnasium as gym

from sdp.envs.panda_pick_place import POSE_DISTRIBUTIONS, PandaPickPlaceEnv

ENV_IDS = ("PandaPickPlace-v0",)

if "PandaPickPlace-v0" not in gym.registry:
    gym.register(id="PandaPickPlace-v0", entry_point="sdp.envs.panda_pick_place:PandaPickPlaceEnv")

__all__ = ["ENV_IDS", "POSE_DISTRIBUTIONS", "PandaPickPlaceEnv"]
