"""Scripted waypoint expert for PandaPickPlace-v0.

The expert reads privileged state from the env (cube pose, TCP pose, and the
integrated TCP target that actions move) and steers the target through fixed
waypoints with a saturated proportional law, producing actions in the same
normalised 7-D space a learned policy uses. Steering the target rather than the
measured TCP avoids overshoot from the arm's tracking lag; phase transitions still
wait for the measured TCP to arrive. It is the source of the
"scripted demos" dataset and a sanity check for the env, recorder and eval harness.

Phases: above cube -> descend -> close -> lift -> above target -> lower -> open -> retreat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sdp.envs.panda_pick_place import PandaPickPlaceEnv

PHASES = ("above_cube", "descend", "close", "lift", "above_target", "lower", "open", "retreat")


def wrap_quarter_turn(angle: float) -> float:
    """Wrap an angle to [-pi/4, pi/4): a cube looks the same every quarter turn."""
    return (angle + np.pi / 4) % (np.pi / 2) - np.pi / 4


@dataclass
class WaypointExpert:
    hover_z: float = 0.12  # TCP height for transit moves
    grasp_z: float = 0.018  # TCP height at the grasp (cube centre is at 0.02)
    place_z: float = 0.03  # TCP height when releasing over the target
    gain: float = 1.0  # fraction of the remaining target error commanded per step
    speed: float = 0.6  # fraction of the env's max per-step delta
    pos_tol: float = 0.008
    yaw_tol: float = 0.04
    close_steps: int = 8
    open_steps: int = 6

    phase: int = field(default=0, init=False)
    _timer: int = field(default=0, init=False)

    def reset(self) -> None:
        self.phase = 0
        self._timer = 0

    @property
    def phase_name(self) -> str:
        return PHASES[self.phase]

    def act(self, env: PandaPickPlaceEnv) -> np.ndarray:
        cube_pos, cube_yaw = env.cube_pose()
        tcp_pos, _ = env.tcp_pose()
        tgt_pos, _ = env.target_pose()
        tgt_yaw = env.tcp_yaw(target=True)
        tx, ty = env.target_center

        grip = 1.0
        yaw_goal = tgt_yaw
        name = self.phase_name
        if name == "above_cube":
            goal = np.array([cube_pos[0], cube_pos[1], self.hover_z])
            yaw_goal = tgt_yaw + wrap_quarter_turn(cube_yaw - tgt_yaw)
        elif name == "descend":
            goal = np.array([cube_pos[0], cube_pos[1], self.grasp_z])
            yaw_goal = tgt_yaw + wrap_quarter_turn(cube_yaw - tgt_yaw)
        elif name == "close":
            goal, grip = tgt_pos, -1.0
        elif name == "lift":
            goal, grip = np.array([tgt_pos[0], tgt_pos[1], self.hover_z]), -1.0
        elif name == "above_target":
            goal, grip = np.array([tx, ty, self.hover_z]), -1.0
        elif name == "lower":
            goal, grip = np.array([tx, ty, self.place_z]), -1.0
        elif name == "open":
            goal = tgt_pos
        else:  # retreat
            goal = np.array([tgt_pos[0], tgt_pos[1], self.hover_z])

        err = goal - tgt_pos
        yaw_err = yaw_goal - tgt_yaw
        self._advance(name, goal - tcp_pos, yaw_goal - env.tcp_yaw(), tcp_pos)

        action = np.zeros(7, dtype=np.float32)
        limit = self.speed
        action[:3] = np.clip(self.gain * err / env.max_dpos, -limit, limit)
        action[5] = np.clip(self.gain * yaw_err / env.max_drot, -limit, limit)
        action[6] = grip
        return action

    def _advance(self, name: str, err: np.ndarray, yaw_err: float, tcp_pos: np.ndarray) -> None:
        reached = np.linalg.norm(err) < self.pos_tol
        if name == "above_cube":
            # xy alignment is what matters before descending; height can still settle.
            if np.linalg.norm(err[:2]) < self.pos_tol and abs(yaw_err) < self.yaw_tol:
                self._next()
        elif name in ("descend", "above_target", "lower"):
            if reached:
                self._next()
        elif name == "close":
            self._timer += 1
            if self._timer >= self.close_steps:
                self._next()
        elif name == "lift":
            if tcp_pos[2] > self.hover_z - 0.02:
                self._next()
        elif name == "open":
            self._timer += 1
            if self._timer >= self.open_steps:
                self._next()

    def _next(self) -> None:
        self.phase = min(self.phase + 1, len(PHASES) - 1)
        self._timer = 0
