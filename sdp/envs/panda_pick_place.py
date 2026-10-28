"""PandaPickPlace-v0: pick a 4 cm cube and place it in a 6 cm target square.

Success (all must hold continuously for ``settle_steps`` control steps = 0.5 s):

* the cube centre is inside the 6 cm target square (|dx|, |dy| <= 3 cm);
* the cube rests on the table (centre height <= half-size + 5 mm);
* the gripper is open (finger gap >= 5 cm, i.e. the cube has been released);
* the cube is still (linear speed < 1 cm/s).

The cube pose (x, y, yaw) is sampled per episode from a named distribution;
``nominal`` is used for demonstrations and the main eval, ``shifted`` is the same
box translated by (+8 cm, -8 cm) for the distribution-shift eval.
"""

from __future__ import annotations

import mujoco
import numpy as np

from sdp.envs.base_mujoco_env import MujocoArmEnv

POSE_DISTRIBUTIONS: dict[str, dict[str, tuple[float, float]]] = {
    "nominal": {"x": (0.45, 0.60), "y": (-0.18, -0.03), "yaw": (-np.pi / 4, np.pi / 4)},
    "shifted": {"x": (0.53, 0.68), "y": (-0.26, -0.11), "yaw": (-np.pi / 4, np.pi / 4)},
}


class PandaPickPlaceEnv(MujocoArmEnv):
    cube_half: float = 0.02  # 4 cm cube
    cube_mass: float = 0.05
    target_center = np.array([0.55, 0.15])
    target_half: float = 0.03  # 6 cm square
    settle_steps: int = 10  # 0.5 s at 20 Hz

    reach_dist: float = 0.03  # TCP within 3 cm of the cube centre counts as "reached"
    lift_height: float = 0.02  # cube raised 2 cm above its resting height counts as "lifted"
    open_gap: float = 0.05  # finger gap above which the gripper counts as open
    still_speed: float = 0.01
    finger_press_force: float = 20.0  # N; finger-table contact above this is a collision

    task_description = "Pick up the red cube and place it in the green square."

    def __init__(self, distribution: str = "nominal", **kwargs):
        if distribution not in POSE_DISTRIBUTIONS:
            raise ValueError(f"unknown distribution {distribution!r}")
        self.distribution = distribution
        super().__init__(**kwargs)
        m = self.model
        self._cube_bid = m.body("cube").id
        self._cube_gid = m.geom("cube").id
        jnt = m.joint("cube_free")
        self._cube_qadr = jnt.qposadr[0]
        self._cube_dadr = jnt.dofadr[0]
        self._episode: dict = {}
        self._contact_force = np.zeros(6)

    # ------------------------------------------------------------------ scene
    def _add_task_bodies(self, spec: mujoco.MjSpec) -> None:
        cube = spec.worldbody.add_body(name="cube", pos=[0.5, -0.1, self.cube_half])
        cube.add_freejoint(name="cube_free")
        cube.add_geom(
            name="cube",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.cube_half] * 3,
            mass=self.cube_mass,
            rgba=[0.85, 0.12, 0.10, 1.0],
            friction=[1.0, 0.01, 0.0005],
            condim=4,
        )
        cx, cy = self.target_center
        spec.worldbody.add_geom(
            name="target",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.target_half, self.target_half, 0.0005],
            pos=[cx, cy, 0.0005],
            rgba=[0.10, 0.75, 0.25, 1.0],
            contype=0,
            conaffinity=0,
        )

    def _sample_task(self, rng: np.random.Generator, options: dict) -> None:
        dist = POSE_DISTRIBUTIONS[options.get("distribution", self.distribution)]
        x = rng.uniform(*dist["x"])
        y = rng.uniform(*dist["y"])
        yaw = rng.uniform(*dist["yaw"])
        q = self.data.qpos
        q[self._cube_qadr : self._cube_qadr + 3] = [x, y, self.cube_half]
        q[self._cube_qadr + 3 : self._cube_qadr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.data.qvel[self._cube_dadr : self._cube_dadr + 6] = 0.0
        self._initial_cube = np.array([x, y, yaw])

    # ------------------------------------------------------------ bookkeeping
    def _on_reset(self) -> None:
        self._settled = 0
        self._carried = False
        self._episode = {
            "initial_cube_xy_yaw": self._initial_cube.tolist(),
            "ever_reached": False,
            "ever_grasped": False,
            "ever_lifted": False,
            "dropped": False,
            "collision": False,
            "steps": 0,
        }
        self._last_grip_cmd = 1.0

    def step(self, action):
        self._last_grip_cmd = float(np.clip(np.asarray(action, dtype=np.float64)[6], -1.0, 1.0))
        return super().step(action)

    def _on_substep(self) -> None:
        if self._episode["collision"]:
            return
        m, d = self.model, self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if self._table_gid not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self._table_gid else c.geom1
            body = m.geom_bodyid[other]
            if body not in self._robot_body_ids or body == m.body("link0").id:
                continue
            if body in self._finger_body_ids:
                mujoco.mj_contactForce(m, d, i, self._contact_force)
                if self._contact_force[0] < self.finger_press_force:
                    continue
            self._episode["collision"] = True
            return

    # ------------------------------------------------------------------- state
    def cube_pose(self) -> tuple[np.ndarray, float]:
        """Cube centre position and yaw (rad) about world z."""
        q = self.data.qpos[self._cube_qadr : self._cube_qadr + 7]
        w, x, y, z = q[3:7]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return q[:3].copy(), float(yaw)

    def tcp_yaw(self, target: bool = False) -> float:
        """Yaw of the finger closing axis about world z (0 = fingers close along world x).

        ``target=True`` gives the yaw of the integrated target instead of the measured TCP.
        """
        if target:
            mat = np.empty(9)
            mujoco.mju_quat2Mat(mat, self._target_quat)
            mat = mat.reshape(3, 3)
        else:
            mat = self.data.site_xmat[self._tcp_sid].reshape(3, 3)
        closing_axis = mat[:, 1]  # TCP y = finger closing direction
        return float(np.arctan2(closing_axis[1], closing_axis[0]))

    def _fingers_touch_cube(self) -> bool:
        m, d = self.model, self.data
        touching = set()
        for i in range(d.ncon):
            c = d.contact[i]
            if self._cube_gid in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self._cube_gid else c.geom1
                touching.add(m.geom_bodyid[other])
        return all(b in touching for b in self._finger_body_ids)

    def _task_info(self) -> dict:
        cube_pos, cube_yaw = self.cube_pose()
        tcp_pos, _ = self.tcp_pose()
        cube_vel = self.data.qvel[self._cube_dadr : self._cube_dadr + 3]
        gap = self.gripper_opening()

        offset = cube_pos[:2] - self.target_center
        in_region = bool(np.all(np.abs(offset) <= self.target_half))
        on_table = bool(cube_pos[2] <= self.cube_half + 0.005)
        gripper_open = gap >= self.open_gap
        still = float(np.linalg.norm(cube_vel)) < self.still_speed
        grasped = self._fingers_touch_cube()
        lifted = cube_pos[2] >= self.cube_half + self.lift_height
        tcp_cube_dist = float(np.linalg.norm(tcp_pos - cube_pos))

        ep = self._episode
        if self._step_count > 0:
            ep["steps"] = self._step_count
            ep["ever_reached"] |= tcp_cube_dist <= self.reach_dist
            ep["ever_grasped"] |= grasped
            ep["ever_lifted"] |= bool(lifted and grasped)
            # A drop: the cube was carried, then landed on the table out of the fingers
            # while the policy was still commanding the gripper closed. Judging at the
            # landing (not at the first lost contact) ignores one-step contact flicker.
            if lifted and grasped:
                self._carried = True
            elif self._carried and on_table:
                if not grasped and self._last_grip_cmd < 0.0:
                    ep["dropped"] = True
                self._carried = False

        placed = in_region and on_table and gripper_open and still
        self._settled = self._settled + 1 if placed else 0
        success = self._settled >= self.settle_steps

        return {
            "success": success,
            "cube_pos": cube_pos,
            "cube_yaw": cube_yaw,
            "tcp_pos": tcp_pos,
            "tcp_yaw": self.tcp_yaw(),
            "gripper_gap": gap,
            "tcp_cube_dist": tcp_cube_dist,
            "cube_target_offset": offset,
            "in_region": in_region,
            "on_table": on_table,
            "grasped": grasped,
            "is_success_candidate": placed,
            "episode": dict(ep),
        }
