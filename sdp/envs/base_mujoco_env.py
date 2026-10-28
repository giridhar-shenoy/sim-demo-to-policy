"""Base class for the MuJoCo arm environments in this repo.

The base class owns everything that is not task-specific:

* loading the Menagerie arm with ``mujoco.MjSpec`` and adding a table, lights,
  a TCP site and two cameras (``front`` fixed, ``wrist`` on the hand);
* the action interface: a 6-DoF delta TCP pose plus an absolute gripper command,
  integrated into a target pose and turned into joint position targets by damped
  least-squares IK, which drive the arm's own position actuators;
* the timing: 500 Hz physics, 20 Hz control (25 physics substeps per ``step``);
* offscreen rendering of the two 96x96 observation cameras.

Subclasses add objects (``_add_task_bodies``), sample the initial object pose
(``_sample_task``), and define ``_task_info``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

MENAGERIE_ENV = "SDP_MENAGERIE_DIR"
# Observation cameras draw world geoms (0), task objects (1) and the arm's collision
# meshes (3); the GIF view (``render``) draws the visual meshes (2) instead.
OBS_GEOM_GROUPS = (0, 1, 3)
_DEFAULT_MENAGERIE = Path(__file__).resolve().parents[2] / "third_party" / "mujoco_menagerie"


def menagerie_dir() -> Path:
    """Locate the mujoco_menagerie checkout (see scripts/fetch_menagerie.sh)."""
    root = Path(os.environ.get(MENAGERIE_ENV, _DEFAULT_MENAGERIE)).expanduser()
    if not (root / "franka_emika_panda" / "panda.xml").is_file():
        raise FileNotFoundError(
            f"mujoco_menagerie not found at {root}. Run scripts/fetch_menagerie.sh "
            f"or set {MENAGERIE_ENV} to an existing checkout."
        )
    return root


@dataclass(frozen=True)
class ArmConfig:
    """What the base env needs to know about a Menagerie arm."""

    model_relpath: str
    arm_joints: tuple[str, ...]
    arm_actuators: tuple[str, ...]
    gripper_actuator: str
    gripper_ctrl_range: tuple[float, float]  # (closed, open)
    finger_joints: tuple[str, ...]
    finger_bodies: tuple[str, ...]
    hand_body: str
    tcp_offset: tuple[float, float, float]  # TCP site position in the hand frame
    home_keyframe: str
    robot_bodies: tuple[str, ...]


PANDA = ArmConfig(
    model_relpath="franka_emika_panda/panda.xml",
    arm_joints=tuple(f"joint{i}" for i in range(1, 8)),
    arm_actuators=tuple(f"actuator{i}" for i in range(1, 8)),
    gripper_actuator="actuator8",
    gripper_ctrl_range=(0.0, 255.0),
    finger_joints=("finger_joint1", "finger_joint2"),
    finger_bodies=("left_finger", "right_finger"),
    hand_body="hand",
    # Midpoint between the fingertip pads: finger origin (0.0584) + pad centre (0.0445).
    tcp_offset=(0.0, 0.0, 0.103),
    home_keyframe="home",
    robot_bodies=tuple(f"link{i}" for i in range(8)) + ("hand", "left_finger", "right_finger"),
)


def lookat_quat(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Quaternion for a MuJoCo camera at ``pos`` looking at ``target`` (camera looks along -z)."""
    pos, target, up = (np.asarray(v, dtype=float) for v in (pos, target, up))
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    mat = np.column_stack([right, cam_up, -fwd])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat


def quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    if angle > 1e-12:
        mujoco.mju_axisAngle2Quat(quat, rotvec / angle, angle)
    return quat


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def quat_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector that rotates ``current`` onto ``target``."""
    neg = np.empty(4)
    mujoco.mju_negQuat(neg, current)
    err = quat_mul(target, neg)
    if err[0] < 0.0:  # shortest path
        err = -err
    vel = np.empty(3)
    mujoco.mju_quat2Vel(vel, err, 1.0)
    return vel


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=float).reshape(9))
    return quat


def _canonical_quat(quat: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Pick the sign of ``quat`` closest to ``ref``.

    The top-down grasp orientation has w close to 0, so the usual "w >= 0" rule would
    flip the sign of the observed quaternion back and forth; comparing against the
    home orientation does not.
    """
    return quat if float(np.dot(quat, ref)) >= 0.0 else -quat


class MultiCameraRenderer:
    """Renders several fixed cameras side by side into one offscreen buffer.

    One ``mjr_readPixels`` per step instead of one per camera. ``geom_groups`` selects
    which geom groups are drawn; the observation cameras draw the arm's collision
    meshes (group 3, ~3.3k vertices) instead of its visual meshes (group 2, ~400k
    vertices): at 96 px they look the same and render about 5x faster.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_names: tuple[str, ...],
        size: int,
        geom_groups: tuple[int, ...] = (0, 1, 2),
    ):
        from mujoco import gl_context

        self._model = model
        self._size = size
        width = size * len(camera_names)
        self._ctx = gl_context.GLContext(width, size)
        self._ctx.make_current()
        self._con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._con)
        self._scene = mujoco.MjvScene(model, maxgeom=1000)
        self._opt = mujoco.MjvOption()
        self._opt.geomgroup[:] = [int(g in geom_groups) for g in range(len(self._opt.geomgroup))]
        self._pert = mujoco.MjvPerturb()
        self._names = camera_names
        self._cams = []
        for name in camera_names:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = model.camera(name).id
            self._cams.append(cam)
        self._buf = np.empty((size, width, 3), dtype=np.uint8)
        self._full = mujoco.MjrRect(0, 0, width, size)

    def render(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        self._ctx.make_current()
        s = self._size
        for i, cam in enumerate(self._cams):
            mujoco.mjv_updateScene(
                self._model,
                data,
                self._opt,
                self._pert,
                cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self._scene,
            )
            mujoco.mjr_render(mujoco.MjrRect(i * s, 0, s, s), self._scene, self._con)
        mujoco.mjr_readPixels(self._buf, None, self._full, self._con)
        img = np.flipud(self._buf)
        return {name: img[:, i * s : (i + 1) * s].copy() for i, name in enumerate(self._names)}

    def close(self) -> None:
        if self._con is not None:
            self._ctx.make_current()
            self._con.free()
            self._ctx.free()
            self._con = None


class MujocoArmEnv(gym.Env):
    """Delta-TCP-pose arm env with two 96x96 cameras. See the module docstring."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    arm: ArmConfig = PANDA
    physics_hz: int = 500
    control_hz: int = 20
    max_episode_steps: int = 250
    camera_names: tuple[str, ...] = ("front", "wrist")

    # Action scaling: a in [-1, 1]^7 -> per-control-step deltas.
    max_dpos: float = 0.02  # m per step (0.4 m/s at 20 Hz)
    max_drot: float = 0.10  # rad per step (2 rad/s at 20 Hz)

    # The TCP target is clamped to this box (world frame, metres) and kept within
    # max_tilt of pointing straight down, with yaw within max_yaw of the home yaw.
    workspace_lo = np.array([0.25, -0.40, 0.005])
    workspace_hi = np.array([0.80, 0.40, 0.60])
    max_tilt: float = 0.5
    max_yaw: float = np.pi / 2

    # Damped least-squares IK.
    ik_iters: int = 4
    ik_damping: float = 0.05
    ik_nullspace_gain: float = 0.05

    def __init__(
        self,
        render_mode: str | None = None,
        image_size: int = 96,
        render_obs: bool = True,
        render_size: tuple[int, int] = (240, 320),
    ):
        self.render_mode = render_mode
        self.image_size = image_size
        self.render_obs = render_obs
        self.render_size = render_size

        self.spec = self._build_spec()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)

        if abs(self.model.opt.timestep - 1.0 / self.physics_hz) > 1e-9:
            raise ValueError(f"model timestep {self.model.opt.timestep} != 1/{self.physics_hz}")
        self.n_substeps = self.physics_hz // self.control_hz

        m = self.model
        self._arm_qadr = np.array([m.joint(j).qposadr[0] for j in self.arm.arm_joints])
        self._arm_dadr = np.array([m.joint(j).dofadr[0] for j in self.arm.arm_joints])
        self._arm_act = np.array([m.actuator(a).id for a in self.arm.arm_actuators])
        self._grip_act = m.actuator(self.arm.gripper_actuator).id
        self._finger_qadr = np.array([m.joint(j).qposadr[0] for j in self.arm.finger_joints])
        self._arm_range = np.array([m.joint(j).range for j in self.arm.arm_joints])
        robot_geoms = np.isin(m.geom_bodyid, [m.body(b).id for b in self.arm.robot_bodies])
        m.geom_rgba[robot_geoms & (m.geom_group == 3)] = [0.85, 0.85, 0.85, 1.0]
        self._tcp_sid = m.site("tcp").id
        self._home_key = m.keyframe(self.arm.home_keyframe).id
        self._q_home = m.key_qpos[self._home_key][self._arm_qadr].copy()
        self._robot_body_ids = {m.body(b).id for b in self.arm.robot_bodies}
        self._finger_body_ids = tuple(m.body(b).id for b in self.arm.finger_bodies)
        self._table_gid = m.geom("table").id

        # Reference TCP orientation at home: used for the observation's quaternion sign
        # and for the tilt / yaw limits.
        mujoco.mj_resetDataKeyframe(m, self._ik_data, self._home_key)
        mujoco.mj_kinematics(m, self._ik_data)
        self._home_quat = mat_to_quat(self._ik_data.site_xmat[self._tcp_sid])

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        img = gym.spaces.Box(0, 255, shape=(image_size, image_size, 3), dtype=np.uint8)
        n_state = 7 + 1 + len(self.arm.arm_joints)
        self.observation_space = gym.spaces.Dict(
            {
                "front": img,
                "wrist": img,
                "state": gym.spaces.Box(-np.inf, np.inf, shape=(n_state,), dtype=np.float32),
            }
        )

        self._obs_renderer: MultiCameraRenderer | None = None
        self._view_renderer: mujoco.Renderer | None = None
        self._q_cmd = self._q_home.copy()
        self._target_pos = np.zeros(3)
        self._target_quat = self._home_quat.copy()
        self._step_count = 0

    # ------------------------------------------------------------------ scene
    def _build_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(menagerie_dir() / self.arm.model_relpath))
        spec.option.timestep = 1.0 / self.physics_hz
        # Elliptic friction cones with a high impratio stop grasped objects from creeping.
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10.0

        # The real Panda's controller compensates gravity; do the same so the
        # position actuators do not sag under load.
        for name in self.arm.robot_bodies:
            spec.body(name).gravcomp = 1.0

        hand = spec.body(self.arm.hand_body)
        hand.add_site(name="tcp", pos=list(self.arm.tcp_offset), size=[0.004] * 3, group=4)
        # Wrist camera on the side of the hand, looking past the fingertips.
        cam_pos = np.array([0.065, 0.0, 0.035])
        hand.add_camera(
            name="wrist",
            pos=cam_pos.tolist(),
            quat=lookat_quat(cam_pos, [0.0, 0.0, 0.17], up=[1.0, 0.0, 0.0]).tolist(),
            fovy=75.0,
        )

        # Flat, shadow-free lighting: the 4096 px shadow map tripled the render time and
        # the arm's shadow covered most of a 96 px image.
        spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
        spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
        spec.visual.quality.shadowsize = 0
        spec.visual.quality.offsamples = 0
        spec.visual.global_.offwidth = 640
        spec.visual.global_.offheight = 480

        tex = spec.add_texture(
            name="table_tex",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.62, 0.55, 0.46],
            rgb2=[0.58, 0.51, 0.42],
            width=256,
            height=256,
        )
        mat = spec.add_material(name="table_mat", texrepeat=[8, 8], reflectance=0.0)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name

        world = spec.worldbody
        world.add_geom(
            name="table",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[1.5, 1.5, 0.05],
            material="table_mat",
        )
        world.add_light(
            name="key",
            pos=[0.4, -0.3, 1.5],
            dir=[0.0, 0.2, -1.0],
            diffuse=[0.4] * 3,
            castshadow=False,
        )
        # Front camera: facing the robot from across the table, framing the cube's
        # spawn boxes and the target square.
        front_pos = [1.05, -0.02, 0.62]
        world.add_camera(
            name="front", pos=front_pos, quat=lookat_quat(front_pos, [0.56, -0.04, 0.0]), fovy=45.0
        )
        # The base link sits on the table; its contact is not a collision.
        spec.add_exclude(bodyname1="link0", bodyname2="world")

        self._add_task_bodies(spec)
        return spec

    def _add_task_bodies(self, spec: mujoco.MjSpec) -> None:
        """Subclasses add objects and markers here."""

    # -------------------------------------------------------------- task hooks
    def _sample_task(self, rng: np.random.Generator, options: dict) -> None:
        """Set object poses in ``self.data`` for a new episode."""

    def _task_info(self) -> dict:
        return {"success": False}

    # ------------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, self._home_key)
        self._sample_task(self.np_random, options)
        mujoco.mj_forward(m, d)

        self._q_cmd = self._q_home.copy()
        d.ctrl[self._arm_act] = self._q_cmd
        d.ctrl[self._grip_act] = self.arm.gripper_ctrl_range[1]
        # Let contacts settle for 0.1 s before the first observation.
        for _ in range(self.physics_hz // 10):
            mujoco.mj_step(m, d)

        self._target_pos = d.site_xpos[self._tcp_sid].copy()
        self._target_quat = mat_to_quat(d.site_xmat[self._tcp_sid])
        self._step_count = 0
        self._on_reset()
        return self._get_obs(), self._task_info()

    def _on_reset(self) -> None:
        """Subclasses reset per-episode bookkeeping here."""

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._integrate_target(a[:3] * self.max_dpos, a[3:6] * self.max_drot)
        self._q_cmd = self._solve_ik(self._target_pos, self._target_quat)

        d = self.data
        d.ctrl[self._arm_act] = self._q_cmd
        lo, hi = self.arm.gripper_ctrl_range
        d.ctrl[self._grip_act] = lo + (a[6] + 1.0) * 0.5 * (hi - lo)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, d)
            self._on_substep()
        self._step_count += 1

        info = self._task_info()
        terminated = bool(info["success"])
        truncated = self._step_count >= self.max_episode_steps and not terminated
        return self._get_obs(), float(terminated), terminated, truncated, info

    def _on_substep(self) -> None:
        """Subclasses can accumulate per-physics-step flags (e.g. collisions) here."""

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._view_renderer is None:
            h, w = self.render_size
            self._view_renderer = mujoco.Renderer(self.model, h, w)
        self._view_renderer.update_scene(self.data, camera="front")
        return self._view_renderer.render().copy()

    def close(self):
        for r in (self._obs_renderer, self._view_renderer):
            if r is not None:
                r.close()
        self._obs_renderer = self._view_renderer = None

    # ------------------------------------------------------------ observations
    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Current TCP position and canonical-sign quaternion (w, x, y, z)."""
        pos = self.data.site_xpos[self._tcp_sid].copy()
        quat = _canonical_quat(mat_to_quat(self.data.site_xmat[self._tcp_sid]), self._home_quat)
        return pos, quat

    def target_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """The integrated TCP target that actions move (position, quaternion w-x-y-z)."""
        return self._target_pos.copy(), self._target_quat.copy()

    def gripper_opening(self) -> float:
        """Distance between the fingers in metres (0 closed, 0.08 fully open)."""
        return float(self.data.qpos[self._finger_qadr].sum())

    def _get_state(self) -> np.ndarray:
        pos, quat = self.tcp_pose()
        q = self.data.qpos[self._arm_qadr]
        return np.concatenate([pos, quat, [self.gripper_opening()], q]).astype(np.float32)

    def _render_cameras(self) -> dict[str, np.ndarray]:
        if not self.render_obs:
            blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            return {name: blank.copy() for name in self.camera_names}
        if self._obs_renderer is None:
            self._obs_renderer = MultiCameraRenderer(
                self.model, self.camera_names, self.image_size, geom_groups=OBS_GEOM_GROUPS
            )
        return self._obs_renderer.render(self.data)

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs = self._render_cameras()
        obs["state"] = self._get_state()
        return obs

    # ------------------------------------------------------------- controller
    def _integrate_target(self, dpos: np.ndarray, drot: np.ndarray) -> None:
        self._target_pos = np.clip(self._target_pos + dpos, self.workspace_lo, self.workspace_hi)
        quat = quat_mul(quat_from_rotvec(drot), self._target_quat)
        self._target_quat = self._limit_orientation(quat)

    def _limit_orientation(self, quat: np.ndarray) -> np.ndarray:
        """Clamp tilt (TCP z-axis vs straight down) and yaw (about world z) of a target."""
        # Decompose quat = q_yaw * q_tilt * q_home, with q_yaw about world z.
        rel = quat_mul(quat, self._conj(self._home_quat))
        rel_mat = np.empty(9)
        mujoco.mju_quat2Mat(rel_mat, rel)
        rel_mat = rel_mat.reshape(3, 3)
        yaw = np.arctan2(rel_mat[1, 0], rel_mat[0, 0])
        q_yaw = quat_from_rotvec(np.array([0.0, 0.0, yaw]))
        q_tilt = quat_mul(self._conj(q_yaw), rel)
        tilt_vec = quat_error(q_tilt, np.array([1.0, 0.0, 0.0, 0.0]))
        tilt = np.linalg.norm(tilt_vec)
        if tilt > self.max_tilt:
            tilt_vec *= self.max_tilt / tilt
        yaw = np.clip(yaw, -self.max_yaw, self.max_yaw)
        q = quat_mul(
            quat_from_rotvec(np.array([0.0, 0.0, yaw])),
            quat_mul(quat_from_rotvec(tilt_vec), self._home_quat),
        )
        return q / np.linalg.norm(q)

    @staticmethod
    def _conj(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def _solve_ik(self, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        """Damped least-squares IK from the previous joint command, with a posture bias."""
        m, d = self.model, self._ik_data
        q = self._q_cmd.copy()
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        lam2 = self.ik_damping**2
        eye7 = np.eye(len(q))
        for _ in range(self.ik_iters):
            d.qpos[self._arm_qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = np.concatenate(
                [
                    target_pos - d.site_xpos[self._tcp_sid],
                    quat_error(target_quat, mat_to_quat(d.site_xmat[self._tcp_sid])),
                ]
            )
            mujoco.mj_jacSite(m, d, jacp, jacr, self._tcp_sid)
            jac = np.vstack([jacp[:, self._arm_dadr], jacr[:, self._arm_dadr]])
            jjt = jac @ jac.T + lam2 * np.eye(6)
            jpinv = jac.T @ np.linalg.solve(jjt, np.eye(6))
            dq = jpinv @ err
            dq += (eye7 - jpinv @ jac) @ (self.ik_nullspace_gain * (self._q_home - q))
            q = np.clip(q + dq, self._arm_range[:, 0], self._arm_range[:, 1])
            if np.linalg.norm(err) < 1e-4:
                break
        return q

    # ---------------------------------------------------------------- contacts
    def _body_contacts(self) -> list[tuple[int, int, int, int]]:
        """(geom1, geom2, body1, body2) for every active contact."""
        m, d = self.model, self.data
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            out.append((c.geom1, c.geom2, m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]))
        return out
