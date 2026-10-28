# sim-demo-to-policy

[![CI](https://github.com/giridhar-shenoy/sim-demo-to-policy/actions/workflows/ci.yml/badge.svg)](https://github.com/giridhar-shenoy/sim-demo-to-policy/actions/workflows/ci.yml)

![ACT on PandaPickPlace-v0: one held-out success (left) and one failure (right)](docs/assets/act_success_failure.gif)

A MuJoCo pick-and-place environment on the Franka Panda, a scripted expert whose
demonstrations are recorded as a LeRobot v3.0 dataset, an ACT policy trained on a laptop
CPU with `lerobot-train`, and an evaluation harness with held-out seeds, a failure taxonomy
and confidence intervals.
It shows the whole demonstration-to-policy loop running end to end on a CPU, with every
success rate traceable to a per-episode CSV whose seeds never overlap the demonstrations.
<!-- sdp-report:headline:begin -->
<!-- sdp-report:headline:end -->

**Language split:** Python only. The imitation-learning stack (LeRobot, PyTorch,
gym-style MuJoCo envs) is Python, and a C++ port would be unusual rather than impressive.
My C++ control and planning work is in
[ur5e-teaching-controllers](https://github.com/giridhar-shenoy/ur5e-teaching-controllers) and
[pointcloud-constrained-planning](https://github.com/giridhar-shenoy/pointcloud-constrained-planning).

## Run it

```bash
docker build -t sdp . && docker run --rm sdp scripts/regenerate_all.sh
```

This reruns everything inside the CI image and prints every number. Results written
inside the container are discarded when it exits.

Without Docker (Python 3.12; the venv is uv-managed because lerobot 0.6.1 caps torch at 2.11):

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install --no-deps -e .
scripts/fetch_menagerie.sh && export SDP_MENAGERIE_DIR=$PWD/third_party/mujoco_menagerie MUJOCO_GL=egl
scripts/regenerate_all.sh   # ~40 min on a 4-core laptop CPU, ~25 min of it training
```

`regenerate_all.sh` measures env throughput, scores the scripted expert, records 50
demos, trains ACT, evaluates 100 + 100 held-out episodes, renders the GIF and rewrites
the tables below. The trained checkpoint is 96 MB, too large to commit; until it is
attached to a GitHub Release, the script retrains it rather than downloading it.

## Results

<!-- sdp-report:success:begin -->
<!-- sdp-report:success:end -->

Failure labels (one per episode, defined in [`sdp/eval/taxonomy.py`](sdp/eval/taxonomy.py)):

<!-- sdp-report:taxonomy:begin -->
<!-- sdp-report:taxonomy:end -->

<!-- sdp-report:training:begin -->
<!-- sdp-report:training:end -->

More tables (dataset, env throughput, seed ranges, pose boxes) and the command behind
every row: [docs/results.md](docs/results.md).

### Reading the numbers

Each ACT row is 100 independent episodes. Each episode starts from a cube pose fixed
by its seed, and the seeds are 10000–10099, from the eval range 10000–19999. The 50
demonstrations come from seeds 0–49, in the demo range 0–999. The two ranges cannot
overlap: `sdp/seeds.py` enforces this and `tests/test_seed_split.py` tests it. The
shifted condition reuses the same seeds with the cube's start box moved by
(+8 cm, −8 cm), so each shifted episode is a paired twin of a nominal one; about 22 %
of the shifted box still overlaps the training box. An episode is a success only if
the cube's centre ends inside the 6 cm square, resting on the table, with the gripper
open and the cube still, and all of that holds for 0.5 s. The episode limit is
12.5 s. The intervals are 95 % Wilson score intervals. With 100 trials a rate near
50 % carries about ±10 points, and a rate near 0 % or 100 % is still only bounded
within about 4 points. So differences of less than about 15 points between rows are
not evidence of anything, and one training seed says nothing about seed-to-seed
variance.

## Where it fails

TODO (filled from the eval results)

## What transfers to hardware

The state half of the observation transfers almost as-is: the joint angles, the finger
gap and a TCP pose computed by forward kinematics are what a real Panda reports. The
image half does not. The sim cameras have no noise, blur, exposure change or
distortion, the lighting is flat, and the observation cameras draw collision meshes
to stay fast. The action is a 20 Hz delta TCP pose that the env turns into joint
targets with damped least-squares IK and near-perfect position actuators. On a real
arm it would become a Cartesian setpoint for a 1 kHz controller with velocity limits.
That is the layer I built on production 6-axis arms (Jacobian TCP jogging,
gravity-compensated hand guiding), and the sim never shows the policy its lag or
overshoot. Sim latency is zero apart from the network's own compute, which the harness
logs per episode. Resets are free and exact here. On hardware they cost supervised
robot time, and the seeds lose their meaning. Contact is MuJoCo's soft model with
elliptic cones, forgiving for a 50 g cube; it would be the first gap to show on a
contact-rich task. The longer version, marked as a draft:
[docs/hardware_transfer.md](docs/hardware_transfer.md).

## What I'd do next

- **Teleoperated data**: keyboard teleop in the gym_hil key layout, then SpaceMouse,
  then the R3 retargeting pipeline as the teleop device. That would give a
  scripted-vs-teleop data comparison on the same eval seeds.
- **Diffusion policy** at the same CPU budget, and full-size ACT and diffusion runs
  (LeRobot defaults, 100k steps) in a Colab/Kaggle notebook.
- **Harder tasks**: PandaPegInsert (2 cm peg, 2.2 cm chamfered hole, contact-rich) and
  So101PickPlace (5-DoF low-cost arm, reduced workspace).
- **Data-size ablation** (10/25/50/100 demos) with several training seeds, so the
  intervals cover seed-to-seed variance and not just trial sampling.
- **Domain randomisation** beyond the object pose (lighting, colours, camera pose), then
  HIL-SERL-style fine-tuning in the gym_hil pattern.
- **Real-robot data** collected through R1's controllers, using the same dataset schema.

## Repository layout

| Path | What |
|---|---|
| `sdp/envs/` | `MujocoArmEnv` base (Menagerie loading, cameras, DLS IK, 500/20 Hz timing) and `PandaPickPlace-v0` |
| `sdp/expert/` | scripted waypoint expert |
| `sdp/record/` | LeRobot v3.0 writer + recorder (per-episode success sidecar) |
| `sdp/train/` | `lerobot-train` wrapper and the ACT CPU config |
| `sdp/eval/` | harness, failure taxonomy, Wilson intervals, ACT adapter, report generator |
| `scripts/` | one script per number or GIF; `regenerate_all.sh` runs them all |
| `results/` | per-episode CSVs and JSON summaries behind every table |
| `docs/` | [architecture](docs/architecture.md), [results](docs/results.md), [tasks](docs/tasks.md), [hardware transfer](docs/hardware_transfer.md) |

## License

MIT, see [LICENSE](LICENSE). Third-party components and their licenses:
[THIRD_PARTY.md](THIRD_PARTY.md).
