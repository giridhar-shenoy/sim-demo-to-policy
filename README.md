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
ACT (run 2: 3000 training steps, 17 min on the laptop CPU, 50 scripted demonstrations) succeeds in 0/100 (0 %, Wilson 95 % 0 %–4 %) held-out test episodes (seeds 10100–10199), and in 0/100 (0 %, Wilson 95 % 0 %–4 %) with the cube's start box shifted by 8 cm.
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
scripts/regenerate_all.sh   # ~70 min on a 4-core laptop CPU, ~40 min of it training
```

`regenerate_all.sh` measures env throughput, scores the scripted expert, records 50
demos, trains both ACT runs, evaluates them on the dev and test seeds, renders the GIF
and rewrites the tables below. Each checkpoint is 96 MB, too large to commit. Until
they are attached to a GitHub Release (`scripts/download_release_assets.sh`), the
script retrains them rather than downloading them.

## Results

<!-- sdp-report:success:begin -->
| Policy / data | Cube poses | Seeds | Trials | Success | Wilson 95 % CI | Mean length, successes (steps) | Per-episode CSV |
|---|---|---|---|---|---|---|---|
| expert_demo_seeds | nominal | 0–99 | 100 | 100/100 (100 %) | 96 %–100 % | 109.9 | [`results/expert_pickplace.csv`](results/expert_pickplace.csv) |
| expert_eval_seeds | nominal | 10000–10099 | 100 | 100/100 (100 %) | 96 %–100 % | 110.6 | [`results/expert_pickplace.csv`](results/expert_pickplace.csv) |
| expert_eval_seeds | shifted | 10000–10099 | 100 | 100/100 (100 %) | 96 %–100 % | 117.2 | [`results/expert_pickplace.csv`](results/expert_pickplace.csv) |
| act_cpu_scripted50 | nominal | 10000–10099 | 100 | 0/100 (0 %) | 0 %–4 % | – | [`results/eval_act_cpu_scripted50_s10000.csv`](results/eval_act_cpu_scripted50_s10000.csv) |
| act_cpu_scripted50 | shifted | 10000–10099 | 100 | 0/100 (0 %) | 0 %–4 % | – | [`results/eval_act_cpu_scripted50_s10000.csv`](results/eval_act_cpu_scripted50_s10000.csv) |
| act_cpu_scripted50 | nominal | 10100–10199 | 100 | 0/100 (0 %) | 0 %–4 % | – | [`results/eval_act_cpu_scripted50_s10100.csv`](results/eval_act_cpu_scripted50_s10100.csv) |
| act_cpu_scripted50 | shifted | 10100–10199 | 100 | 0/100 (0 %) | 0 %–4 % | – | [`results/eval_act_cpu_scripted50_s10100.csv`](results/eval_act_cpu_scripted50_s10100.csv) |
| act_cpu_v2_scripted50 | nominal | 10100–10199 | 100 | 0/100 (0 %) | 0 %–4 % | – | [`results/eval_act_cpu_v2_scripted50_s10100.csv`](results/eval_act_cpu_v2_scripted50_s10100.csv) |
| act_cpu_v2_scripted50 | shifted | 10100–10199 | 100 | 0/100 (0 %) | 0 %–4 % | – | [`results/eval_act_cpu_v2_scripted50_s10100.csv`](results/eval_act_cpu_v2_scripted50_s10100.csv) |
<!-- sdp-report:success:end -->

Failure labels (one per episode, defined in [`sdp/eval/taxonomy.py`](sdp/eval/taxonomy.py)):

<!-- sdp-report:taxonomy:begin -->
| Policy / data | Cube poses | success | collision | never_reached | grasp_missed | dropped | placed_outside | timeout |
|---|---|---|---|---|---|---|---|---|
| expert_demo_seeds | nominal | 100 | 0 | 0 | 0 | 0 | 0 | 0 |
| expert_eval_seeds | nominal | 100 | 0 | 0 | 0 | 0 | 0 | 0 |
| expert_eval_seeds | shifted | 100 | 0 | 0 | 0 | 0 | 0 | 0 |
| act_cpu_scripted50 | nominal | 0 | 3 | 97 | 0 | 0 | 0 | 0 |
| act_cpu_scripted50 | shifted | 0 | 0 | 100 | 0 | 0 | 0 | 0 |
| act_cpu_scripted50 | nominal | 0 | 3 | 97 | 0 | 0 | 0 | 0 |
| act_cpu_scripted50 | shifted | 0 | 0 | 100 | 0 | 0 | 0 | 0 |
| act_cpu_v2_scripted50 | nominal | 0 | 7 | 93 | 0 | 0 | 0 | 0 |
| act_cpu_v2_scripted50 | shifted | 0 | 5 | 95 | 0 | 0 | 0 | 0 |
<!-- sdp-report:taxonomy:end -->

<!-- sdp-report:training:begin -->
| Run | Config | Steps | Wall time | Final loss | Machine |
|---|---|---|---|---|---|
| act_cpu_scripted50 | `sdp/train/configs/act_cpu.yaml` | 3000 | 21.0 min | 0.096 | i5-1135G7 (4C/8T), 14 GB RAM, CPU only, torch 4 threads |
| act_cpu_v2_scripted50 | `sdp/train/configs/act_cpu_v2.yaml` | 3000 | 17.1 min | 0.082 | i5-1135G7 (4C/8T), 14 GB RAM, CPU only, torch 4 threads |
<!-- sdp-report:training:end -->

More tables (dataset, env throughput, seed ranges, pose boxes) and the command behind
every row: [docs/results.md](docs/results.md).

### Reading the numbers

Every ACT row is 100 independent episodes. Each episode's starting cube pose is fixed
by its seed. The 50 demonstrations use seeds 0–49, from the demo range 0–999. Eval
seeds come from 10000–19999, so the ranges cannot overlap: `sdp/seeds.py` enforces it
and `tests/test_seed_split.py` tests it. There were two training runs. Run 1 scored
0/100 on seeds 10000–10099, which made those the *dev* seeds. A check on run 1's own
training frames showed it ignored the cube's position, so run 2 changed two settings
(below). Both runs were then scored once on seeds 10100–10199, the *test* seeds,
which no configuration choice had seen. The headline uses those. The shifted
condition reuses the same seeds with the cube's start box moved by (+8 cm, −8 cm), so
each shifted episode is a paired twin of a nominal one; about 22 % of the shifted box
still overlaps the training box. An episode counts as a success only if the cube's
centre ends inside the 6 cm square, resting on the table, with the gripper open and
the cube still, and all of that holds for 0.5 s within the 12.5 s limit. The
intervals are 95 % Wilson score intervals. With 100 trials a rate near 50 % carries
about ±10 points, and 0/100 still allows up to 4 %. So differences under ~15 points
between rows are not evidence, and one training seed per run says nothing about
seed-to-seed variance.

## Where it fails

- **The ACT policy never reaches the cube.** Both CPU runs score 0/100, nominal and
  shifted, on dev and test seeds. 93–100 % of the episodes are `never_reached` and the
  rest are `collision` (fingers pressed into the table). The policy descends straight
  down from the home pose instead of first moving over the cube. On its own training
  frames it reproduces the descend, close, lift and carry actions. But its first x, y
  and yaw outputs are identical (std 0.00) across 20 training episodes whose cubes sit
  in different places, while the recorded actions vary (std 0.38 in x, 0.55 in yaw)
  ([`results/openloop_*.json`](results/)). Swapping in another episode's images barely
  moves the output, while black images change it a lot. So the network uses the
  images, but not to find the cube. The 15-D proprioceptive state predicts every
  phase except the ~10 approach frames per episode, and 3000 steps of batch 8
  (≈ 4.4 epochs) did not learn that remaining, vision-only part from 96 px images.
  Run 2's changes (backbone lr 1e-4, CVAE off) did not change this. This is
  underfitting under the spec'd CPU budget, not distribution shift. The shifted
  condition says nothing yet, because there is no nominal success to lose.
- **Only scripted data.** All 50 demos come from one deterministic waypoint expert, so
  every demonstration follows the same path with the same timing. That is the easiest
  data an imitation learner can get. Teleoperated demos are slower, noisier and
  multimodal, and they are what the method has to handle on a real robot. There is no
  teleop row because teleop needs a person at the keyboard; that scripted-vs-teleop gap
  is unmeasured.
- **Image resolution.** At 96×96 the 4 cm cube covers 41–59 pixels (a bounding box
  6–10 px wide) in the front image, measured over 20 eval seeds per distribution. The policy finds it mostly from the wrist camera once the hand is close, and
  the cube's yaw is barely resolvable from the front view.
- **CPU-sized model and schedule.** dim_model 256, 2 encoder and 1 decoder layers,
  3000 steps of batch 8 is about 4.4 epochs over 5.5k frames, with one training seed.
  LeRobot's defaults are 100k steps. These rates describe this budget, not ACT.
- **No peg insertion and no So-101 yet.** The contact-rich task and the low-cost
  5-DoF arm are deferred, so nothing here says how the pipeline does on contact-rich
  manipulation or on a reduced workspace.
- **Only one kind of shift.** The shifted condition moves the cube's start box. The
  lighting, colours, camera poses, target position and object are never varied.

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

- **Make ACT localise the cube** before anything else: train longer than the spec'd
  3000 CPU steps, or at full size on a free GPU. Alternatively, withhold or delay the
  proprioceptive state, so it can't stand in for the cube's position, and check that
  the open-loop image-swap test then moves the approach actions.
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
