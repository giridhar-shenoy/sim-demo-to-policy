# NOTES — sim-demo-to-policy (R4)

Working log for the sprint session. Newest sections at the bottom. A fresh session
should be able to resume from this file alone.

## Session 1 — night of 29 Oct 2026 (overnight mode, started 02:42 IST)

### Plan (v0.1 MUST list only)

1. Skeleton: LICENSE, README stub, NOTES/CHANGELOG/THIRD_PARTY, Dockerfile
   (python:3.12-slim), devcontainer, CI (docker build + lint + pytest), pre-commit,
   docs/results/scripts placeholders. Push to the existing public repo, get CI green
   (cap 1 h of CI fixing).
2. `PandaPickPlace-v0` gymnasium env on the Menagerie Panda: table, 4 cm cube, 6 cm
   target square, 500 Hz physics / 20 Hz control, two 96×96 cameras (front, wrist),
   seedable cube pose, success = cube in region + gripper open + 0.5 s settled.
   Action = 6-DoF delta EE pose + absolute gripper, executed by damped-least-squares
   IK onto the Panda's position actuators (not mocap; see Decisions).
   Tests: seed determinism, action bounds, success on a scripted trajectory.
3. Throughput check (≥ 200 env steps/s headless) → CHECKPOINT 1 numbers.
4. Scripted waypoint expert; success rate on 100 seeds → CHECKPOINT 1.
5. Recorder: LeRobot v3.0 dataset writer (LeRobotDataset.create/add_frame/
   save_episode/finalize), per-episode success flag, demo seeds 0–999.
   Record 50 scripted demos. Load-back test with 2 tiny episodes → CHECKPOINT 2.
6. Training wrapper `sdp train` → `lerobot-train` with overrides from YAML
   (ACT CPU config: dim_model 256, 2 enc / 1 dec, chunk 20, 96×96, batch 8,
   3000 steps). One run ≤ 45 min under heavy+build+gazebo locks, wall time recorded.
7. Eval harness: ≥ 50 nominal + ≥ 50 shifted-pose trials, eval seeds 10000–19999,
   Wilson 95 % CIs, episode length, failure taxonomy; CSV + JSON → CHECKPOINT 3.
8. One labelled GIF (one success, one failure) from offscreen rendering.
9. README (required shape + "Reading the numbers" + "What transfers to hardware"),
   docs/architecture.md (Mermaid), docs/results.md, CI smoke tests (50-step train,
   3-episode eval) → CHECKPOINT 4.

Out of scope this week (goes to "What's next"): PandaPegInsert, So101PickPlace,
diffusion policy, data-size ablation, Colab notebook, teleop (keyboard/SpaceMouse),
R3 bridge, domain randomisation beyond object pose.

### Verified facts (commands run this session)

- `python3 --version` → 3.14.4 (system). Repo venv: `uv venv --python 3.12 .venv`
  → Python 3.12.16 (uv 0.13.6, `~/.local/bin/python3.12`).
  Why 3.12 and not 3.14: the CI/devcontainer image is `python:3.12-slim` (sprint
  override), so the laptop venv uses the same interpreter minor version as CI.
- `lerobot==0.6.1` caps torch at 2.11: a dry run on top of torch 2.15+cpu wanted to
  replace it with `torch==2.11.0` from PyPI plus 15 `nvidia-*` CUDA wheels. Fix:
  install `torch==2.11.0 torchvision==0.26.0` from
  `https://download.pytorch.org/whl/cpu` first; the lerobot dry run then pulls no
  nvidia packages.
- Installed (`uv pip freeze` → `requirements.lock`): torch 2.11.0+cpu, torchvision
  0.26.0+cpu, lerobot 0.6.1 (extras dataset,training), mujoco 3.17.0, gymnasium
  1.4.0, numpy 2.2.6, av 15.1.0, torchcodec 0.11.1, datasets 4.8.5, ruff 0.17.3,
  pytest 9.2.0, pre-commit 4.7.0.
- `torch.cuda.is_available()` → False; `torch.get_num_threads()` → 4.
- Import paths (0.6.1): `lerobot.policies.act.modeling_act.ACTPolicy`,
  `lerobot.policies.act.configuration_act.ACTConfig`,
  `lerobot.datasets.lerobot_dataset.LeRobotDataset`.
- `LeRobotDataset.create(repo_id, fps, features, root=None, robot_type=None,
  use_videos=True, ..., image_writer_threads=0, video_backend=None, ...)`;
  `add_frame(frame: dict)`; `save_episode(episode_data=None, parallel_encoding=True)`;
  `finalize()` exists.
- ACTConfig defaults (0.6.1): chunk_size 100, n_action_steps 100, dim_model 512,
  n_encoder_layers 4, n_decoder_layers 1, use_vae True, vision_backbone resnet18,
  pretrained_backbone_weights ResNet18_Weights.IMAGENET1K_V1, **push_to_hub True**
  (so the wrapper must pass `--policy.push_to_hub=false`).
- `lerobot-train --help` (0.6.1) override syntax: draccus dotted flags,
  `--key=value`. Relevant: `--dataset.repo_id`, `--dataset.root`,
  `--dataset.video_backend`, `--policy.type=act`, `--policy.device=cpu`,
  `--output_dir`, `--job_name`, `--seed`, `--steps`, `--batch_size`,
  `--num_workers`, `--log_freq`, `--save_freq`, `--save_checkpoint`,
  `--wandb.enable=false`, `--dataloader_multiprocessing_context` (default spawn),
  `--env_eval_freq` (default 20000; we have no lerobot env, keep > steps or 0).
  Policy sub-fields (`--policy.dim_model=256` etc.) are not printed by --help because
  the policy is a choice type; verified by a real 1-step run (see training section).
- MuJoCo offscreen on the laptop (`python scripts/probe_render.py`): egl ok, osmesa ok.
- Menagerie pinned at f3b6013629c7b94f8ee93fd977b161fa8350f9bc
  (`~/work/menagerie`, shared read-only). `franka_emika_panda/` is 36 MB (33 MB
  meshes), Apache-2.0, so it is fetched by `scripts/fetch_menagerie.sh` (sparse,
  depth-1 fetch of the pinned commit) instead of vendored. Locally:
  `export SDP_MENAGERIE_DIR=~/work/menagerie`.
- Docker is not installed on the laptop, so `docker build .` cannot be run here; the
  image is built only by GitHub Actions.
- markdownlint MD060 (aligned table pipes) disabled: generated tables can't keep
  column alignment.

### Decisions

- **Action space: DLS IK, not mocap.** The env converts a delta EE pose into joint
  position targets with damped least squares and sends them to the Menagerie
  Panda's own position actuators (kp 4500/3500/2000). A mocap weld would teleport
  the hand and make joint state and actuator limits meaningless; IK keeps the
  joint-space observation honest and is closer to how a real arm is commanded.
  (The prompt lists this as "ask me"; overnight mode — recorded in MORNING.md.)
- Python 3.12 via uv (see above).

### Commands that worked

```bash
uv venv --python 3.12 .venv
flock ~/work/.locks/build.lock uv pip install --python .venv/bin/python \
  "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cpu
flock ~/work/.locks/build.lock uv pip install --python .venv/bin/python \
  "lerobot[dataset,training]==0.6.1" mujoco gymnasium pytest ruff pre-commit imageio
uv pip install --python .venv/bin/python --no-deps -e .
```

### Time log

- 02:42 start; 03:10 skeleton written, local checks green.
