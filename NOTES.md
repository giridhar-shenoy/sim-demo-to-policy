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

- gym-hil 0.1.14 license on PyPI: `Apache-2.0` (classifier "Apache Software License").
- LeRobot 0.6.1 checkpoint layout: `<out>/checkpoints/<step>/pretrained_model/`
  {config.json, model.safetensors, train_config.json, policy_preprocessor.json +
  normalizer safetensors, policy_postprocessor.json + unnormalizer safetensors};
  `checkpoints/last` symlink. Preprocessor steps: rename_observations, to_batch,
  device, normalizer; postprocessor: unnormalizer, device. Eval mirrors
  `lerobot_eval`: preprocessor -> `policy.select_action` -> postprocessor.
  Load with `ACTPolicy.from_pretrained(path)` +
  `lerobot.policies.factory.make_pre_post_processors(cfg, pretrained_path=path,
  preprocessor_overrides={"device_processor": {"device": "cpu"}})`.
- Checkpoint size: 96 MB (> 50 MB) → cannot be committed; needs a GitHub Release
  (Giri uploads) → WAITING FOR GIRI.
- lerobot-train accepted `--policy.dim_model=256 --policy.n_encoder_layers=2
  --policy.n_decoder_layers=1 --policy.chunk_size=20 --policy.n_action_steps=10
  --policy.optimizer_lr=0.0001 --policy.optimizer_lr_backbone=1e-05
  --dataset.episodes=[0,...,49]` (40-step probe, returncode 0).
- Probe speed: steady state `updt_s` ≈ 0.50 s, `data_s` ≈ 0.006 s per step (batch 8,
  num_workers 2) → 3000 steps ≈ 26 min projected (< 45 min budget).
- Default video codec in LeRobot 0.6.1 datasets: AV1 (SVT-AV1 encoder, CRF 30, gop 2);
  `SVT_LOG=1` silences its info output.
- `pgrep -f "gz sim"` matches its own caller's command line (and any echo text that
  contains "gz sim"); use `pgrep -f "[g]z sim"` from a script whose text has no literal
  "gz sim". Cost me two false "waiting for Gazebo" starts.

### Env findings

- First env throughput was 26 steps/s with images. Profiling: 4096 px shadow map
  (14 → 4.7 ms per 96 px frame), then the Panda's 400k-vertex visual meshes
  (readPixels sync 3.9 ms). Fix: shadows off; observation cameras draw the collision
  meshes (group 3, 3.3k vertices) → 448 steps/s with images, 1097 without.
- The expert first steered the *measured* TCP; the integrated target ran ahead of the
  lagging arm and drove the fingers into the table at the grasp (flagged as
  collisions). Steering the integrated target fixed it: 100/100.
- Drop detection first fired on one-step finger-contact flicker during carry; now a
  drop is judged when the carried cube lands out of the fingers while the gripper is
  commanded closed.
- Eval seed design: nominal and shifted use the same seeds (paired); with identical
  uniform draws the shifted pose is exactly nominal + (0.08, −0.08) m.

### Training

- LR: ACT default 1e-5 (tuned for ~100k steps) → 1e-4 for 3000 steps, backbone 1e-5.
  No ablation was run (no time); recorded as an open question.
- Full run started 03:25:33 IST under heavy+build+gazebo locks via
  `/tmp/r4/run_train_locked.sh` → `scripts/train_cpu.sh`; log `/tmp/r4/train_full.log`,
  outputs `outputs/train/act_cpu_scripted50/`, wrapper log/csv/json beside it.

### Time log

- 02:42 start; 02:57 skeleton committed; 03:00 CI green on skeleton (4m52s).
- 03:17 CHECKPOINT 1 (448 steps/s, expert 100/100). 03:21 CHECKPOINT 2 (50 demos).
- 03:25 ACT training started (projected end ~03:52).
- 03:32 **systemd-oomd killed this session and the training run** (step 590/3000; swap
  was full machine-wide, two sessions killed). No checkpoint was saved (save_freq
  1000). New machine rules from 03:35: wait for ≥ 5 GB MemAvailable inside the locks
  before heavy jobs; dataloader workers 0 or 1 → `num_workers: 1` in act_cpu.yaml
  (was 2). Batch size unchanged (8).
- 03:34 Commit d8bde37 ("feat: add lerobot-train wrapper and the ACT CPU config")
  also contains the eval adapter, `sdp eval`/`sdp report`, report.py, the GIF
  script and the docs (architecture, results, tasks, hardware_transfer,
  THIRD_PARTY): everything was still staged from a pre-commit run when I committed
  the first group. It is unpushed, but splitting it is a history rewrite, which the
  overnight rules do not allow, so it stays as one commit and is flagged in MORNING.md.

### ACT run 1 and run 2 (29 Oct, 03:46–04:45)

- Run 1 (`act_cpu.yaml`): 03:46:03–04:07:03, wall 1259.8 s (21.0 min), final loss
  0.096 (L1 0.068, KL 0.003). Eval on seeds 10000–10099 (now "dev"): nominal 0/100
  (97 never_reached, 3 collision), shifted 0/100 (100 never_reached). Policy compute:
  mean ≈ 3 ms per step, re-plan 26–42 ms (one 212 ms outlier in nominal).
- Diagnosis (`/tmp/r4/debug_openloop.py`, now `scripts/diagnose_openloop.py`): on its
  own training frames run 1 reproduces descend (z −0.59 vs −0.6), gripper, lift (z
  0.5 vs 0.6) and carry (y 0.61 vs 0.6), but its first-step x / y / yaw outputs have
  std 0.00 across 20 training episodes (data std 0.38 / 0.04 / 0.55). It ignores the
  cube's position and also misses the constant y = −0.6 of the first frames.
  Env observation at seed 0 vs dataset frame 0: state identical, images differ by
  1.6 (front) / 2.1 (wrist) out of 255 (AV1). So not a pipeline bug.
- Run 2 (`act_cpu_v2.yaml`): backbone lr 1e-4 and use_vae false. A third change
  (`replace_final_stride_with_dilation: true`) failed in 4 s: torchvision ResNet-18
  BasicBlock raises "Dilation > 1 not supported", so it was dropped. Run 2:
  04:27:57–04:45:03, wall 1026.4 s (17.1 min), final L1 0.082 (not comparable to run
  1's L1, which used the CVAE latent computed from the true actions).
- Training-CSV bug found and fixed (0cf43ef): lerobot prints `step:2K`, so the step
  column is rebuilt from `log_freq`; run 1's CSV was rebuilt from its log with
  `python -m sdp.train.train --output-dir outputs/train/act_cpu_scripted50 --reparse-log`.
- 04:45 post-run job (open-loop diagnosis of both runs, test-seed evals 10100–10199,
  GIF) queued on the heavy lock behind R2's chain (`/tmp/r4/run_post_v2.sh`, log
  `/tmp/r4/post_v2.log`).

### Evals, diagnosis, CI (29 Oct, 04:45–05:45)

- 05:04–05:16 test-seed evals (10100–10199): run 1 and run 2 both 0/100 nominal and
  0/100 shifted. Run 2 labels: nominal 93 never_reached + 7 collision, shifted 95 + 5.
- Image test (`/tmp/r4/debug_img.py`, run 05:17): same state, another episode's
  images → output change ≤ 0.005; black / noise images → z, y, gripper change by up to
  0.6–1.0. Image normalisation uses ImageNet stats (`dataset.use_imagenet_stats`
  default). The images are used, but not to locate the cube.
- Decision: no third run without Giri (it would need more than the spec'd 3000 steps
  or a change to the policy inputs) → MORNING #9.
- GIF fallback: with no ACT success, the left panel is the scripted expert on the
  failure's seed, labelled "Scripted expert (reference)".
- 05:30 CHECKPOINT 3 logged; pushed; CI run 40791132757 green: 37 passed in 39.75 s
  incl. the 50-step train + 3-episode eval smoke; job ~3 min.
- `results/eval_act_pickplace.{csv,json}` is now the combined file the spec names
  (written by `sdp report`), alongside the per-call files.
- 05:28 queued `/tmp/r4/run_diag.sh` (open-loop with image-swap fields, training-seed
  rollouts, GIF) behind R2's `chain2.sh`, which holds all three locks.
