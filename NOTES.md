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
- 05:57 README install command verified by dry run in a throwaway venv
  (`/tmp/r4/venvcheck`): `uv pip install -r requirements.lock --extra-index-url
  https://download.pytorch.org/whl/cpu` FAILS (uv's first-index strategy finds certifi
  on the PyTorch index at another version); adding `--index-strategy unsafe-best-match`
  resolves 96 packages, torch 2.11.0+cpu, no nvidia wheels (torchcodec 0.11.1+cpu).
- 05:52 ran the full pytest (incl. the 50-step training smoke) without the heavy lock;
  checked afterwards that no session held any lock at the time. Full test runs now go
  under heavy+build.
- 05:54 release tarballs in `outputs/release/`: act_cpu_scripted50 89 MB,
  act_cpu_v2_scripted50 62 MB, pickplace_scripted_50 5.2 MB.

## Session 1 — CHANGELOG entry

```markdown
## [Unreleased] — session of 29 Oct 2026 (overnight)

### Added
- Repo skeleton (MIT, python:3.12-slim Dockerfile, devcontainer, CI, pre-commit, docs).
- PandaPickPlace-v0 (Menagerie Panda, DLS-IK delta-TCP actions, 500/20 Hz, two 96 px
  cameras, seedable nominal/shifted cube poses, 0.5 s settled success); 448 steps/s
  with images on the laptop.
- Scripted waypoint expert: 100/100 on demo and eval seeds, both pose distributions.
- LeRobot v3.0 recorder with a per-episode success sidecar; 50 scripted demos.
- lerobot-train wrapper; ACT CPU configs run 1 (act_cpu.yaml) and run 2 (act_cpu_v2.yaml).
- Eval harness: disjoint seed split (dev/test), failure taxonomy, Wilson 95 % CIs,
  policy timing; sdp record/eval/report; generated doc tables.
- Diagnostics: open-loop + image-swap check, training-seed rollouts.
- README GIF (scripted expert reference vs ACT failure), docs, release-asset scripts.

### Results
- ACT run 1 (21.0 min) and run 2 (17.1 min): 0/100 nominal and 0/100 shifted on dev
  and test seeds; both ignore the cube's position (underfitting at 3000 steps).

### Fixed
- Training CSV step column (lerobot prints rounded "2K"), Wilson exact 0/1 bounds,
  adapter warm-up for latency, README uv install command.
```

## Open questions for Giri

1. **Third ACT run.** Which option from MORNING #9: (a) more than the spec'd 3000 steps
   (e.g. 8000 ≈ 45 min), (b) withhold the proprioceptive state from the policy, or
   (c) ship v0.1 at 0 % and do the full-size Colab run next week? I recommend (a).
2. Was run 2 within what you approved? I read "the ACT training run (45 min or less)"
   as a per-run budget. If you meant exactly one run, run 2 and its rows should go.
3. Is the GIF substitution acceptable: scripted expert as the "success" panel, labelled
   as the reference, because ACT has no success?
4. OK to publish release v0.1.0 with the three tarballs in `outputs/release/` (157 MB)?
5. Repo topics (proposed list in MORNING #1) and branch protection on `main`?
6. Leave the bundled commit d8bde37 as is, or split it (history rewrite)?
7. `docs/hardware_transfer.md` is a first-person DRAFT. Please edit; it uses only
   "production 6-axis arms", no employer name.
8. The ACT learning rate (1e-4; backbone 1e-5 in run 1, 1e-4 in run 2) was chosen
   without an ablation. Do you want one?
9. Presentation: the README's third sentence is the 0/100 headline. Keep it there,
   or lead with the diagnosis?
10. When can you do the keyboard-teleop session (SHOULD, 30 demos)?
11. The README links to R1/R2 404 until those repos are public.

## Session 2 — 29 Oct 2026, from 12:08 IST (Giri awake; overnight rules off)

- Giri answered open question 1: option **(a)**, one more ACT CPU run identical to run 2
  except for the step count (8000), within the 45-min cap, locks taken with `flock -o`,
  ≥ 5 GB free first, evaluated on fresh test seeds 10200–10299. Open question 11 is
  resolved: the R1/R2 links in the README return 200 (MORNING item dropped).
- `sdp/train/configs/act_cpu_v3.yaml` = `act_cpu_v2.yaml` with `steps: 8000` (diff shows
  only the header comment and that line).
- Timing probe (150 steps, all three locks, 12:08:46–12:09:41): steady state `updt_s`
  0.27–0.29 + `data_s` 0.004 → 0.28 s/step → 8000 steps ≈ 37.3 min + ~15 s startup ≈
  38 min projected (< 45). Step-100 loss 0.282 = run 2's step-100 loss (same seed and
  config up to that point).
- Run 3 started 12:09:58 (`/tmp/r4/run_train_v3.sh`, log `/tmp/r4/train_v3.log`). Early
  rate 0.32 s/step, slower than the probe because the desktop browser was using ~40 %
  of a core; projected ~43 min. Rule I applied: if the projection (elapsed + remaining ×
  3-min rate) exceeds 45 min after step 1000, stop the run and tell Giri
  (`/tmp/r4/watch_v3.sh`).
- **12:17 run 3 stopped at step ~1190 of 8000, projected over the 45-min cap.** INFO-line
  timestamps: step 50 at 12:10:19, step 1000 at 12:15:50 → 0.348 s/step (200-step
  windows 0.32, 0.38, 0.345, 0.365) → ~46.4 min of steps + startup + checkpoint saves ≈
  47 min. The probe's 0.28 s/step was measured with an idle desktop; during the run the
  browser used ~40 % of a core. The cap allows ≤ 0.3375 s/step. Not shortened, not
  continued; asked Giri. The partial run is in `outputs/train/act_cpu_v3_scripted50/`
  (checkpoint 001000), untouched. (The tail-of-log projection watcher was useless:
  the log flushes only at newlines, so its step readings lag; use INFO timestamps.)
- 15:07 Giri chose option (c): restart run 3 from scratch with the cap raised to 50 min
  for this run only, laptop in normal use. The 12:17 partial run was moved aside, not
  deleted: `outputs/train/act_cpu_v3_scripted50_aborted_1217{,.log}`.
- 15:07:53 run 3 restarted (same config, same locks, `flock -o`, ≥ 5 GB free). Projections
  from INFO timestamps: 45.7 min at step 1050 (15:14), then the desktop load rose (Giri's
  `piper_teleop` ROS nodes + RViz, ~140 % CPU; load average 7.4). Recent-rate projection:
  52.0 min at step 2650 (15:24:30, first OVER-CAP), 53.7 at 3100, 55.7 at 3500, 54.8 at
  4050 (last-1000-step rate 0.40–0.45 s/step).
- **15:34 run 3 stopped at step ~4065 of 8000 (26.0 min), projected ~55 min > 50.**
  I noticed the first over-cap flag ~10 min late (15:24:30 → 15:34): I was in a blocking
  wait loop instead of acting on the monitor events. Checkpoints 001000–004000 are kept
  in `outputs/train/act_cpu_v3_scripted50/` (loss 0.069 at step 4050). Asked Giri how to
  proceed; nothing evaluated.
- 17:28 Giri: laptop quiet apart from a browser; run 3 fresh from scratch (option B),
  cap 50 min, on over-cap stop and resume from the latest checkpoint (report segments).
  The 15:07 attempt was moved aside with its checkpoints:
  `outputs/train/act_cpu_v3_scripted50_aborted_1534{,.log}` (checkpoints 1000–4000).
- 17:28:54 re-probe (150 steps, `outputs/probe_v3b`): 0.28–0.32 s/step, startup ~55 s
  (cold cache) → ~42 min projected.
- 17:30:54 run 3 attempt 3 started. A guard ran as its own background job
  (`/tmp/r4/guard_v3.sh`, log `/tmp/r4/guard_v3.log`). Every 30 s it projected from the
  INFO timestamps (elapsed + remaining × last-1000-step rate). It would have stopped the
  run after the next checkpoint if the projection exceeded 50 min, independent of my
  waits. Projections rose from 41.3 min (step 600) to 45.3 min (step 7550) as the rate
  drifted from 0.308 to 0.348 s/step. The CPU reported 2.4 GHz (base clock, no turbo),
  consistent with a power/thermal limit; the same drift appears in every long run.
- **18:16:15 run 3 finished in one segment: 2719.1 s = 45.3 min** (inside the 50-min
  cap, 0.3 min over the original 45). Final loss 0.063 (L1 only), epoch 11.61. The
  resume fallback was not needed.
- 18:16–18:22 post-run (`/tmp/r4/run_post_v3.sh`, heavy+build `flock -o`):
  `sdp eval … --seed-start 200` → nominal **0/100** (87 never_reached, 13 collision),
  shifted **0/100** (87 never_reached, 13 collision; different seeds from nominal's 13,
  so the matching counts are a coincidence), Wilson 95 % 0–4 % each. Policy compute: mean
  2.4 ms per step, re-plan max 37–52 ms. Open loop: first-step d_x/d_y/d_yaw std
  0.0000/0.0002/0.0000, image swap ≤ 0.002, black images up to 2.0. Training seeds 0–9:
  0/10 (9 never_reached, 1 collision). GIF regenerated with run 3 (expert reference vs
  ACT run 3 on seed 10200, 0.59 MB).
- Conclusion change: with 2.7× the steps the policy is just as blind to the cube, so
  "underfitting under the spec'd budget" was too narrow. It is a proprioceptive-state
  shortcut that more training at this scale does not remove. README "Where it fails"
  and "What I'd do next" are updated accordingly.
- Release tarballs now include `act_cpu_v3_scripted50.tar.gz` (62 MB).

## Session 3 — 29 Oct 2026 evening: ACT run 4 (proprioception withheld)

- Giri approved run 4: run 2's config with the proprioceptive state withheld from the
  policy; same locks, memory wait, re-probe, 50-min cap with resume-on-over-cap; fresh
  test seeds 10300–10399.
- How: LeRobot fills `policy.input_features` from the dataset only when the config
  leaves it empty (`lerobot/policies/factory.py`, `if not cfg.input_features`). draccus
  parses `--policy.input_features=<JSON>` (checked: `robot_state_feature` → None, both
  image features present), and ACT accepts image-only inputs (`validate_features`).
  `act_cpu_v4.yaml` = `act_cpu_v2.yaml` + that one key; `train.py` passes
  `input_features` as one JSON argument (`JSON_LEAVES`). The dataset is unchanged.
- **Criterion, fixed before any run-4 result:** the diagnosis counts as fixed only if,
  in `scripts/diagnose_openloop.py`, swapping images moves the first-step d_x or d_yaw
  by ≥ 0.1, **and** the first-step d_x or d_yaw std across the 20 training episodes is
  ≥ 0.1 (data: 0.38 and 0.55). Success rate is reported separately.
- 19:47 probe (150 steps, `outputs/probe_v4`): 0.26–0.28 s/step → ~15 min; probe
  checkpoint `config.json` input_features = [front, wrist] only, use_vae False,
  backbone lr 1e-4, chunk 20, dim 256.
- 19:48:54–20:06:05 run 4, one segment, **1030.5 s = 17.2 min** (guard projections
  15.6–18.3 min; cap 50). Final L1 0.073 (run 2: 0.082).
- 20:06–20:12 eval on test 3 seeds 10300–10399: nominal **0/100** (100 never_reached),
  shifted **0/100** (99 never_reached, 1 collision), Wilson 95 % 0–4 %. Policy compute
  mean 2.4–3.6 ms per step, max ~90 ms.
- Diagnostics: first-step d_x/d_y/d_yaw std 0.0000/0.0002/0.0000; image swap change
  d_x 0.0000, d_yaw 0.0001 (max over all dims 0.0055, gripper); black images change z
  0.68, y 0.31, gripper 1.74, x 0.000, yaw 0.000. Episode-0 open-loop error z 0.071,
  gripper 0.010 (run 2: 0.083, 0.012). Training seeds 0/10 (10 never_reached).
- **Pre-registered criterion: not met.** The image-only policy reads the arm's progress
  from the images, but its x/yaw (the cube-dependent dimensions) never respond to the
  images. Revised diagnosis: the state shortcut was not the whole story; the vision
  path at 96 px (3×3 ResNet-18 feature map, cube 6–10 px, approach ≈ 9 % of targets)
  does not learn the cube's position within 3000–8000 CPU steps.
- Run 4 does not beat run 3 (both 0/100), so the GIF was not regenerated (Giri's rule).

## Current state and open questions (29 Oct 2026, 23:00 IST)

Supersedes the session-1 list above (its Q1 led to runs 3 and 4, Q11 is resolved).

- Four ACT CPU runs, all 0/100 nominal and shifted on their own test seeds (Wilson
  95 % 0–4 %): run 1 (spec'd config) and run 2 (backbone lr 1e-4, CVAE off) on
  10100–10199; run 3 (run 2 + 8000 steps, 45.3 min, cap raised to 50) on 10200–10299;
  run 4 (run 2 with the state withheld, 17.2 min) on 10300–10399.
- Diagnosis: no run uses the images to locate the cube. With the state, the networks
  fit the state-predictable phases. Without it (run 4), they read the arm's progress
  from the images but the cube-dependent x/yaw actions never respond to them. Likely
  cause: a 3×3 ResNet-18 feature map at 96 px for a 6–10 px cube whose approach is ~9 %
  of the targets.
- Release tarballs (all four checkpoints + dataset) are packed in `outputs/release/`;
  `scripts/download_release_assets.sh` fetches all four.

Open questions:

1. Next experiment (MORNING #8). Recommendation: run 4's config at 128 px (4×4 feature
   maps; re-record the 50 demos at 128 px). Estimated training time ≈ 31 min (run 4's
   17.2 min × (128/96)² ≈ 1.78), ≈ 40 min end to end. Confirm with a probe; same
   image-swap pass criterion.
2. Release upload and repo topics (commands in MORNING #1 and #4).
3. Branch protection, the bundled commit d8bde37, the hardware-transfer draft, the
   teleop session, the GIF substitution: unchanged from the session-1 list (Q3, Q5–Q7,
   Q10).
