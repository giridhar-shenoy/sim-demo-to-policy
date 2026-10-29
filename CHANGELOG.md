# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Repository skeleton: MIT license, Dockerfile (python:3.12-slim), devcontainer,
  GitHub Actions CI, pre-commit, docs and scripts placeholders.
- `PandaPickPlace-v0` gymnasium env on the Menagerie Franka Panda: 500 Hz physics,
  20 Hz control, delta-TCP-pose actions through damped least-squares IK, front and wrist
  96×96 cameras, seedable nominal/shifted cube-pose distributions, a 0.5 s settled
  success predicate, and per-episode event flags.
- Scripted waypoint expert (100/100 on demo seeds 0–99 and on eval seeds 10000–10099
  under both pose distributions).
- LeRobot v3.0 recorder with a per-episode success/label sidecar, plus `sdp record`;
  50 scripted demonstrations (local dataset, summary in `results/`).
- `lerobot-train` wrapper with the ACT CPU config (`sdp/train/configs/act_cpu.yaml`).
- Eval harness with a disjoint seed split, failure taxonomy, Wilson 95 % intervals and
  per-episode policy timing; `sdp eval`, `sdp report` (generated doc tables) and the
  GIF script.
- Docs: architecture (Mermaid), results, tasks, hardware-transfer draft.
- Two ACT CPU training runs (21.0 and 17.1 min) and their evaluation: 0/100 nominal and
  0/100 shifted for both, on dev seeds 10000–10099 (run 1) and test seeds 10100–10199
  (both runs); combined results in `results/eval_act_pickplace.{csv,json}`.
- Diagnostics: open-loop check with image swap (`scripts/diagnose_openloop.py`) and
  training-seed rollouts (`scripts/diagnose_train_seeds.py`); both runs ignore the
  cube's position.
- ACT run 3 (`act_cpu_v3.yaml`: run 2 with 8000 steps instead of the spec'd 3000,
  approved by Giri; cap raised from 45 to 50 min for this run because the laptop was in
  use): 45.3 min, 0/100 nominal and 0/100 shifted on fresh test seeds 10200–10299
  (Wilson 95 % 0–4 %). The open-loop check shows it still ignores the cube's position.
- ACT run 4 (`act_cpu_v4.yaml`: run 2 with the proprioceptive state withheld, cameras
  only, via `policy.input_features`; approved by Giri): 17.2 min, 0/100 nominal and
  0/100 shifted on fresh test seeds 10300–10399 (Wilson 95 % 0–4 %). It fails the
  pre-registered image-swap criterion: its approach actions do not depend on the images.
- `train.py` passes dict-valued `input_features` to lerobot-train as one JSON argument.
- ACT run 5 (`act_cpu_v5.yaml`: run 4 with both cameras at 128 px; demos re-recorded
  at 128 px into `pickplace_scripted_50_128px`, same seeds; approved by Giri): recording
  55 s, training 21.5 min, eval 5.0 min; 0/100 nominal and 0/100 shifted on fresh test
  seeds 10400–10499 (Wilson 95 % 0–4 %). Fails the same pre-registered criterion.
- ACT run 6 (`act_cpu_v6.yaml`: run 2 with lr 1e-5 and backbone lr 1e-5, LeRobot's
  defaults; tests the hypothesis that lr 1e-4 caused the collapse; approved by Giri):
  16.3 min training, 4.5 min eval; 0/100 nominal and 0/100 shifted on fresh test seeds
  10600–10699 (Wilson 95 % 0–4 %). Image-swap sensitivity in x/yaw 0.0031/0.0029 →
  0.0009/0.0012 from step 1000 to 3000: the pre-registered criterion is not met.
- `diagnose_openloop.py` names intermediate checkpoints `<run>_step<N>`; the diagnostics
  table gains d_x/d_yaw swap columns and a criterion column.
- `--image-size` for `sdp record`, `record_demos.sh` and `eval_expert.py`; eval,
  diagnostics and the GIF take the image size from the checkpoint; recording and eval
  wall times are saved and tabulated.
- README GIF: scripted expert (reference) beside an ACT failure on the same seed.
- CI: 50-step training smoke and 3-episode eval smoke (37 tests, ~40 s).

### Fixed

- Training CSV step column (lerobot prints `step:2K`); Wilson bounds exactly 0/1 at
  0/n and n/n; ACT adapter warm-up so latency excludes torch initialisation.
