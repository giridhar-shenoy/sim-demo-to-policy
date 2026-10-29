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
- README GIF: scripted expert (reference) beside an ACT failure on the same seed.
- CI: 50-step training smoke and 3-episode eval smoke (37 tests, ~40 s).

### Fixed

- Training CSV step column (lerobot prints `step:2K`); Wilson bounds exactly 0/1 at
  0/n and n/n; ACT adapter warm-up so latency excludes torch initialisation.
