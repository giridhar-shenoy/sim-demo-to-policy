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
