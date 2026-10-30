# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- README "Use the trained policies": what each v0.1.0 archive contains, and a
  verify-without-training recipe (curl `smallbc_abs1000_25k.tar.gz`, then `sdp eval` on
  test seeds 10800–10899). The download + extract + eval path was tested on 5 nominal
  episodes, which were identical to the official CSV.
- `scripts/fetch_release_asset.sh`: fetch and extract one release archive (curl, no gh
  auth).

### Fixed

- `scripts/regenerate_all.sh` failed on a fresh clone: it read ACT runs 7 and 8 from
  gitignored `experiments/privileged/` paths. When those are missing, it now fetches
  `act_cpu_v7_abs` and `act_cpu_v8_abs_dart` from the v0.1.0 release.
- Stale docs: the README's regenerate time (about 6 h, not ~3 h) and its note that runs
  7 and 8 were not yet released; the GIF name in `scripts/README.md`, `CHECKPOINTS.md`
  and `record_gifs.py`'s default `--out`.

## [0.1.0] - 2026-10-30

Released as [v0.1.0](https://github.com/giridhar-shenoy/sim-demo-to-policy/releases/tag/v0.1.0)
(tag on 7f766e0): 13 assets, 583262517 bytes. They are the eight ACT checkpoints, smallbc
@ 25k with its config, and the four LeRobot v3.0 datasets. smallbc's 5.8 GB dataset is not
attached; `python -m sdp.policies.smallbc.record` regenerates it from demo seeds 0–999.

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
- ACT run 3 (`act_cpu_v3.yaml`: run 2 with 8000 steps instead of the spec'd 3000; cap
  raised from 45 to 50 min for this run because the laptop was in use): 45.3 min, 0/100
  nominal and 0/100 shifted on fresh test seeds 10200–10299 (Wilson 95 % 0–4 %). The
  open-loop check shows it still ignores the cube's position.
- ACT run 4 (`act_cpu_v4.yaml`: run 2 with the proprioceptive state withheld, cameras
  only, via `policy.input_features`): 17.2 min, 0/100 nominal and 0/100 shifted on fresh
  test seeds 10300–10399 (Wilson 95 % 0–4 %). It fails the pre-registered image-swap
  criterion: its approach actions do not depend on the images.
- `train.py` passes dict-valued `input_features` to lerobot-train as one JSON argument.
- ACT run 5 (`act_cpu_v5.yaml`: run 4 with both cameras at 128 px; demos re-recorded at
  128 px into `pickplace_scripted_50_128px`, same seeds): recording 55 s, training 21.5
  min, eval 5.0 min; 0/100 nominal and 0/100 shifted on fresh test seeds 10400–10499
  (Wilson 95 % 0–4 %). Fails the same pre-registered criterion.
- ACT run 6 (`act_cpu_v6.yaml`: run 2 with lr 1e-5 and backbone lr 1e-5, LeRobot's
  defaults; tests the hypothesis that lr 1e-4 caused the collapse): 16.3 min training,
  4.5 min eval; 0/100 nominal and 0/100 shifted on fresh test seeds 10600–10699 (Wilson
  95 % 0–4 %). Image-swap sensitivity in x/yaw 0.0031/0.0029 → 0.0009/0.0012 from step
  1000 to 3000: the pre-registered criterion is not met.
- `diagnose_openloop.py` names intermediate checkpoints `<run>_step<N>`; the diagnostics
  table gains d_x/d_yaw swap columns and a criterion column.
- `--image-size` for `sdp record`, `record_demos.sh` and `eval_expert.py`; eval,
  diagnostics and the GIF take the image size from the checkpoint; recording and eval
  wall times are saved and tabulated.
- README GIF: scripted expert (reference) beside an ACT failure on the same seed.
- CI: 50-step training smoke and 3-episode eval smoke (37 tests, ~40 s).

- Absolute TCP-target actions (`sdp/envs/abs_target.py`): `abs_target`, `abs_to_delta`
  and `AbsoluteTargetWrapper`, ported from `experiments/privileged/priv_common.py`;
  `sdp record --action-mode abs_target` (5-D named action); the ACT eval adapter infers
  the action mode from the checkpoint's action shape and wraps the env. Motivated by a
  privileged-state experiment: delta actions 0/50, absolute targets 50/50.

- **ACT succeeds with absolute TCP-target actions.** On test seeds 10700–10799,
  pre-registered: run 8 (run 7 + 200 demos, half with DART noise, 20000 steps) 63/100
  nominal (Wilson 95 % 53–72 %) and 16/100 shifted (10–24 %); run 7 (run 2 with
  absolute targets, 6000 steps) 57/100 (47–66 %) and 19/100 (13–28 %) on the same seeds.
  Runs 1–6 with per-step deltas stay at 0/100.
- `experiments/`: the retrospective, the integrity checks, the privileged-state
  comparison (delta 0/50 vs absolute 50/50) and the run 7/8 code. They found the fix.
- `results/repro/`: the repo's `sdp eval` reproduces the experiment's run 7 dev result
  on all 50 seeds.
- Report: paired-comparison table; the headline uses the policy fixed before testing.
- `scripts/package_release_abs.sh`; `package_release_assets.sh` takes a checkpoint step.

- **smallbc is the headline policy.** It is a from-scratch spatial-softmax BC
  (573,650 params, 985 demos, batch 64, the 25k checkpoint of a 50k-step run), chosen on
  dev before testing. Official test, seeds 10800–10899: 100/100 nominal (Wilson 95 %
  96–100 %) and 72/100 shifted (63–80 %). On ACT's test seeds 10700–10799: 97/100 and
  70/100. The ACT runs are reported alongside; the gap is a recipe comparison.
- `sdp/policies/smallbc/` (ported from `experiments/smallbc/`, kept as the record):
  `python -m sdp.policies.smallbc.record` / `.train`; `sdp eval` dispatches `.pt`
  checkpoints to it and reproduces the experiment's dev episodes exactly.
- Paired McNemar table in the report; `scripts/diagnose_collisions.py`;
  `scripts/smallbc_results_from_run.py`; mypy on `sdp/policies` in CI.
- README GIF renamed `docs/assets/rollouts.gif`: the expert, smallbc nominal and
  smallbc shifted.

### Fixed

- Training CSV step column (lerobot prints `step:2K`); Wilson bounds exactly 0/1 at
  0/n and n/n; ACT adapter warm-up so latency excludes torch initialisation.

[Unreleased]: https://github.com/giridhar-shenoy/sim-demo-to-policy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/giridhar-shenoy/sim-demo-to-policy/releases/tag/v0.1.0
