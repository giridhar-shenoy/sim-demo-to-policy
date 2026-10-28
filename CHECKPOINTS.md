# CHECKPOINTS — sim-demo-to-policy

Overnight mode (29 Oct 2026): each checkpoint is logged here, committed, and work continues.

## Plumbing checkpoint — 29 Oct 2026 03:10 IST

Wall-clock so far: 28 min (session started 02:42 IST).

`git status` at the time of the report: clean apart from the uncommitted env work
in progress (`sdp/envs/`, `sdp/expert/`).

| Item | Status | Evidence |
|---|---|---|
| `docker build .` succeeds locally | **N/A on this laptop** | Docker is not installed (MACHINE.md; session addendum forbids installing it). The image is built by CI instead: step "Build image" in run 40772172057 succeeded. |
| CI green on the skeleton | **PASS** | <https://github.com/giridhar-shenoy/sim-demo-to-policy/actions/runs/40772172057> — `completed success … 4m52s`. Log: `MUJOCO_GL=egl: ok`, `MUJOCO_GL=osmesa: ok`, `All checks passed!`, `2 passed in 0.01s`. |
| `pre-commit run --all-files` passes | **PASS** | end-of-file, trailing-whitespace, large-files, ruff check, ruff format, yamllint, markdownlint-cli2 all `Passed`. |
| LICENSE has my name; required files exist | **PASS** | `grep -c "Giridhar J Shenoy" LICENSE` → 1; `ls` lists THIRD_PARTY.md NOTES.md CHANGELOG.md docs/architecture.md docs/results.md scripts/regenerate_all.sh. |
| Conventional commits, none before today | **PASS** | `git log --date=iso`: six commits, all `2026-10-29 02:57:46 +0530`, prefixes chore/ci/docs. |

Not done (recorded in MORNING.md → WAITING FOR GIRI): repo topics (`gh repo edit
--add-topic` was denied by the permission layer), branch protection on `main`.

Does Sunday 1 Nov hold? **Yes** — the skeleton took 28 min; the MUST list is on track.

## CHECKPOINT 1 — env throughput and scripted-expert success — 29 Oct 2026 03:18 IST

Wall-clock so far: 36 min. `git status`: clean after commit 80dfb12.

Scope note: the sprint override cuts PandaPegInsert and So101PickPlace, so only
PandaPickPlace-v0 exists.

**Throughput (≥ 200 steps/s headless): PASS.**

```text
$ SDP_MENAGERIE_DIR=~/work/menagerie MUJOCO_GL=egl python scripts/bench_env_speed.py
{
  "env": "PandaPickPlace-v0",
  "steps": 2000,
  "steps_per_s_with_images": 448.5,
  "steps_per_s_without_images": 1096.9,
  "mujoco_gl": "egl",
  "load_avg_1min_at_end": 1.66
}
```

One control step = IK + 25 physics substeps + both 96×96 cameras. The first
measurement was 26 steps/s; profiling showed the 4096 px shadow map and then the
Panda's 400k-vertex visual meshes dominated. Fixes: shadows off, and the observation
cameras draw the collision meshes (3.3k vertices) — see commit 9a315fa.

**Scripted expert on 100 seeds: 100/100 (Wilson 95 % 96–100 %). PASS (≥ 90 %).**

```text
$ python scripts/eval_expert.py
expert_demo_seeds  nominal  100/100 (100 %, Wilson 95 % 96-100 %), mean length 109.9 steps, labels {'success': 100, ...all others 0}
expert_eval_seeds  nominal  100/100 (100 %, Wilson 95 % 96-100 %), mean length 110.6 steps
expert_eval_seeds  shifted  100/100 (100 %, Wilson 95 % 96-100 %), mean length 117.2 steps
```

Per-episode rows: `results/expert_pickplace.csv`; summary `results/expert_pickplace.json`.
The headline number is seeds 0–99 (nominal), the population the demos come from.

**Tests:** `pytest -q` → `29 passed in 2.42s` (env spaces, timing, seed determinism
incl. images, distribution bounds, action clipping, workspace clamp, success after
the 0.5 s settle on a scripted trajectory, no success without release, render;
taxonomy priority; seed split; Wilson).

What is left: recorder + 50 demos (CP2), training wrapper + 45-min run, eval harness
for ACT (CP3), GIF, README/docs, CI smokes (CP4).

Does Sunday 1 Nov hold? **Yes.**

## CHECKPOINT 2 — dataset loads — 29 Oct 2026 03:21 IST

Wall-clock so far: 39 min. `git status`: clean after the recorder commits.

```text
$ flock ~/work/.locks/heavy.lock flock ~/work/.locks/build.lock scripts/record_demos.sh
episode 50/50 seed 49: success (109 frames, 47 s elapsed)
wrote 50 episodes to datasets/pickplace_scripted_50 (50 successful)
{
  "dataset": "pickplace_scripted_50",
  "codebase_version": "v3.0",
  "episodes": 50,
  "frames": 5513,
  "fps": 20,
  "successful_episodes": 50,
  "mean_episode_length_steps": 110.26,
  "min_max_episode_length_steps": [104, 117],
  "seeds": [0, 49],
  ...
}
real 0m52.992s
```

| Dataset | Episodes | Frames | Mean length | Successful | Seeds |
|---|---|---|---|---|---|
| pickplace_scripted_50 | 50 | 5513 | 110.3 steps (5.5 s) | 50 | 0–49 |

- Loads with `LeRobotDataset("local/pickplace_scripted_50", root=...)` (the summary
  script does exactly that); `meta/info.json` → `codebase_version: "v3.0"`.
- Recorder round-trip test (`tests/test_recorder.py`, 2 tiny episodes): 3 passed.
- **Not done: the LeRobot visualiser** (`lerobot-dataset-viz`) needs a GUI (rerun
  viewer) — listed under WAITING FOR GIRI.
- Only the scripted dataset exists: teleop datasets (50 + 50 + 30) are cut by the
  sprint override (teleop needs Giri at the keyboard; peg/So-101 envs not built).

What is left: training wrapper + ACT CPU run (≤ 45 min), ACT eval (CP3), GIF, docs, CI smokes.

Does Sunday 1 Nov hold? **Yes.**
