# MORNING — sim-demo-to-policy

## STATUS (04:15 IST, 30 Oct)

**ACT works with absolute TCP-target actions.** Official test, seeds 10700–10799,
pre-registered in NOTES.md (commit 8060419) before any test eval, the same seeds for
both policies:

| Policy | Nominal | Shifted |
|---|---|---|
| run 8 @ 20k (headline: run 7 + 200 demos, half DART, 20000 steps) | 63/100 (Wilson 53–72 %) | 16/100 (10–24 %) |
| run 7 @ 6k (run 2 with absolute targets, 6000 steps) | 57/100 (47–66 %) | 19/100 (13–28 %) |
| runs 1–6 (per-step delta actions) | 0/100 each | 0/100 each |

- Nominal failures are mostly grasping (run 8: 30 grasp_missed, 5 dropped). Shifted
  failures are mostly never_reached (69 for each policy).
- The run 8 vs run 7 difference is not resolved: on the paired seeds, 23 succeed only
  with run 8 and 17 only with run 7.
- Repro check: the repo's `sdp eval` reproduces the experiment's run 7 dev result
  exactly (31/50, identical on all 50 seeds; `results/repro/`).
- Credit for the fix: `experiments/` (retrospective, integrity checks, privileged-state
  comparison), now committed.

Commits this turn: 2cfecdd (fast-forward of `feat/abs-actions`; branch and worktree
deleted), 8060419 (pre-registration + repro), d048604 (`experiments/`), and the results
and docs commits pushed with this status (see `git log`). CI: see the latest run on
<https://github.com/giridhar-shenoy/sim-demo-to-policy/actions>.

After smallbc finishes (~08:00), when the CPU is free:

1. `scripts/package_release_abs.sh`: packs run 7 @ 6000 and run 8 @ 20000 with their
   datasets into `outputs/release/` (about 150 MB of gzip).
2. Upload everything:
   `gh release create v0.1.0 outputs/release/*.tar.gz --title "sim-demo-to-policy v0.1.0"`.
   Final asset list: pickplace_scripted_50, pickplace_scripted_50_128px,
   pickplace_abs_50, pickplace_abs_dart_200, act_cpu_scripted50,
   act_cpu_v2_scripted50, act_cpu_v3_scripted50, act_cpu_v4_scripted50,
   act_cpu_v5_scripted50_128px, act_cpu_v6_scripted50, act_cpu_v7_abs and
   act_cpu_v8_abs_dart (all `.tar.gz`).
3. Re-render the README GIF with run 8:
   `python scripts/record_gifs.py --checkpoint experiments/privileged/run8/train/act_cpu_v8_abs_dart/checkpoints/020000 --csv results/eval_act_cpu_v8_abs_dart_s10700.csv --policy-label "ACT run 8"`.
   It currently shows run 3.

## WAITING FOR GIRI

1. **Repo topics.** Proposed: `mujoco, lerobot, imitation-learning, robot-learning,
   behavior-cloning, franka-panda, pick-and-place, act-policy`. Why: discoverability.
   `gh repo edit --add-topic …` was denied by the permission layer, so nothing was set.
   Recommendation: run
   `gh repo edit giridhar-shenoy/sim-demo-to-policy --add-topic mujoco,lerobot,imitation-learning,robot-learning,behavior-cloning,franka-panda,pick-and-place,act-policy`.
   Blocks: nothing.
2. **Branch protection on `main`** (require the `ci / build-test` check). Not enabled,
   to match R1–R3 where you are deciding this. Recommendation: enable it now that v0.1
   is in. Blocks: nothing.
3. **Commit d8bde37 bundles three changes** (training wrapper; eval adapter, report,
   GIF script; docs) under the training-wrapper message, because of a staging mistake
   right after the OOM restart. Splitting it would be a history rewrite.
   Recommendation: leave it. Blocks: nothing.
4. **Upload the release assets** after smallbc: run `scripts/package_release_abs.sh`,
   then `gh release create v0.1.0 outputs/release/*.tar.gz` (asset list in STATUS).
   Blocks: a download-only one-command demo.
5. **Look at the dataset in the LeRobot visualiser** (`lerobot-dataset-viz`, needs a
   GUI and the `viz` extra). Blocks: nothing.
6. **Edit `docs/hardware_transfer.md`** (marked DRAFT, first person, no employer names).
7. **Keyboard teleop demos (SHOULD)**: skipped; needs you at the keyboard.
8. **Next ACT step.** Grasp failures dominate nominal; the shifted box is mostly
   never_reached. Candidates: chunk-relative actions, more DART demos near the grasp,
   a wider demo pose box, several training seeds. Seeds 10800–10899 are reserved for
   the smallbc eval, 11000+ for a final check.

The aborted run-3 attempts are kept, not reported: `outputs/train/act_cpu_v3_scripted50_aborted_1217*`
and `…_aborted_1534*` (checkpoints 1000–4000). Delete them when you no longer need them.

## DECISIONS MADE OVERNIGHT (review)

- **Action space = DLS IK, not mocap** (the prompt's "ask me before"; overnight rules
  let me decide). A 6-DoF delta TCP pose is integrated into a target pose; damped
  least-squares IK (4 iterations, λ = 0.05, posture bias to home) turns it into joint
  targets for the Menagerie Panda's own position actuators, with gravity compensation
  on the arm bodies (the real Panda's controller does this). Mocap would teleport the
  hand and make the joint-state observation meaningless. Reversible: the action
  interface is one method (`MujocoArmEnv._integrate_target` + `_solve_ik`).
- **Python 3.12 via uv** for the repo venv (CI image is python:3.12-slim; lerobot 0.6.1
  also caps torch at 2.11).
- **ACT learning rate 1e-4 (backbone 1e-5) for run 1**, not LeRobot's 1e-5: the default
  is tuned for ~100k steps. No ablation was run.
- **A second ACT training run (run 2)**, inside the same ≤ 45 min per-run budget. Run 1
  scored 0/100 on eval seeds 10000–10099. A training-set open-loop check showed it
  ignored the cube's position: its first-step x/y/yaw std was 0.00 across 20 training
  episodes. Run 2 changes the backbone lr (1e-5 → 1e-4) and turns the CVAE off; dim,
  layers, chunk, image size, batch and steps are unchanged. I read "the ACT training
  run (45 min or less)" as approving training runs within that per-run budget. If you
  meant exactly one run, run 2 is the one to discard.
- **Dev/test seed protocol.** Because run 1's eval informed run 2, seeds 10000–10099
  are labelled "dev" and both runs are scored once on 10100–10199 ("test"). The
  headline uses test seeds. Both runs' rows are published.
- **One dataloader worker** (was 2) after the 03:32 OOM, per the new machine rules.
