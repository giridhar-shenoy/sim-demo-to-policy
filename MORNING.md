# MORNING — sim-demo-to-policy

## STATUS (23:45 IST, 29 Oct)

**v0.1 MUST list: done. CI: green. ACT: five CPU runs, all 0/100.** Run 5 (your
approved experiment: run 4 at 128 px) recorded in 55 s, trained in 21.5 min and
evaluated in 5.0 min. It fails the criterion fixed before training, exactly like run 4:
the policy tracks the arm from the images but never uses them to find the cube.
Training length, the state shortcut and resolution are now each ruled out at this
scale. My recommendation for Sunday is to **ship v0.1 with the diagnosis**.

| Run | Change from the spec'd CPU config | Steps | Training | Test seeds | Nominal | Shifted |
|---|---|---|---|---|---|---|
| 1 | spec'd config; lr 1e-4, backbone 1e-5, CVAE on | 3000 | 21.0 min | 10100–10199 (dev 10000–10099) | 0/100 (Wilson 0–4 %) | 0/100 (0–4 %) |
| 2 | run 1 + backbone lr 1e-4, CVAE off | 3000 | 17.1 min | 10100–10199 | 0/100 (0–4 %) | 0/100 (0–4 %) |
| 3 | run 2 + 8000 steps (cap raised to 50 min) | 8000 | 45.3 min | 10200–10299 | 0/100 (0–4 %) | 0/100 (0–4 %) |
| 4 | run 2 + proprioceptive state withheld | 3000 | 17.2 min | 10300–10399 | 0/100 (0–4 %) | 0/100 (0–4 %) |
| 5 | run 4 + 128 px cameras (demos re-recorded) | 3000 | 21.5 min | 10400–10499 | 0/100 (0–4 %) | 0/100 (0–4 %) |

Commits: run 3 (91bac96, eccfd43, 048856f); run 4 (05965f0, b574bb3, 043c036,
7839069); run 5 in the commits pushed with this status (see `git log`). Earlier:

| What | Commit(s) |
|---|---|
| Skeleton (license, Dockerfile, devcontainer, CI, pre-commit, docs stubs) | 6cf6118 … ba38e80 |
| PandaPickPlace-v0 env (DLS IK, 500/20 Hz, two cameras, 448 steps/s at 96 px) | 9a315fa |
| Scripted expert, 100/100 on demo + eval seeds (also at 128 px) | bdd4b6b, 80dfb12 |
| Eval harness, taxonomy, seed split, Wilson | 479624b, 7d1fbb1, 75f3295 |
| LeRobot v3.0 recorder + 50 scripted demos | 713a7aa, 4e9bc9e |
| Training wrapper + ACT CPU config (bundled commit, see #3) | d8bde37 |
| ACT runs 1 and 2 + results; diagnostics | 3c8851e, 6796baf, 06794f7, 57535a3, 768d54e, 5f4ed0f |

CI: see the latest run on <https://github.com/giridhar-shenoy/sim-demo-to-policy/actions>.

Blocked (decisions below): release upload (#4), topics and branch protection (#1, #2),
hardware-transfer draft (#6), what to do about ACT after Sunday (#8).

Next three steps:

1. Upload the release (`gh release create v0.1.0 outputs/release/*.tar.gz`: five
   checkpoints and both datasets).
2. Edit `docs/hardware_transfer.md` and the README's hardware paragraph in your voice.
3. After Sunday, #8: approach-frame weighting or a slower expert approach first, then
   an auxiliary cube-position loss, then a full-size GPU run.

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
4. **Upload the v0.1 release assets** (over the 50 MB commit limit). Packed in
   `outputs/release/`: act_cpu_scripted50 89 MB; act_cpu_v2/v3/v4_scripted50 and
   act_cpu_v5_scripted50_128px 62 MB each; pickplace_scripted_50 5.2 MB;
   pickplace_scripted_50_128px 6.6 MB. Command:
   `gh release create v0.1.0 outputs/release/*.tar.gz` (then
   `scripts/download_release_assets.sh` works). Until then the README's run path
   retrains (~2 h for the five runs). Blocks: a download-only one-command demo.
5. **Look at the dataset in the LeRobot visualiser** (`lerobot-dataset-viz`, needs a
   GUI and the `viz` extra). Blocks: nothing.
6. **Edit `docs/hardware_transfer.md`** (marked DRAFT, first person, no employer names).
7. **Keyboard teleop demos (SHOULD)**: skipped; needs you at the keyboard.
8. **ACT after Sunday.** Runs 3 (8000 steps), 4 (state withheld) and 5 (128 px) each
   ruled out one cause; all five runs are 0/100. What is left is the data and the
   objective: the approach is 11.9 % of the targets, and L1 regression settles on their
   mean. Options, in my order: (a) oversample or up-weight the approach frames (LeRobot
   0.6.1 has a `--sample_weighting` config I have not verified yet); (b) a slower,
   unsaturated expert approach (env/expert change plus re-record); (c) an auxiliary
   cube-position loss during training (code in this repo); (d) a full-size GPU run.
   Each is ~1 h end to end on the laptop, with no guarantee. Recommendation: ship v0.1
   with the diagnosis on Sunday and do (a) then (b) next week. Blocks: a non-zero
   headline.

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
