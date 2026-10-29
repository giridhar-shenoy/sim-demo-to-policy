# MORNING — sim-demo-to-policy

## STATUS (18:30 IST, 29 Oct)

**v0.1 MUST list: done. CI: green. ACT: three CPU runs, all 0/100.** Run 3 (your
option (a): 8000 steps; cap raised to 50 min for this run) took 45.3 min in one segment
and still ignores the cube's position, so training longer is not the fix at this scale.

| Run | Change from the spec'd CPU config | Steps | Wall time | Seeds | Nominal | Shifted |
|---|---|---|---|---|---|---|
| 1 | spec'd config; lr 1e-4, backbone 1e-5, CVAE on | 3000 | 21.0 min | 10000–10099 (dev), 10100–10199 | 0/100 (Wilson 0–4 %) | 0/100 (0–4 %) |
| 2 | run 1 + backbone lr 1e-4, CVAE off | 3000 | 17.1 min | 10100–10199 | 0/100 (0–4 %) | 0/100 (0–4 %) |
| 3 | run 2 + 8000 steps | 8000 | 45.3 min | 10200–10299 | 0/100 (0–4 %) | 0/100 (0–4 %) |

Done since this morning (all pushed): run 3 config and three-run reporting (91bac96),
run 3 results (eccfd43), docs for all three runs (048856f). Earlier commits are in the
table below.

| What | Commit(s) |
|---|---|
| Skeleton (license, Dockerfile, devcontainer, CI, pre-commit, docs stubs) | 6cf6118 … ba38e80 |
| PandaPickPlace-v0 env (DLS IK, 500/20 Hz, two 96 px cameras, 448 steps/s) | 9a315fa |
| Scripted expert, 100/100 on demo + eval seeds | bdd4b6b, 80dfb12 |
| Eval harness, taxonomy, seed split, Wilson | 479624b, 7d1fbb1, 75f3295 |
| LeRobot v3.0 recorder + 50 scripted demos | 713a7aa, 4e9bc9e |
| Training wrapper + ACT CPU config (bundled commit, see #3) | d8bde37 |
| ACT runs 1 and 2 + results | 3c8851e, 6796baf, 06794f7 |
| Diagnostics (open loop, image swap, training seeds) | 57535a3, 768d54e, 5f4ed0f |
| ACT run 3 + results + docs | 91bac96, eccfd43, 048856f |

CI: <https://github.com/giridhar-shenoy/sim-demo-to-policy/actions/runs/40877347214>
(commit 048856f).

Blocked (decisions below): release upload (#4); topics and branch protection (#1, #2);
hardware-transfer draft (#6); the next ACT experiment (#8).

Next three steps:

1. Decide #8. My recommendation is a run that withholds the proprioceptive state
   from the policy, with everything else as in run 2. Accept it only if the open-loop
   image-swap test moves the approach actions.
2. Upload the release (`gh release create v0.1.0 outputs/release/*.tar.gz`), then add
   the download-only run path to the README.
3. Edit `docs/hardware_transfer.md` and the README's hardware paragraph in your voice.

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
   `outputs/release/`: act_cpu_scripted50.tar.gz 89 MB, act_cpu_v2_scripted50.tar.gz
   62 MB, act_cpu_v3_scripted50.tar.gz 62 MB, pickplace_scripted_50.tar.gz 5.2 MB.
   Command: `gh release create v0.1.0 outputs/release/*.tar.gz` (then
   `scripts/download_release_assets.sh` works). Until then the README's run path
   retrains (~85 min for the three runs). Blocks: a download-only one-command demo.
5. **Look at the dataset in the LeRobot visualiser** (`lerobot-dataset-viz`, needs a
   GUI and the `viz` extra). Blocks: nothing.
6. **Edit `docs/hardware_transfer.md`** (marked DRAFT, first person, no employer names).
7. **Keyboard teleop demos (SHOULD)**: skipped; needs you at the keyboard.
8. **Next ACT experiment** (item 9 from the night is resolved: option (a) was run 3, 0/100).
   Options: (a) withhold or delay the proprioceptive state from the policy (keeps 3000
   steps; a policy-input change, so it needs your OK); (b) randomise the arm's start pose
   so the state stops encoding episode progress (an env change after the dataset exists,
   so it needs your OK and a re-record); (c) leave v0.1 at 0 % with the diagnosis and do
   the full-size Colab run next week. Recommendation: (a). Blocks: a non-zero headline.

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
