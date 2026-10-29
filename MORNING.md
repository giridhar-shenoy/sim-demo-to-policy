# MORNING — sim-demo-to-policy

## STATUS (00:50 IST, 30 Oct)

**v0.1 MUST list: done. CI: green. ACT: six CPU runs, all 0/100.** Run 6 tested the
retrospective's leading hypothesis (`experiments/RETROSPECTIVE.md`): that the
transformer lr of 1e-4 in runs 1–5 caused the collapse. With LeRobot's defaults (lr
1e-5, backbone 1e-5, clip 10) it scores 0/100 nominal and 0/100 shifted. The
pre-registered criterion is not met at any checkpoint: image-swap sensitivity in x/yaw
is 0.0031/0.0029 (1k) → 0.0018/0.0018 (2k) → 0.0009/0.0012 (3k), so 10–30× runs 1–5
but shrinking. **The learning rate is not the sole cause.**

| Run | Change from the spec'd CPU config | Steps | Training | Test seeds | Nominal | Shifted |
|---|---|---|---|---|---|---|
| 1 | spec'd config; lr 1e-4, backbone 1e-5, CVAE on | 3000 | 21.0 min | 10100–10199 (dev 10000–10099) | 0/100 (Wilson 0–4 %) | 0/100 (0–4 %) |
| 2 | run 1 + backbone lr 1e-4, CVAE off | 3000 | 17.1 min | 10100–10199 | 0/100 (0–4 %) | 0/100 (0–4 %) |
| 3 | run 2 + 8000 steps (cap raised to 50 min) | 8000 | 45.3 min | 10200–10299 | 0/100 (0–4 %) | 0/100 (0–4 %) |
| 4 | run 2 + proprioceptive state withheld | 3000 | 17.2 min | 10300–10399 | 0/100 (0–4 %) | 0/100 (0–4 %) |
| 5 | run 4 + 128 px cameras (demos re-recorded) | 3000 | 21.5 min | 10400–10499 | 0/100 (0–4 %) | 0/100 (0–4 %) |
| 6 | run 2 + lr 1e-5, backbone lr 1e-5 | 3000 | 16.3 min | 10600–10699 | 0/100 (0–4 %) | 0/100 (0–4 %) |

Run 6 changed the failure mix (nominal: 82 never_reached, 15 collision, 2 grasp_missed,
1 placed_outside, the first episode in any run that lifted the cube; shifted: 68
collision). The approach is still image-blind, so that lift is most likely luck.

Commits: run 6 in the commits pushed with this status (see `git log`); runs 3–5 in
91bac96 … 17bf863. Earlier:

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
`experiments/` is untracked on purpose (other agents' work) and is not in any commit.

Blocked (decisions below): the next ACT step (#8, Giri is sending it), release upload
(#4), topics and branch protection (#1, #2), hardware-transfer draft (#6).

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
   `outputs/release/`: act_cpu_scripted50 89 MB; act_cpu_v2/v3/v4/v6_scripted50 and
   act_cpu_v5_scripted50_128px ~62 MB each; pickplace_scripted_50 5.2 MB;
   pickplace_scripted_50_128px 6.6 MB. Command:
   `gh release create v0.1.0 outputs/release/*.tar.gz` (then
   `scripts/download_release_assets.sh` works). Until then the README's run path
   retrains (~2.5 h for the six runs). Blocks: a download-only one-command demo.
5. **Look at the dataset in the LeRobot visualiser** (`lerobot-dataset-viz`, needs a
   GUI and the `viz` extra). Blocks: nothing.
6. **Edit `docs/hardware_transfer.md`** (marked DRAFT, first person, no employer names).
7. **Keyboard teleop demos (SHOULD)**: skipped; needs you at the keyboard.
8. **Next ACT step: Giri is sending it** (after run 6). The retrospective's plan
   continues with a pipeline proof (privileged-state ACT, integrity checks), a small
   spatial-softmax BC, and an overnight ACT with the combined recipe. Seed ledger:
   10500–10599 for those experiments, 10700–10999 for later runs, 11000–11099 the final
   test (used once).

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
