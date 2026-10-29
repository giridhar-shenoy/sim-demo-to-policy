# MORNING — sim-demo-to-policy

## STATUS (06:05 IST, 29 Oct)

**v0.1 MUST list: done. CI: green. ACT success rate: 0/100 on every condition. It is
diagnosed, and the fix needs your decision (#9 below).**

Done (commits on `main`, all pushed):

| What | Commit(s) |
|---|---|
| Skeleton (license, Dockerfile, devcontainer, CI, pre-commit, docs stubs) | 6cf6118 … ba38e80 |
| PandaPickPlace-v0 env (DLS IK, 500/20 Hz, two 96 px cameras, 448 steps/s) | 9a315fa |
| Scripted expert, 100/100 on demo + eval seeds | bdd4b6b, 80dfb12 |
| Eval harness, taxonomy, seed split, Wilson | 479624b, 7d1fbb1, 75f3295 |
| LeRobot v3.0 recorder + 50 scripted demos | 713a7aa, 4e9bc9e |
| Training wrapper + ACT CPU config (bundled commit, see #3) | d8bde37 |
| ACT run 1 (21.0 min) / run 2 (17.1 min) + results: 0/100 everywhere | 3c8851e, 6796baf, 06794f7 |
| Diagnostics (open loop, image swap, training seeds) | 57535a3, 768d54e, 5f4ed0f |
| GIF (expert reference vs ACT failure), README, docs | fbd15c4, 75e8500 |
| CHECKPOINT 4 | 281e1f3 |

CI: latest run green, 37 passed in 40.74 s, job ~3 min:
<https://github.com/giridhar-shenoy/sim-demo-to-policy/actions/runs/40794185957>

Blocked (decisions below): a non-zero ACT number (#9), release upload (#4), topics and
branch protection (#1, #2), hardware-transfer draft (#7).

Next three steps:

1. Decide #9. With (a), run `scripts/train_cpu.sh … act_cpu_v3.yaml` at 8000 steps
   (~45 min), then `sdp eval … --seed-start 200` on fresh test seeds 10200–10299, and
   `sdp report`.
2. Upload the release (`gh release create v0.1.0 outputs/release/*.tar.gz`), then add
   the download-only run path to the README.
3. Edit `docs/hardware_transfer.md` and the README's hardware paragraph in your voice;
   then make R1/R2 public so the README links resolve.

## WAITING FOR GIRI

1. **Repo topics.** Proposed: `mujoco, lerobot, imitation-learning, robot-learning,
   behavior-cloning, franka-panda, pick-and-place, act-policy`. Why: discoverability.
   `gh repo edit --add-topic …` was denied by the permission layer, so nothing was set.
   Recommendation: run
   `gh repo edit giridhar-shenoy/sim-demo-to-policy --add-topic mujoco,lerobot,imitation-learning,robot-learning,behavior-cloning,franka-panda,pick-and-place,act-policy`.
   Blocks: nothing.
2. **Branch protection on `main`** (require the `ci / build-test` check). Not enabled,
   to match R1–R3 where you are deciding this; enabling it tonight would also block my
   direct pushes. Recommendation: enable it once v0.1 is merged. Blocks: nothing.

3. **Commit d8bde37 bundles three changes** (training wrapper; eval adapter, report,
   GIF script; docs) under the training-wrapper message, because of a staging mistake
   right after the OOM restart. Splitting it would be a history rewrite (not allowed
   tonight). Recommendation: leave it. If you want it split, run
   `git rebase -i d8bde37~1` before anyone else pulls. Blocks: nothing.

4. **Upload the v0.1 release assets** (over the 50 MB commit limit). They are already
   packed in `outputs/release/`: act_cpu_scripted50.tar.gz 89 MB,
   act_cpu_v2_scripted50.tar.gz 62 MB, pickplace_scripted_50.tar.gz 5.2 MB. Command:
   `gh release create v0.1.0 outputs/release/*.tar.gz` (then
   `scripts/download_release_assets.sh` works). Until then the README's
   run path retrains (~45 min for both runs). Blocks: a download-only one-command demo.
5. **Look at the dataset in the LeRobot visualiser** (`lerobot-dataset-viz`, needs a
   GUI and the `viz` extra). Not done overnight. Blocks: nothing.
6. **README links to R1 and R2 return 404 while those repos are private.** Blocks: nothing
   once they are public.
7. **Edit `docs/hardware_transfer.md`** (marked DRAFT, first person, no employer names).
8. **Keyboard teleop demos (SHOULD)**: skipped; needs you at the keyboard.

9. **ACT scores 0/100 on every condition (CHECKPOINT 3).** Both CPU runs learn the
   state-predictable phases but ignore where the cube is (diagnosis in CHECKPOINTS.md
   and README "Where it fails"). Options, in my order of preference:
   (a) approve one more CPU run that departs from the spec'd 3000 steps, e.g. 8000
   steps ≈ 45 min at run 2's speed (0.34 s/step), which is still inside the 45-min cap;
   (b) approve a run that withholds the proprioceptive state from the policy (keeps
   3000 steps; removes the shortcut, which is a dataset-feature change);
   (c) leave v0.1 at 0 % with the diagnosis and do the full-size run on Colab next week.
   Recommendation: (a) first, because it changes nothing but the step count. Blocks: a
   non-zero headline number.

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
