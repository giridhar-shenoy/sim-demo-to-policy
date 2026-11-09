# ACT at 0/100: retrospective, research and plan (30 Oct 2026, ~00:30 IST)

Working notes for R4. Written by Claude Code from:
- a read of the repo;
- the five run checkpoints;
- two web-research passes;
- the integrity and privileged-state checks under `experiments/`.

## What we did

| Run | Change | Steps (epochs) | Result |
|---|---|---|---|
| 1 | spec'd CPU config: dim 256, 2/1 layers, chunk 20 / exec 10, **lr 1e-4**, backbone 1e-5, CVAE on | 3000 (4.4) | 0/100 |
| 2 | backbone lr 1e-4, CVAE off | 3000 | 0/100 |
| 3 | 8000 steps | 8000 (11.6) | 0/100 |
| 4 | joint state withheld (camera-only) | 3000 | 0/100 |
| 5 | 128 px | 3000 | 0/100 |

The symptom was the same in every run: identical first actions across episodes, an image-swap sensitivity of about 1e-4, and failure even on the policy's own training seeds. Checked in the checkpoints' `train_config.json`: all five runs used transformer **lr 1e-4**, grad clip 10, weight decay 1e-4, no LR schedule and no image augmentation.

## What went wrong in the process

1. **No end-to-end proof that the pipeline can succeed.** Five runs went into tuning a system that had never scored above 0. A privileged-state baseline (cube pose as input), an overfit test on a few demos, and a zero-image baseline were never run, and these are the standard first checks (Karpathy's recipe; robomimic's dataset replay). Changing one factor at a time can't find a fix when two or more factors are wrong at once.
2. **The learning rate was changed early and never questioned.** The spec's config moved the transformer from LeRobot's default 1e-5 to 1e-4 "because 3000 steps is short". Every later run inherited it.
3. **The conclusions were stronger than the evidence.** "Training length ruled out" came from 3000 vs 8000 steps at the same lr. Run 4 changed the inputs and run 5 the resolution, both on top of the same suspect optimiser setting.
4. **The data recipe is hard on BC.** The expert is deterministic, so there is no recovery data. The approach phase is short: 13 frames, a third of them saturated. Actions are chained per-step deltas, so only about 12 % of the targets depend on the cube's position. The task spread is also wide (15×15 cm, ±45° yaw) for 50 demos: ManiSkill PickCube from RGB gets ACT 28 % with 100 demos.
5. **The time budget drove the design.** A 45-min CPU cap per run, daytime laptop use, and a Sunday deadline pushed toward short one-change runs instead of a few decisive ones.

What went right: the pass criteria were written before training, every run had fresh test seeds, the image-swap and black-image diagnostics were sound, and the README stayed accurate.

## What the research says

All sources are in the agents' reports, summarised here.

- **Our exact symptom is documented.** [physical-ai-studio #998](https://github.com/open-edge-platform/physical-ai-studio/issues/998): 35 episodes, 4.2k frames, batch 8. ACT collapsed to one constant chunk (outputs equal to 4e-6), with the transformer and ResNet at lr 1e-4. Fixed by LeRobot's defaults (lr 1e-5 for both, clip 10). Outputs then responded to images within about 1.4k steps.
- **Training length:** the original ACT reached about 90 % on 50 scripted demos with roughly 10k steps at batch 8 and lr 1e-5. LeRobot's references run 100k steps (about 40 epochs). Its model card says success "jumps up towards the end".
- **Actions:** ACT reports that deltas did worse than absolute targets. An action-space study ([2602.23408](https://arxiv.org/abs/2602.23408)) finds chunk-relative actions best for ACT (EE 63 → 78 %), and per-step deltas worst, with error growing as O(k). UMI reports chunk-relative 20/20, per-step delta 16/20.
- **Encoder:** at 84–96 px, the strong results all use a CNN trained from scratch with GroupNorm, a spatial-softmax head and random-shift augmentation:
  - spatial softmax vs global average pooling: 90 vs 81;
  - from scratch vs frozen ImageNet: 0.94 vs 0.58;
  - no shift augmentation: −35 to −47 % relative.
  ResNet-18 at 96 px gives ACT only 3×3 tokens per camera.
- **Data:** more and noisier demos are cheap here. On ManiSkill with ACT from RGB, going from 100 to 1000 demos raised success from 28 % to 98 %. DART-style noise injection gives recovery data.
- **Video:** LeRobot's own references train on AV1 yuv420p and evaluate on raw renders without trouble. Compression can't explain a 1e-4 image sensitivity, but the integrity check measures it.

## Plan for tonight

The CPU runs one heavy job at a time.

1. **Pipeline proof (running):**
   - Privileged-state ACT, no images, with delta vs absolute targets (`experiments/privileged/`).
   - Data and codec integrity checks (`experiments/integrity/`).
2. **Run 6, LR only:** run 2 with lr 1e-5 / backbone 1e-5. Image-swap probe at every 1k-step checkpoint, then eval on fresh seeds. A cheap, clean test of the leading hypothesis, about 25 min.
3. **Small spatial-softmax BC** (`experiments/smallbc/`): from scratch, random shift, absolute or chunk-relative targets, chunk 16 / exec 8, 200–500 noise-injected demos stored uncompressed.
4. **Overnight ACT with the combined recipe:** lr 1e-5, the best action form from step 1, more demos, 30k steps or more, checkpoints every 5k with the swap probe.

Then a final eval of the best policy on untouched seeds 11000–11099 (nominal + shifted).

Seed ledger:
- 10000–10499: spent (runs 1–5).
- 10500–10599: dev, for the experiments.
- 10600–10699: run 6 test.
- 10700–10999: later runs.
- 11000–11099: the final test, used once.

## Results tonight (updated 01:40 IST)

| Experiment | Change | Result |
|---|---|---|
| Integrity checks (`integrity/REPORT.md`) | n/a | Pipeline sound: cube in the pixels (R² ≥ 0.994), replay exact, no off-by-one, normalisation identical train/eval, codec harmless (≤ 3e-4 output change). The net doesn't even fit the constant frame-0 d_y = −0.6. |
| Run 6 (R4 session, in repo) | run 2 + lr 1e-5 / backbone 1e-5 | **0/100** on test seeds 10600–10699. Swap 0.001–0.003, shrinking with training. **The LR is not the cause.** |
| Privileged (a) (`privileged/REPORT.md`) | cube pose as input, no images, delta actions | **0/50**, swap 0.00014. It fails even with perfect perception. |
| Privileged (b) | cube pose as input, absolute TCP targets | **50/50** (93–100 %); 10/10 on demo seeds. |
| **Run 7** (`privileged/run7/NOTES.txt`) | run 2 (pixels + state, 96 px, 6000 steps) + absolute targets only | **31/50 = 62 %** nominal (dev seeds 10500–10549, Wilson 48–74 %). 7/10 on demo seeds, 11/50 shifted. Swap 1.15 (x) / 1.98 (yaw) as delta, about 10× the criterion. |

**Root cause:** the per-step delta action representation. The cube information lives in about 12 short, saturated pulse frames per episode, and the L1-trained policy settles on their median, 0. With absolute targets, the cube's x, y and yaw are in the label for 40+ frames and vary smoothly with its pose. This matches ACT's own note that delta actions did worse, and the action-space study (2602.23408).

## Run 8 (03:44 IST)

The recipe: 200 demos (seeds 0–199), half with DART noise (std 0.3 on x, y, z and yaw), clean absolute-target labels, 20k steps at batch 8. Training took 113 min. Details are in `privileged/run8/NOTES.txt`.

| Checkpoint | Dev nominal (10500–10549) | Wilson 95 % | Failures |
|---|---|---|---|
| 12k | 22/50 | 31–58 % | never_reached 7, grasp_missed 21 |
| 16k | 21/50 | 29–56 % | never_reached 1, grasp_missed 27, dropped 1 |
| **20k** | **36/50 = 72 %** | 58–83 % | grasp_missed 10, dropped 4 |
| 20k, shifted | 10/50 | 11–33 % | never_reached 31 |

- Run 8's 72 % is not clearly better than run 7's 62 %: the intervals overlap. The checkpoint-to-checkpoint swing (42 → 72 %) is as large as any effect here.
- On nominal poses the approach is solved; the failures left are grasps.
- The shifted box is still around 20 %, an extrapolation failure that DART noise does not address.

**Revised seed ledger:** runs 7 and 8 are evaluated on the same test seeds, 10700–10799, nominal and shifted (a paired comparison). Run 8 @ 20k is pre-registered as the headline. 10800–10899 is held for smallbc, and 11000+ stays untouched.

**Next (running from 03:46):**
- The R4 session merges `feat/abs-actions`.
- It checks that `sdp eval` reproduces run 7's 31/50 on dev, episode by episode.
- It then runs the official test evals and updates the README.
- smallbc (1000 demos, 50k steps, capped at 190 min) starts once the evals hand the CPU back.

## Official test results (03:56 IST)

The R4 session ran these with the repo's own `sdp eval`. As a reproduction check, run 7 @ 6k on dev 10500–10549 gave 31/50, matching the experiment's eval (`results/repro/`).

Test seeds 10700–10799, the same seeds for both policies:

| Policy | Nominal | Wilson 95 % | Shifted | Wilson 95 % |
|---|---|---|---|---|
| **Run 8 @ 20k (pre-registered headline)** | **63/100** | 53–72 % | 16/100 | 10–24 % |
| Run 7 @ 6k | 57/100 | 47–66 % | 19/100 | 13–28 % |

- **Paired comparison:**
  - Nominal: run 8 succeeds and run 7 fails on 23 seeds; the reverse on 17; both succeed on 40. Exact McNemar p = 0.43.
  - Shifted: 8 vs 11, both on 8. p = 0.65.
  - So the two runs are indistinguishable. 4× data, DART noise and 3.3× the steps did not measurably help.
- **Failures:**
  - Nominal is mostly grasp_missed: run 8 30, run 7 37; never_reached 1–3.
  - Shifted is mostly never_reached, 69 for each run.
- **Selection optimism:** dev → test was 72 → 63 % for run 8 and 62 → 57 % for run 7.

## smallbc results (30 Oct 2026, ~06:15–07:00 IST; appended after the overnight run)

- **Checkpoint choice:** smallbc picked 25k on dev seeds 10500–10549 with the rule fixed
  in `run_full.sh` (25k 49/50, 40k 48/50, last 48/50). It was pre-registered as the
  repo's headline policy before any test eval (docs/preregistration.md, commit 0d9b50d).
- **Official test, seeds 10800–10899:** nominal **100/100** (Wilson 95 % 96–100 %);
  shifted **72/100** (63–80 %; 26 collision, 2 dropped).
- **Paired seeds 10700–10799** (where ACT runs 7 and 8 were scored): nominal 97/100
  (92–99 %), shifted 70/100 (60–78 %). Against run 8, nominal: 35 seeds succeed only
  with smallbc and 1 only with run 8, exact McNemar p 1.077e-09. Shifted: 54 vs 0,
  p 1.11e-16 (`sdp report`, paired table).
- **Collisions:** all 26 shifted collisions on 10800–10899 are open fingers pressed
  into the table short of a cube that started beyond the nominal box (start x
  0.6112–0.67926 m; the box ends at 0.60). Source:
  `results/collisions_smallbc_abs1000_25k_s10800_shifted.json`.
- **Caveat:** this is a recipe comparison, not an architecture one. smallbc had ~5×
  the demos, batch 64 vs 8, a 16/8 chunk vs 20/10, and the commanded TCP target as an
  extra input.
- **Into the package:** the code is now in `sdp/policies/smallbc/`; this folder stays
  as the record.
