# smallbc: small from-scratch CNN + spatial-softmax BC with action chunks

This is an alternative to the five LeRobot ACT runs, which all scored 0 %. Those used an
ImageNet ResNet-18 with a 3×3 or 4×4 feature map, chained per-step delta targets and 50
deterministic demos, and never localised the cube. This policy follows the recipe that
works under small compute: a robomimic / Diffusion-Policy-style encoder trained from
scratch (GroupNorm, spatial-softmax keypoints), DrQ random-shift augmentation, absolute
or chunk-relative position targets, action chunking, and many demos with DART-style
noise injection.

Status: built and smoke-tested only. The full record, train and eval have **not** been
run (another job held `heavy.lock`).

## Design

**Data (`record.py`).** The scripted `WaypointExpert` is recorded into uncompressed,
memory-mappable `.npy` shards (no video codec). Each shard holds up to 25 successful
episodes. Fields per frame:

| Field | Contents |
| --- | --- |
| `front`, `wrist` | uint8 images at `--image-size` (default 96) |
| `state` | the env's 15-D state |
| `cube` | x, y, z, yaw |
| `target_obs` | the integrated TCP target at observation time: x, y, z, yaw |
| `tilt` | tilt of that target |
| `action_exec` | the delta action actually sent to `env.step` |
| `action_clean` | the expert's clean delta action |
| `target_abs` | the target after the clean action: x, y, z, yaw, gripper |
| `phase` | the expert phase |
| `frame_index` | step index within the episode |

Yaw is measured about world z relative to the home orientation, using the same
decomposition the env uses to clamp it.

- **DART noise.** `--noise-frac` sets the share of episodes that get noise; the choice is
  made per seed and is deterministic. In those episodes, Gaussian noise
  (`--noise-std`, in normalised action units) is added to the dims in `--noise-dims` of
  the **executed** action. The labels stay the clean expert output for the state
  actually visited.
- **Kept episodes.** Only successful episodes are written. The success rate is logged
  separately for clean and noised episodes.
- **Seeds.** Seeds come from the demo range 0–999; `sdp.seeds.check_demo` rejects
  anything else.
- **Clean-label self-check.** For clean episodes the recorder checks that the env's new
  target equals the stored `target_abs`. It is exact (error 0.0).
- **No overwrites.** The recorder refuses to overwrite an existing shard.
- **Parallel recording.** `--workers N` splits the seeds over N processes.

**Noise dims: use `--noise-dims 0,1,2,5`.** The default, `0,1,2,3,4,5`, is the requested
"6 delta dims". The expert never corrects roll or pitch, so noise on dims 3 and 4
random-walks the tilt. With std 0.3 the target hit the env's 0.5 rad tilt clamp, and 1
of 2 noised episodes ended in a collision. With noise on x, y, z and yaw only, 3 of 3
succeeded with zero tilt. The policy never commands roll or pitch either, so tilt noise
would only add images it can never reproduce. `run_full.sh` uses `0,1,2,5`.

**Model (`model.py`, 574 k parameters at the defaults).**

- **Per-camera encoder.** Four conv layers with GroupNorm(8) and ReLU: 5×5 stride 2 (32
  channels), 3×3 stride 2 (64), 3×3 stride 2 (64), 3×3 stride 1 (64). A 96 px input
  gives a 12×12 map; 128 px gives 16×16.
- **Spatial-softmax head.** A 1×1 conv produces 32 heatmaps. A softmax with a learned
  temperature turns each into an expected (x, y) in [-1, 1], so each camera yields 64
  coordinates.
- **MLP head.** Input: the front and wrist keypoints plus the normalised proprio vector
  (on by default; `--no-proprio` gives cameras only). Proprio is the 15-D state plus the
  integrated target [x, y, z, yaw], 19-D in all. The MLP is 512-512 and outputs a K×5
  chunk (K = 16 by default).
- **Targets (`--target`).**
  - `abs`: [x, y, z, yaw, gripper] of the integrated target after each of the next K
    clean actions.
  - `rel`: the same, with x, y, z and yaw taken as offsets from the target at
    observation time (yaw wrapped). The gripper stays absolute.

  Chunks past the end of an episode repeat the last target.
- **Normalisation.** Mean and std are computed on the training split, per output
  dimension (each of the K×5 entries) and per proprio dimension. They are stored as
  buffers, so a checkpoint is self-contained. The loss is L1 in normalised space.
- **Augmentation.** DrQ random shift: replicate-pad 4 px, then a random crop back to 96,
  independently per sample and per camera. It runs in training only.

**Training (`train.py`).**

- Adam, lr 3e-4, a 500-step linear warmup, then cosine decay to 0. Batch 64, 4 torch
  threads, channels_last (about 10 % faster).
- Batches are plain numpy gathers from the memmaps (num_workers 0). 10 % of episodes are
  held out by seed.
- Logs `train_log.csv` with step, loss, lr, steps/s, data ms and compute ms.
- Every `--val-every` steps it also logs:
  - validation L1;
  - xy, z and yaw error in physical units;
  - `val_xy_mm_approach`: the xy error on frames in the expert's above_cube and descend
    phases.
- Writes `ckpt_<step>.pt` and `last.pt`. `--max-minutes` stops early and still writes a
  checkpoint. `--data` accepts several record dirs or individual shard dirs, so a subset
  can be trained without re-recording.

`val_xy_mm_approach` targets the ACT failure directly. A policy that ignores the cube
predicts the mean approach target and should score roughly 30–40 mm (the cube box is
15×15 cm). A policy that localises the cube should go well below 10 mm.

**Policy (`policy.py`, harness `Policy`).**

- `reset()` clears the chunk queue.
- `act(obs, env)` predicts a chunk every `n_exec` steps (default 8). `rel` chunks are
  turned into absolute targets by adding the target at observation time.
- Each absolute target becomes the env's delta action:
  `clip((target - current integrated target) / max_step, -1, 1)`, using `max_dpos`
  (0.02 m) and `max_drot` (0.1 rad). The yaw difference is wrapped and roll/pitch are 0,
  as in the expert. The gripper is binarised by sign (`--no-binarize-gripper` turns that
  off).
- `--temporal-ensemble` predicts every step and averages all chunks that cover the
  current step, with ACT weights `exp(-m*i)`, oldest first (m = 0.01).
- The policy reads `env.target_pose()`, the robot's own commanded pose, which the delta
  interface needs anyway. It never reads the cube state.

**Eval (`eval.py`).** Runs `sdp.eval.harness.run_condition`, so it uses the same failure
taxonomy and Wilson intervals as the ACT runs. It writes a per-episode CSV and a JSON
holding the harness summary (success, `wilson95`, labels, seeds) plus metadata.

- The default seeds are the dev range 10500–10549.
- Seeds 10000–10499 (spent on the ACT runs) are refused before anything runs.
- The final test range is still open: pass `--seed-start` and `--n`.

**Converter check (`check_replay.py`).** An oracle predictor feeds the recorded clean
label chunks through `SmallBCPolicy`, with no network, on each demo's own seed, using
the harness. It covers abs ×1, abs ×8, rel ×8, abs with temporal ensembling and rel with
temporal ensembling.

`common.py` sets `MUJOCO_GL=egl` and, if unset, `SDP_MENAGERIE_DIR=~/work/menagerie`.

## Commands

Run from `~/work/r4`.

**One shot.** This takes `~/work/.locks/heavy.lock` and waits for MemAvailable ≥ 6 GB.
It records 1000 demos, trains 50k steps, then evaluates the 25k, 40k and last
checkpoints on dev seeds 10500–10549. It skips recording if the data dir already has a
`record_*.json`.

```bash
mkdir -p experiments/smallbc/logs
nohup experiments/smallbc/run_full.sh > experiments/smallbc/logs/run_full.out 2>&1 &
# variants: EPISODES=600 STEPS=40000 TARGET=rel RUN_NAME=rel600 experiments/smallbc/run_full.sh
```

**Step by step:**

```bash
PY=.venv/bin/python; D=experiments/smallbc
$PY $D/record.py --out $D/data/main1000 --episodes 1000 --seed-start 0 \
    --noise-std 0.3 --noise-frac 0.5 --noise-dims 0,1,2,5 --workers 4 --log-every 50
$PY $D/train.py --data $D/data/main1000 --out $D/runs/abs1000 --steps 50000 --target abs \
    --ckpt-every 5000 --val-every 1000 --max-minutes 200
$PY $D/eval.py --ckpt $D/runs/abs1000/last.pt --seed-start 10500 --n 50                         # dev, nominal
$PY $D/eval.py --ckpt $D/runs/abs1000/last.pt --seed-start 10500 --n 50 --temporal-ensemble     # dev, TE
$PY $D/eval.py --ckpt $D/runs/abs1000/last.pt --seed-start 10500 --n 50 --distribution shifted  # dev, shifted
$PY $D/check_replay.py --data $D/data/demo5                                                     # converter check
```

## Measured speeds (smoke tests)

All runs used `nice -n 10` while another `lerobot-train` job used about 4.6 of the 8
logical CPUs (load average 5–6). On an idle machine expect faster.

| What | Measured |
| --- | --- |
| Record, clean, 1 process | 5 demos in 4.2 s including env start = **71 ep/min**; steady **80 ep/min** (0.75 s per ~111-step episode) |
| Record, DART-noised (x, y, z, yaw; std 0.3), 1 process | 3 demos in 3.7 s = 49 ep/min including start; 56 ep/min steady; 114 frames per episode |
| Record, `--workers 2` | 2 episodes in 3.0 s. Process spawn dominates, so this is not a scaling number |
| Disk | 55.3 KB per frame at 96 px; 5 demos = 30 MB, so **1000 demos ≈ 6.2 GB** |
| Train, batch 64, 4 threads | 100 steps in 21.9 s = **4.6 steps/s** on average; 25-step windows ran at 3.1 / 4.4 / 6.0 / 6.2 steps/s; data loading about 1 ms per batch from the page cache; peak RSS 0.6 GB |
| Eval | 2 × 250-step episodes in 3.4 s; policy 1.0 ms per step on average (11.9 ms max, on chunk steps). A 50-episode eval takes ≤ 1.5 min |

## Smoke results

- **Record, 5 clean demos (seeds 0–4).** 5/5 succeeded, 555 frames, label self-check
  error 0.0.
- **Converter check.** 25/25 replays succeeded (5 seeds × 5 configs), each with the same
  step count as the recording. The largest difference between executed actions and the
  expert's clean actions was 2e-6. Results are in `results/check_replay.json`.
- **Train, 100 steps** (4 training episodes, 1 validation). Loss fell from 0.80 to 0.28,
  and validation L1 was 0.283. The checkpoint saved and reloaded.
- **Eval, 2 episodes** (seeds 10500–10501). 0/2, both `never_reached`, as expected after
  100 steps on 4 demos. The JSON has the Wilson interval and labels:
  `results/eval_smoke_last_x8_nominal_s10500_n2.json`.
- **Other paths.** The `rel` target, cameras-only input, the temporal-ensemble eval,
  multi-dir `--data` and the spent-seed guard all ran.
- **Demos used.** 10 in total: seeds 0–4 clean, 5–6 with noise on all 6 dims, and 7–9
  with noise on x, y, z and yaw.

## Projected plan for about 4 h of CPU

The machine has 4 cores and 8 threads, so run one training job at a time.

| Stage | Setting | Estimated wall time |
| --- | --- | --- |
| Record | 1000 demos (seeds 0–999), 50 % DART-noised (std 0.3, x, y, z, yaw), 4 workers | 5–15 min. Measured 80 ep/min per process under load. 4 workers should give 150–250 ep/min if EGL rendering scales; a single process would take 13–18 min |
| Train | `abs`, 50k steps, batch 64, chunk 16 | 2.2–3.0 h at 4.6–6.2 steps/s |
| Eval | 3 checkpoints × 50 dev episodes | about 4 min |
| **Total** | | **about 2.5–3.3 h** |

1000 demos give about 112k frames, about 100k of them for training. 50k × 64 = 3.2M
samples, about 32 epochs. `--max-minutes 200` caps training if the machine is slower
than measured.

**If memory is tight.** The 6.2 GB of images should stay in the page cache (MemTotal is
15.6 GB). Watch `data_ms` in `train_log.csv`: if it climbs above about 30 ms, batches are
coming from disk. In that case train on a subset of shards, for example
`--data $D/data/main1000/shard_0[0-5]*` (about 600 demos), or record `EPISODES=600`.

**What to look at first.** Watch `val_xy_mm_approach` in `train_log.csv`. If it stays
near 30–40 mm, the encoder is not localising the cube, the same failure as ACT, and the
eval will be 0 %.

**With time left over.** The same budget again buys `TARGET=rel STEPS=40000
RUN_NAME=rel1000`, which reuses the recorded data. Or a 128 px variant: re-record with
`--image-size 128`; the map becomes 16×16 and training is about 1.8× slower.

## Files

- `common.py`: env setup, yaw decomposition, `abs_to_delta`, seed constants.
- `record.py`: the demo recorder.
- `data.py`: shard loading, chunk labels, split, normalisation statistics.
- `model.py`: the encoder, spatial softmax and MLP head.
- `train.py`: the training loop.
- `policy.py`: the harness `Policy`.
- `eval.py`: harness eval to CSV and JSON.
- `check_replay.py`: the converter check.
- `run_full.sh`: the gated one-shot pipeline.
- Smoke outputs:
  - `data/demo5`, `data/noise2`, `data/noise3_xyzyaw`;
  - `runs/smoke`, `runs/smoke_rel`, `runs/smoke_multi`;
  - `results/`.
