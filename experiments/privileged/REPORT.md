# Privileged-state ACT sanity experiment

Saved by the coordinating session from the privileged agent's report (30 Oct, about 00:55).

**Question:** can the record → `lerobot-train` → eval pipeline in this repo produce a working ACT policy if the policy is *given* the cube pose? Two variants: (a) the current 7-D delta-TCP actions; (b) absolute TCP targets.

## Results

| Run | Seeds | Success | Wilson 95 % | Labels | Mean steps |
|---|---|---|---|---|---|
| (a) delta, `act_priv_delta` | eval 10500–10549 (fresh) | 0/50 | 0–7.1 % | never_reached 50 | 250 (timeout) |
| (a) delta | demo 0–9 | 0/10 | 0–27.8 % | never_reached 10 | 250 |
| (b) absolute, `act_priv_abs` | eval 10500–10549 (fresh) | **50/50** | 92.9–100 % | success 50 | 107.7 |
| (b) absolute | demo 0–9 | **10/10** | 72.2–100 % | success 10 | 109.0 |

### Open-loop sensitivity (frame 2, first 20 training episodes)

| | (a) delta | (b) absolute (raw target) | (b) as the wrapper's delta |
|---|---|---|---|
| first-step x: pred std / data std | 0.0001 / 0.376 | 0.031 m / 0.027 m (corr 0.997) | 0.54 / 0.38 |
| first-step yaw: pred std / data std | 0.00003 / 0.552 | 0.186 / 0.170 rad (corr 0.974) | 0.71 / 0.55 |
| cube-pose swap, max change in x | **0.00014** | 0.039 m | **1.07** |
| cube-pose swap, max change in yaw | **0.00005** | 0.36 rad | **1.50** |

Run (a) fails the repo's own pre-registered criterion (≥ 0.1 change and ≥ 0.1 std in x or yaw) just as runs 4 and 5 did. Run (b) passes it by more than 10×.

### Why (a) fails
- Running the model on the exact training batch path gives the same outputs as the adapter, so this is not an adapter bug.
- At frame 0 the robot state is identical in all 50 episodes and the recorded y is −0.6 in every one. The model still predicts −0.01.
- Approach-frame (0–11) L1 is 0.146, 0.145 and 0.145 at steps 2k, 4k and 6k. On frames 40–99 it is 0.060, 0.035 and 0.028. The approach is stuck, not converging slowly.
- In delta form the cube signal is about 12 sparse, saturated pulse frames per episode, and L1 settles on their median, 0. In absolute form the cube's x, y and yaw are in the target for the whole approach, descend and close (40+ frames), and they vary smoothly with the cube pose.

## Setup
- **Datasets:** `datasets/priv_{delta,abs}_50`, demo seeds 0–49, 5513 frames, no video.
  - `observation.state` is the 15-D robot state. `observation.environment_state` is the cube pose [x, y, z, sin yaw, cos yaw].
  - Normalisation is MEAN_STD for ENV as well, set explicitly.
- **Wrapper check:** replaying (b)'s absolute actions through `AbsoluteTargetWrapper` succeeds on 50/50 seeds, with deltas within 2.3e-6 of the expert's.
- **Training:** run 2's architecture (dim 256, 2/1 layers, chunk 20 / exec 10, lr 1e-4, CVAE off). 6000 steps at batch 32 (34.8 epochs), about 7 steps/s, about 14 min each. Final loss: (a) 0.051, (b) 0.045.
- **Code:** `priv_common.py` (env_state, abs_target, abs_to_delta, AbsoluteTargetWrapper), `record_privileged.py`, `verify_replay.py`, `act_priv_{delta,abs}.yaml`, `run_train.sh`, `priv_policy.py`, `eval_privileged.py`, `sensitivity.py`, `run_eval.sh`.

## Conclusion
- The pipeline is sound.
- The delta-TCP action representation is what breaks learning: with the cube pose given directly, the delta policy still ignores it.
- Absolute targets fix it at this budget.
- Caveats: one training seed per variant, nominal distribution only. Privileged success does not show that pixels are enough to localise the cube.
