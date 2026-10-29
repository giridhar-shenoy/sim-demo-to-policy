# Integrity checks: does the cube information reach the ACT policy?

Scripts and JSON outputs are in this folder. Cost: 23 env episodes plus 100 reset-only calls; peak RSS 1.1 GB. Saved by the coordinating session from the integrity agent's report.

## 1. Video compression (`check1_codec.py`)
Seeds 0–2 were re-run at 96 px (331 frames). The states and actions match the dataset bit for bit.
- Front: mean abs diff 1.88, max 216, PSNR 35.1 dB. Wrist: 2.52, max 253, PSNR 31.8 dB. At 128 px (seed 0): 35.9 / 33.6 dB.
- Red-cube mask: R>90, R>2G, R>2B.
  - Front, frames 0/5/10: raw vs decoded counts are 59/58, 57/59 and 51/52. Centroid shift is 0.10–0.17 px.
  - Wrist: the cube is desaturated (R 217→195) and its colour bleeds outward. Centroid shift is 0.07–0.83 px.
- Verdict: the codec softens and desaturates the cube but does not erase it. There is a slight train (decoded) vs eval (raw) mismatch.

## 2. Is the cube position in the pixels? (`check2_cube_in_pixels.py`)
A linear fit from the frame-0 red centroid to the true cube (x, y), leave-one-out over the 50 episodes:
- Front: R² 0.997 / 0.998 (error 2.3 / 1.8 mm).
- Wrist: R² 0.996 / 0.994.
- At 128 px both cameras reach ≥ 0.997.

## 3. Is the action learnable from the cube pose? (`check3_action_learnable.py`)
Held-out R² (5-fold by episode), frames 0–5:

| target | linear | poly3 | kNN (x, y, yaw) | kNN on the matching coordinate |
|---|---|---|---|---|
| d_x | 0.68 | 0.85 | 0.71 | 0.99 |
| d_y | 0.29 | 0.77 | 0.78 | 0.96 |
| d_yaw | 0.89 | 0.93 | 0.86 | 0.98 |

- Frame-0 d_x = clip(50·(x−0.5545), ±0.6), R² 1.000. It is saturated in 80 % of episodes.
- Approach phase: 13.1 frames on average (8–18), 11.9 % of all frames.
- Saturation during the approach: d_x 21 %, d_y 62 %, d_yaw 49 %.

## 4. Recorder alignment (`check4_alignment.py`, `check5b_train_items.py`)
- Open-loop replay of seeds 0–9: 10/10 success, max state divergence 0.0.
- A shadow expert's action at obs_t equals the recorded action[t] at every frame, so there is no off-by-one.
- Training items pair the right images with action[t:t+20] and the correct padding flags.

## 5. Normalisation (`check5_normalisation.py`)
- Images: MEAN_STD with ImageNet statistics, the same for training and eval.
- Action and state: MEAN_STD with dataset statistics.
- Action std: d_x 0.141, d_y 0.303, d_z 0.413, roll 0, pitch 0, d_yaw 0.149, grip 0.999.
- The adapter's preprocessed tensors differ from the training path by **0.0**.
- Latent hazard (runs with state input): tcp_qw and tcp_qz have std of about 2.6e-4.

## 6. Policy input sensitivity (`check6_policy_sensitivity.py`)
- Run 5, raw vs decoded input: the first action changes by at most 5e-5 at frame 0, and by at most 3e-4 over frames 0–10.
- For scale: swapping in another episode's images changes it by at most 7.6e-4. The recorded actions vary by about 0.55 (std).

## Verdict
1. **No bug that blocks the information.** The cube is in the pixels, the actions are a fixed function of the cube pose, and recording, training items and normalisation all match exactly.
2. **Codec mismatch: real but harmless.**
3. **Data property:** cube-dependent targets are rare (11.9 % of frames) and mostly saturated.
4. **Latent:** the tcp_qw and tcp_qz state std is tiny.
5. **The network never fits the approach frames** (`check7_chunk_probe.py`). This includes frame-0 d_y = −0.6, which is identical in all 50 demos on an input that is identical in every episode. The prediction there is about −0.005. Predicted approach x/y/yaw stay flat (std ≤ 3e-4, against 0.10–0.29 recorded), while d_z and the gripper are fitted. The cause is under-fitting or collapse during training, not the pipeline.
