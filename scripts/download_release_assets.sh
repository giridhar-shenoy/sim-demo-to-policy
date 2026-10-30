#!/usr/bin/env bash
# Fetch the v0.1 release assets (the LeRobot datasets, the eight ACT checkpoints and
# smallbc @ 25k) from the GitHub Release, so the evals can be rerun without retraining:
#   scripts/download_release_assets.sh && scripts/eval_all.sh
set -euo pipefail
TAG=${1:-v0.1.0}
BASE=https://github.com/giridhar-shenoy/sim-demo-to-policy/releases/download/$TAG
mkdir -p datasets outputs/train
for ds in pickplace_scripted_50 pickplace_scripted_50_128px pickplace_abs_50 pickplace_abs_dart_200; do
  curl -fsSL "$BASE/$ds.tar.gz" | tar -xz -C datasets
done
for run in act_cpu_scripted50 act_cpu_v2_scripted50 act_cpu_v3_scripted50 act_cpu_v4_scripted50 \
  act_cpu_v5_scripted50_128px act_cpu_v6_scripted50 act_cpu_v7_abs act_cpu_v8_abs_dart; do
  curl -fsSL "$BASE/$run.tar.gz" | tar -xz -C outputs/train
done
curl -fsSL "$BASE/smallbc_abs1000_25k.tar.gz" | tar -xz -C outputs
echo "datasets -> datasets/, checkpoints -> outputs/train/act_cpu*, outputs/smallbc_abs1000_25k/"
echo "smallbc: sdp eval --checkpoint outputs/smallbc_abs1000_25k/ckpt_025000.pt ..."
echo "runs 7 and 8: scripts/eval_all.sh <run1..6 dirs> outputs/train/act_cpu_v7_abs outputs/train/act_cpu_v8_abs_dart"
