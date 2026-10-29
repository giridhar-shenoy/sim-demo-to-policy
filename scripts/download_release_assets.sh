#!/usr/bin/env bash
# Fetch the v0.1 datasets (96 and 128 px) and the five ACT checkpoints from the GitHub
# Release, so the evals can be rerun without retraining:
#   scripts/download_release_assets.sh && scripts/eval_all.sh
set -euo pipefail
TAG=${1:-v0.1.0}
BASE=https://github.com/giridhar-shenoy/sim-demo-to-policy/releases/download/$TAG
mkdir -p datasets outputs/train
for ds in pickplace_scripted_50 pickplace_scripted_50_128px; do
  curl -fsSL "$BASE/$ds.tar.gz" | tar -xz -C datasets
done
for run in act_cpu_scripted50 act_cpu_v2_scripted50 act_cpu_v3_scripted50 act_cpu_v4_scripted50 \
  act_cpu_v5_scripted50_128px; do
  curl -fsSL "$BASE/$run.tar.gz" | tar -xz -C outputs/train
done
echo "datasets -> datasets/pickplace_scripted_50{,_128px}, checkpoints -> outputs/train/act_cpu*"
