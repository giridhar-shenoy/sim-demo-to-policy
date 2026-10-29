#!/usr/bin/env bash
# Fetch the v0.1 dataset and the four ACT checkpoints from the GitHub Release, so the evals
# can be rerun without retraining:
#   scripts/download_release_assets.sh && scripts/eval_all.sh
set -euo pipefail
TAG=${1:-v0.1.0}
BASE=https://github.com/giridhar-shenoy/sim-demo-to-policy/releases/download/$TAG
mkdir -p datasets outputs/train
curl -fsSL "$BASE/pickplace_scripted_50.tar.gz" | tar -xz -C datasets
for run in act_cpu_scripted50 act_cpu_v2_scripted50 act_cpu_v3_scripted50 act_cpu_v4_scripted50; do
  curl -fsSL "$BASE/$run.tar.gz" | tar -xz -C outputs/train
done
echo "dataset -> datasets/pickplace_scripted_50, checkpoints -> outputs/train/act_cpu{,_v2,_v3,_v4}_scripted50"
