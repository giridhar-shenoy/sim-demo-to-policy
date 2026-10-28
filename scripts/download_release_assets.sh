#!/usr/bin/env bash
# Fetch the v0.1 dataset and ACT checkpoint from the GitHub Release, so `sdp eval` can run
# without retraining:
#   scripts/download_release_assets.sh && scripts/eval_all.sh outputs/train/act_cpu_scripted50
set -euo pipefail
TAG=${1:-v0.1.0}
BASE=https://github.com/giridhar-shenoy/sim-demo-to-policy/releases/download/$TAG
mkdir -p datasets outputs/train
curl -fsSL "$BASE/pickplace_scripted_50.tar.gz" | tar -xz -C datasets
curl -fsSL "$BASE/act_cpu_scripted50.tar.gz" | tar -xz -C outputs/train
echo "dataset -> datasets/pickplace_scripted_50, checkpoint -> outputs/train/act_cpu_scripted50"
