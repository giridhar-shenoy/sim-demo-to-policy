#!/usr/bin/env bash
# Evaluate the CPU-config ACT checkpoint: 100 nominal + 100 shifted-pose episodes on
# eval seeds 10000-10099 (disjoint from demo seeds 0-999), then refresh the doc tables.
#   usage: scripts/eval_all.sh [checkpoint_dir]
set -euo pipefail
CKPT=${1:-outputs/train/act_cpu_scripted50}
sdp eval --checkpoint "$CKPT" --condition act_cpu_scripted50 --trials 100 \
  --distributions nominal shifted --out results/eval_act_pickplace
sdp report
