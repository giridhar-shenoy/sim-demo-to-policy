#!/usr/bin/env bash
# Evaluate both CPU-config ACT checkpoints and refresh the doc tables.
#   usage: scripts/eval_all.sh [run1_dir] [run2_dir]
# Seeds (all from the eval range 10000-19999, disjoint from demo seeds 0-999):
#   10000-10099 "dev": run 1's first eval; its result led to run 2's config.
#   10100-10199 "test": used once for each run after run 2's config was frozen.
# Every call runs 100 nominal + 100 shifted-pose episodes on the same (paired) seeds.
set -euo pipefail
RUN1=${1:-outputs/train/act_cpu_scripted50}
RUN2=${2:-outputs/train/act_cpu_v2_scripted50}
sdp eval --checkpoint "$RUN1" --condition act_cpu_scripted50 --trials 100 --seed-start 0
sdp eval --checkpoint "$RUN1" --condition act_cpu_scripted50 --trials 100 --seed-start 100
sdp eval --checkpoint "$RUN2" --condition act_cpu_v2_scripted50 --trials 100 --seed-start 100
sdp report
