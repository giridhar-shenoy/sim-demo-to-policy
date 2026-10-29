#!/usr/bin/env bash
# Evaluate the five CPU-config ACT checkpoints and refresh the doc tables.
#   usage: scripts/eval_all.sh [run1_dir] [run2_dir] [run3_dir] [run4_dir] [run5_dir]
# Seeds (all from the eval range 10000-19999, disjoint from demo seeds 0-999):
#   10000-10099 "dev": run 1's first eval; its result led to run 2's config.
#   10100-10199 "test 1": runs 1 and 2, once each, after run 2's config was frozen.
#   10200-10299 "test 2": run 3 (run 2 with 8000 steps), never seen by any choice.
#   10300-10399 "test 3": run 4 (run 2 without the proprioceptive state input).
#   10400-10499 "test 4": run 5 (run 4 with 128 px cameras).
# Every call runs 100 nominal + 100 shifted-pose episodes on the same (paired) seeds.
set -euo pipefail
RUN1=${1:-outputs/train/act_cpu_scripted50}
RUN2=${2:-outputs/train/act_cpu_v2_scripted50}
RUN3=${3:-outputs/train/act_cpu_v3_scripted50}
RUN4=${4:-outputs/train/act_cpu_v4_scripted50}
RUN5=${5:-outputs/train/act_cpu_v5_scripted50_128px}
sdp eval --checkpoint "$RUN1" --condition act_cpu_scripted50 --trials 100 --seed-start 0
sdp eval --checkpoint "$RUN1" --condition act_cpu_scripted50 --trials 100 --seed-start 100
sdp eval --checkpoint "$RUN2" --condition act_cpu_v2_scripted50 --trials 100 --seed-start 100
sdp eval --checkpoint "$RUN3" --condition act_cpu_v3_scripted50 --trials 100 --seed-start 200
sdp eval --checkpoint "$RUN4" --condition act_cpu_v4_scripted50 --trials 100 --seed-start 300
sdp eval --checkpoint "$RUN5" --condition act_cpu_v5_scripted50_128px --trials 100 --seed-start 400
sdp report
