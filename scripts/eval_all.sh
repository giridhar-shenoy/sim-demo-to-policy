#!/usr/bin/env bash
# Evaluate the ACT checkpoints (runs 1-8) and refresh the doc tables.
#   usage: scripts/eval_all.sh [run1_dir] … [run6_dir] [run7_ckpt] [run8_ckpt] [smallbc_ckpt]
# Seeds (all from the eval range 10000-19999, disjoint from demo seeds 0-999):
#   10000-10099 "dev": run 1's first eval; its result led to run 2's config.
#   10100-10199 "test 1": runs 1 and 2, once each, after run 2's config was frozen.
#   10200-10299 "test 2": run 3 (run 2 with 8000 steps), never seen by any choice.
#   10300-10399 "test 3": run 4 (run 2 without the proprioceptive state input).
#   10400-10499 "test 4": run 5 (run 4 with 128 px cameras).
#   10500-10599: reserved for the experiments under experiments/ (not used here).
#   10600-10699 "test 5": run 6 (run 2 with lr 1e-5 / backbone 1e-5).
#   10700-10799 "test 6": runs 7 and 8 (absolute targets), the same seeds for both
#                         (paired); policies and seeds fixed in docs/preregistration.md
#                         (commit 8060419) before testing.
#   10800-10899 "test 7": smallbc @ 25k (official); it is also scored on 10700-10799
#                         for the paired comparison with runs 7 and 8. 11000+: untouched.
# Every call runs 100 nominal + 100 shifted-pose episodes on the same (paired) seeds.
set -euo pipefail
RUN1=${1:-outputs/train/act_cpu_scripted50}
RUN2=${2:-outputs/train/act_cpu_v2_scripted50}
RUN3=${3:-outputs/train/act_cpu_v3_scripted50}
RUN4=${4:-outputs/train/act_cpu_v4_scripted50}
RUN5=${5:-outputs/train/act_cpu_v5_scripted50_128px}
RUN6=${6:-outputs/train/act_cpu_v6_scripted50}
RUN7=${7:-outputs/train/act_cpu_v7_abs}
RUN8=${8:-outputs/train/act_cpu_v8_abs_dart}
sdp eval --checkpoint "$RUN1" --condition act_cpu_scripted50 --trials 100 --seed-start 0
sdp eval --checkpoint "$RUN1" --condition act_cpu_scripted50 --trials 100 --seed-start 100
sdp eval --checkpoint "$RUN2" --condition act_cpu_v2_scripted50 --trials 100 --seed-start 100
sdp eval --checkpoint "$RUN3" --condition act_cpu_v3_scripted50 --trials 100 --seed-start 200
sdp eval --checkpoint "$RUN4" --condition act_cpu_v4_scripted50 --trials 100 --seed-start 300
sdp eval --checkpoint "$RUN5" --condition act_cpu_v5_scripted50_128px --trials 100 --seed-start 400
sdp eval --checkpoint "$RUN6" --condition act_cpu_v6_scripted50 --trials 100 --seed-start 600
sdp eval --checkpoint "$RUN8" --condition act_cpu_v8_abs_dart --trials 100 --seed-start 700
sdp eval --checkpoint "$RUN7" --condition act_cpu_v7_abs --trials 100 --seed-start 700
# smallbc @ 25k (headline): official test 10800-10899, paired comparison 10700-10799.
SMALLBC=${9:-outputs/smallbc_abs1000_25k/ckpt_025000.pt}
for seed in 10800 10700; do
  for dist in nominal shifted; do
    sdp eval --checkpoint "$SMALLBC" --condition smallbc_abs1000_25k --trials 100 \
      --seed-start $((seed - 10000)) --distributions "$dist" \
      --out "results/eval_smallbc_abs1000_25k_s${seed}_${dist}"
  done
done
sdp report
