#!/usr/bin/env bash
# Regenerate docs/assets/act_success_failure.gif from run 2's test-seed eval CSV.
set -euo pipefail
python scripts/record_gifs.py --checkpoint "${1:-outputs/train/act_cpu_v2_scripted50}" \
  --csv results/eval_act_cpu_v2_scripted50_s10100.csv
