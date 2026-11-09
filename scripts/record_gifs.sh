#!/usr/bin/env bash
# Regenerate docs/assets/rollouts.gif: the scripted expert (reference), smallbc @ 25k on a
# held-out nominal seed, and smallbc's first failure in the shifted pose box.
#   usage: scripts/record_gifs.sh [smallbc_checkpoint.pt]
#   default: the release checkpoint that scripts/download_release_assets.sh fetches.
set -euo pipefail
python scripts/record_gifs.py --checkpoint "${1:-outputs/smallbc_abs1000_25k/ckpt_025000.pt}" \
  --csv results/eval_smallbc_abs1000_25k_s10800_nominal.csv \
  --shifted-csv results/eval_smallbc_abs1000_25k_s10800_shifted.csv \
  --expert-panel --policy-label smallbc --out docs/assets/rollouts.gif
