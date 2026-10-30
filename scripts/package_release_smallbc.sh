#!/usr/bin/env bash
# Pack smallbc @ 25k (ckpt_025000.pt + config.json) for the GitHub Release.
# The 5.8 GB smallbc dataset is not shipped. `python -m sdp.policies.smallbc.record`
# (flags in scripts/regenerate_all.sh) regenerates it from demo seeds 0-999: seeds,
# expert and DART noise are deterministic, and only the renderer can differ slightly
# across machines.
#   usage: scripts/package_release_smallbc.sh [run_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-experiments/smallbc/runs/abs1000}
OUT=outputs/release
mkdir -p "$OUT"
tar -czf "$OUT/smallbc_abs1000_25k.tar.gz" -C "$(dirname "$RUN")" \
  --transform "s,^$(basename "$RUN"),smallbc_abs1000_25k," \
  "$(basename "$RUN")/ckpt_025000.pt" "$(basename "$RUN")/config.json"
ls -lh "$OUT/smallbc_abs1000_25k.tar.gz"
