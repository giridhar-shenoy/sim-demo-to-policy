#!/usr/bin/env bash
# Record the 50 scripted PandaPickPlace demos (demo seeds 0-49) as a LeRobot v3.0 dataset.
#   usage: scripts/record_demos.sh [out_dir]
set -euo pipefail
OUT=${1:-datasets/pickplace_scripted_50}
sdp record --env PandaPickPlace-v0 --episodes 50 --source scripted \
  --out "$OUT" --repo-id local/sdp_pickplace_scripted_50
python scripts/dataset_summary.py "$OUT"
