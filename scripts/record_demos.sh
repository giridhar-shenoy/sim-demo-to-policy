#!/usr/bin/env bash
# Record the 50 scripted PandaPickPlace demos (demo seeds 0-49) as a LeRobot v3.0 dataset.
#   usage: scripts/record_demos.sh [out_dir] [image_size] [action_mode: delta|abs_target]
set -euo pipefail
OUT=${1:-datasets/pickplace_scripted_50}
SIZE=${2:-96}
MODE=${3:-delta}
sdp record --env PandaPickPlace-v0 --episodes 50 --source scripted --image-size "$SIZE" --action-mode "$MODE" \
  --out "$OUT" --repo-id "local/$(basename "$OUT")"
python scripts/dataset_summary.py "$OUT"
