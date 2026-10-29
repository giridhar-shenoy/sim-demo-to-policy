#!/usr/bin/env bash
# Pack the dataset and one run's final ACT checkpoint (pretrained_model only, ~96 MB)
# for a GitHub Release; run it once per run dir. Uploading is a manual step (see MORNING.md / README):
#   gh release create v0.1.0 outputs/release/*.tar.gz
set -euo pipefail
DATASET=${1:-datasets/pickplace_scripted_50}
RUN=${2:-outputs/train/act_cpu_scripted50}
OUT=outputs/release
mkdir -p "$OUT"
tar -czf "$OUT/$(basename "$DATASET").tar.gz" -C "$(dirname "$DATASET")" "$(basename "$DATASET")"
tar -czf "$OUT/$(basename "$RUN").tar.gz" -C "$RUN/checkpoints/last" pretrained_model \
  --transform "s,^pretrained_model,$(basename "$RUN")/pretrained_model,"
ls -lh "$OUT"
