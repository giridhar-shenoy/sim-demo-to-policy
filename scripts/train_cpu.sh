#!/usr/bin/env bash
# Train ACT (CPU config) on a recorded dataset and copy the loss log + wall time to results/.
#   usage: scripts/train_cpu.sh [dataset_root] [output_dir]
set -euo pipefail
DATASET=${1:-datasets/pickplace_scripted_50}
OUT=${2:-outputs/train/act_cpu_scripted50}
NAME=$(basename "$OUT")
python -m sdp.train.train --config sdp/train/configs/act_cpu.yaml --dataset "$DATASET" --output-dir "$OUT"
mkdir -p results
cp "$OUT.csv" "results/train_${NAME}.csv"
python - "$OUT.json" "results/train_${NAME}.json" <<'PY'
import json, platform, sys
meta = json.load(open(sys.argv[1]))
meta.pop("log", None)
meta["config"] = "sdp/train/configs/act_cpu.yaml"
meta["machine"] = "i5-1135G7 (4C/8T), 14 GB RAM, CPU only, torch 4 threads"
meta["python"] = platform.python_version()
json.dump(meta, open(sys.argv[2], "w"), indent=2)
open(sys.argv[2], "a").write("\n")
PY
