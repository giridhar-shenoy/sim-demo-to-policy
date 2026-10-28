#!/usr/bin/env bash
# Regenerate docs/assets/act_success_failure.gif from the eval CSV and the CPU checkpoint.
set -euo pipefail
python scripts/record_gifs.py --checkpoint "${1:-outputs/train/act_cpu_scripted50}"
