#!/usr/bin/env bash
# Pack the absolute-target runs for the GitHub Release: run 7 @ 6000 (with its 50-demo
# dataset) and run 8 @ 20000 (with its 198-episode DART dataset). About 150 MB of gzip;
# run it only when no other heavy job needs the CPU.
#   usage: scripts/package_release_abs.sh
set -euo pipefail
cd "$(dirname "$0")/.."
P=experiments/privileged
scripts/package_release_assets.sh $P/run7/datasets/pickplace_abs_50 $P/run7/train/act_cpu_v7_abs 006000
scripts/package_release_assets.sh $P/run8/datasets/pickplace_abs_dart_200 $P/run8/train/act_cpu_v8_abs_dart 020000
