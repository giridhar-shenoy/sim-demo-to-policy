#!/usr/bin/env bash
# Regenerate every number and the GIF in README.md and docs/results.md from scratch.
# Needs: the repo venv active, SDP_MENAGERIE_DIR set (scripts/fetch_menagerie.sh), MUJOCO_GL=egl.
# Wall time on a 4-core laptop CPU: about 40 min, of which training is about 25 min.
# The dataset and checkpoint go to a fresh directory, so nothing existing is overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${RUN_DIR:-outputs/regen-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$RUN_DIR"
echo "== run dir: $RUN_DIR"

echo "== env throughput";   python scripts/bench_env_speed.py
echo "== scripted expert";  python scripts/eval_expert.py
echo "== 50 scripted demos"; scripts/record_demos.sh "$RUN_DIR/pickplace_scripted_50"
echo "== ACT CPU training"; scripts/train_cpu.sh "$RUN_DIR/pickplace_scripted_50" "$RUN_DIR/act_cpu_scripted50"
echo "== ACT eval";         scripts/eval_all.sh "$RUN_DIR/act_cpu_scripted50"
echo "== GIF";              scripts/record_gifs.sh "$RUN_DIR/act_cpu_scripted50"
echo "== tables";           sdp report
