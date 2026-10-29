#!/usr/bin/env bash
# Regenerate every number and the GIF in README.md and docs/results.md from scratch.
# Needs: the repo venv active, SDP_MENAGERIE_DIR set (scripts/fetch_menagerie.sh), MUJOCO_GL=egl.
# Wall time on a 4-core laptop CPU: about 2.5 h, of which training (four runs) is ~100 min.
# The dataset and checkpoint go to a fresh directory, so nothing existing is overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${RUN_DIR:-outputs/regen-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$RUN_DIR"
echo "== run dir: $RUN_DIR"

echo "== env throughput";   python scripts/bench_env_speed.py
echo "== scripted expert";  python scripts/eval_expert.py
echo "== 50 scripted demos"; scripts/record_demos.sh "$RUN_DIR/pickplace_scripted_50"
echo "== ACT run 1";       scripts/train_cpu.sh "$RUN_DIR/pickplace_scripted_50" "$RUN_DIR/act_cpu_scripted50" sdp/train/configs/act_cpu.yaml
echo "== ACT run 2";       scripts/train_cpu.sh "$RUN_DIR/pickplace_scripted_50" "$RUN_DIR/act_cpu_v2_scripted50" sdp/train/configs/act_cpu_v2.yaml
echo "== ACT run 3";       scripts/train_cpu.sh "$RUN_DIR/pickplace_scripted_50" "$RUN_DIR/act_cpu_v3_scripted50" sdp/train/configs/act_cpu_v3.yaml
echo "== ACT run 4";       scripts/train_cpu.sh "$RUN_DIR/pickplace_scripted_50" "$RUN_DIR/act_cpu_v4_scripted50" sdp/train/configs/act_cpu_v4.yaml
echo "== open-loop check"; for r in act_cpu_scripted50 act_cpu_v2_scripted50 act_cpu_v3_scripted50 act_cpu_v4_scripted50; do
  python scripts/diagnose_openloop.py --checkpoint "$RUN_DIR/$r" --dataset "$RUN_DIR/pickplace_scripted_50"; done
echo "== ACT eval";        scripts/eval_all.sh "$RUN_DIR/act_cpu_scripted50" "$RUN_DIR/act_cpu_v2_scripted50" "$RUN_DIR/act_cpu_v3_scripted50" "$RUN_DIR/act_cpu_v4_scripted50"
echo "== GIF";             scripts/record_gifs.sh "$RUN_DIR/act_cpu_v3_scripted50"
echo "== tables";           sdp report
