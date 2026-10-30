#!/usr/bin/env bash
# Regenerate every number and the GIF in README.md and docs/results.md from scratch.
# Needs: the repo venv active, SDP_MENAGERIE_DIR set (scripts/fetch_menagerie.sh), MUJOCO_GL=egl.
# Wall time on a 4-core laptop CPU: about 6 h, of which training is ~4.5 h (ACT runs 1-6
# ~2.5 h, smallbc ~2.2 h; ACT runs 7-8 are trained by experiments/privileged/ and not here).
# The dataset and checkpoint go to a fresh directory, so nothing existing is overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${RUN_DIR:-outputs/regen-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$RUN_DIR"
echo "== run dir: $RUN_DIR"

echo "== env throughput";   python scripts/bench_env_speed.py
echo "== scripted expert";  python scripts/eval_expert.py
echo "== expert, 128 px";   python scripts/eval_expert.py --image-size 128 --out results/expert_check_128px
echo "== 50 scripted demos"; scripts/record_demos.sh "$RUN_DIR/pickplace_scripted_50"
echo "== same, 128 px";      scripts/record_demos.sh "$RUN_DIR/pickplace_scripted_50_128px" 128
echo "== ACT run 1";       scripts/train_cpu.sh "$RUN_DIR/pickplace_scripted_50" "$RUN_DIR/act_cpu_scripted50" sdp/train/configs/act_cpu.yaml
echo "== ACT run 2";       scripts/train_cpu.sh "$RUN_DIR/pickplace_scripted_50" "$RUN_DIR/act_cpu_v2_scripted50" sdp/train/configs/act_cpu_v2.yaml
echo "== ACT run 3";       scripts/train_cpu.sh "$RUN_DIR/pickplace_scripted_50" "$RUN_DIR/act_cpu_v3_scripted50" sdp/train/configs/act_cpu_v3.yaml
echo "== ACT run 4";       scripts/train_cpu.sh "$RUN_DIR/pickplace_scripted_50" "$RUN_DIR/act_cpu_v4_scripted50" sdp/train/configs/act_cpu_v4.yaml
echo "== ACT run 5";       scripts/train_cpu.sh "$RUN_DIR/pickplace_scripted_50_128px" "$RUN_DIR/act_cpu_v5_scripted50_128px" sdp/train/configs/act_cpu_v5.yaml
echo "== ACT run 6";       scripts/train_cpu.sh "$RUN_DIR/pickplace_scripted_50" "$RUN_DIR/act_cpu_v6_scripted50" sdp/train/configs/act_cpu_v6.yaml
echo "== open-loop check"; for r in act_cpu_scripted50 act_cpu_v2_scripted50 act_cpu_v3_scripted50 act_cpu_v4_scripted50; do
  python scripts/diagnose_openloop.py --checkpoint "$RUN_DIR/$r" --dataset "$RUN_DIR/pickplace_scripted_50"; done
python scripts/diagnose_openloop.py --checkpoint "$RUN_DIR/act_cpu_v5_scripted50_128px" --dataset "$RUN_DIR/pickplace_scripted_50_128px"
for step in 001000 002000; do   # run 6: intermediate checkpoints too
  python scripts/diagnose_openloop.py --checkpoint "$RUN_DIR/act_cpu_v6_scripted50/checkpoints/$step" --dataset "$RUN_DIR/pickplace_scripted_50"; done
python scripts/diagnose_openloop.py --checkpoint "$RUN_DIR/act_cpu_v6_scripted50" --dataset "$RUN_DIR/pickplace_scripted_50"
echo "== smallbc demos";  python -m sdp.policies.smallbc.record --out "$RUN_DIR/smallbc_main1000" \
  --episodes 1000 --seed-start 0 --noise-std 0.3 --noise-frac 0.5 --noise-dims 0,1,2,5 \
  --workers 4 --log-every 50
echo "== smallbc train";  python -m sdp.policies.smallbc.train --data "$RUN_DIR/smallbc_main1000" \
  --out "$RUN_DIR/smallbc_abs1000" --steps 50000 --target abs --ckpt-every 5000 \
  --val-every 1000 --max-minutes 190
for c in ckpt_025000 ckpt_040000 last; do   # dev-seed checkpoint choice (25k was chosen)
  sdp eval --checkpoint "$RUN_DIR/smallbc_abs1000/$c.pt" --condition "smallbc_$c" --trials 50 \
    --seed-start 500 --distributions nominal --out "$RUN_DIR/eval_smallbc_${c}_dev10500"; done
echo "== ACT eval";        scripts/eval_all.sh "$RUN_DIR/act_cpu_scripted50" "$RUN_DIR/act_cpu_v2_scripted50" "$RUN_DIR/act_cpu_v3_scripted50" "$RUN_DIR/act_cpu_v4_scripted50" "$RUN_DIR/act_cpu_v5_scripted50_128px" "$RUN_DIR/act_cpu_v6_scripted50" \
  experiments/privileged/run7/train/act_cpu_v7_abs/checkpoints/006000 \
  experiments/privileged/run8/train/act_cpu_v8_abs_dart/checkpoints/020000 \
  "$RUN_DIR/smallbc_abs1000/ckpt_025000.pt"
python scripts/smallbc_results_from_run.py --run "$RUN_DIR/smallbc_abs1000" --checkpoint-step 25000 \
  --data "$RUN_DIR/smallbc_main1000"
echo "== GIF";             scripts/record_gifs.sh "$RUN_DIR/smallbc_abs1000/ckpt_025000.pt"
echo "== tables";           sdp report
