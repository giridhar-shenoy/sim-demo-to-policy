#!/usr/bin/env bash
# Full smallbc pipeline: record demos -> train -> eval on the dev seeds (10500-10549).
# Takes ~/work/.locks/heavy.lock for the whole run, so it waits for (and blocks) other
# heavy jobs. Override any setting through the environment, e.g.
#   EPISODES=600 STEPS=40000 TARGET=rel RUN_NAME=rel600 experiments/smallbc/run_full.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ "${1:-}" != "--locked" ]; then
  mkdir -p ~/work/.locks
  exec flock ~/work/.locks/heavy.lock "$0" --locked
fi

PY=.venv/bin/python
D=experiments/smallbc
EPISODES=${EPISODES:-1000}          # demo seeds 0..EPISODES-1 (must stay <= 1000)
NOISE_STD=${NOISE_STD:-0.3}         # DART noise, normalised action units
NOISE_FRAC=${NOISE_FRAC:-0.5}       # fraction of episodes noised
NOISE_DIMS=${NOISE_DIMS:-0,1,2,5}   # x, y, z, yaw (roll/pitch noise is never corrected by the expert)
WORKERS=${WORKERS:-4}
DATA=${DATA:-$D/data/main${EPISODES}}
STEPS=${STEPS:-50000}
TARGET=${TARGET:-abs}               # abs | rel
RUN_NAME=${RUN_NAME:-${TARGET}${EPISODES}}
RUN=$D/runs/$RUN_NAME
MAX_MIN=${MAX_MIN:-200}             # hard stop for training (minutes), checkpoints on exit
mkdir -p "$D/logs"

until [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -ge 6 ]; do
  echo "waiting for MemAvailable >= 6 GB"; sleep 60
done

if ! ls "$DATA"/record_*.json >/dev/null 2>&1; then
  echo "[$(date +%T)] recording $EPISODES demos -> $DATA"
  $PY $D/record.py --out "$DATA" --episodes "$EPISODES" --seed-start 0 \
    --noise-std "$NOISE_STD" --noise-frac "$NOISE_FRAC" --noise-dims "$NOISE_DIMS" \
    --workers "$WORKERS" --log-every 50 2>&1 | tee "$D/logs/record_$(basename "$DATA").log"
fi

echo "[$(date +%T)] training $RUN ($STEPS steps, target $TARGET)"
$PY $D/train.py --data "$DATA" --out "$RUN" --steps "$STEPS" --target "$TARGET" \
  --ckpt-every 5000 --val-every 1000 --max-minutes "$MAX_MIN" 2>&1 | tee "$D/logs/train_$RUN_NAME.log"

for c in "$RUN"/ckpt_$(printf %06d $((STEPS / 2))).pt "$RUN"/ckpt_$(printf %06d $((STEPS * 4 / 5))).pt "$RUN"/last.pt; do
  [ -f "$c" ] || continue
  echo "[$(date +%T)] eval $c"
  $PY $D/eval.py --ckpt "$c" --seed-start 10500 --n 50 2>&1 | tail -n 3 | tee -a "$D/logs/eval_$RUN_NAME.log"
done
echo "[$(date +%T)] done"
