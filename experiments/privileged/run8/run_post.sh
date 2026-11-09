#!/usr/bin/env bash
# Run 8 after training: image-swap diagnostics on 8k/12k/16k/20k (on run 7's clean
# 50-demo dataset, so the numbers compare 1:1 with run 7), nominal dev eval
# (10500-10549) of 12k/16k/20k, then shifted dev + demo-seed (0-9) eval of the best.
# Best = most nominal dev successes; ties go to the later checkpoint (fixed in advance).
set -u
cd "$(dirname "$0")/../../.."
export SDP_MENAGERIE_DIR=${SDP_MENAGERIE_DIR:-$HOME/work/menagerie} MUJOCO_GL=egl
P=experiments/privileged/run8
R7=experiments/privileged/run7
CK=$P/train/act_cpu_v8_abs_dart/checkpoints
run() {
  until [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -ge 5 ]; do sleep 30; done
  mkdir -p ~/work/.locks
  flock -o ~/work/.locks/heavy.lock flock -o ~/work/.locks/build.lock timeout 1200 "$@"
}
run .venv/bin/python $R7/diagnose_swap.py --checkpoints $CK/008000 $CK/012000 $CK/016000 $CK/020000 \
  --dataset $R7/datasets/pickplace_abs_50 --out $P/results/swap_sensitivity.json \
  > $P/logs/diagnose.out 2>&1; echo "diagnose exit $?"
for s in 012000 016000 020000; do
  run .venv/bin/python $R7/eval_run7.py --checkpoint $CK/$s --name v8_$s --seeds dev \
    --results-dir $P/results > $P/logs/eval_${s}_dev_nominal.out 2>&1; echo "eval $s exit $?"
done
BEST=$(.venv/bin/python -c "
import json
k={s: json.load(open(f'$P/results/eval_v8_{s}_dev10500_nominal.json'))['conditions'][0]['successes'] for s in ('012000','016000','020000')}
m=max(k.values()); print([s for s in ('012000','016000','020000') if k[s]==m][-1])")
echo "best $BEST"
run .venv/bin/python $R7/eval_run7.py --checkpoint $CK/$BEST --name v8_$BEST --seeds dev --distribution shifted \
  --results-dir $P/results > $P/logs/eval_${BEST}_dev_shifted.out 2>&1; echo "eval shifted exit $?"
run .venv/bin/python $R7/eval_run7.py --checkpoint $CK/$BEST --name v8_$BEST --seeds demo \
  --results-dir $P/results > $P/logs/eval_${BEST}_demo_nominal.out 2>&1; echo "eval demo exit $?"
