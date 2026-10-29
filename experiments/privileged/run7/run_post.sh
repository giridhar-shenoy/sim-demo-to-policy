#!/usr/bin/env bash
# Run 7 after training: swap diagnostics on 2k/4k/6k, nominal dev eval (10500-10549) of
# 4k and 6k, then demo-seed (0-9) and shifted dev eval of the best one. Best = more
# nominal dev successes; a tie goes to 6k (rule fixed before any result was seen).
set -u
cd "$(dirname "$0")/../../.."
export SDP_MENAGERIE_DIR=${SDP_MENAGERIE_DIR:-$HOME/work/menagerie} MUJOCO_GL=egl
P=experiments/privileged/run7
CK=$P/train/act_cpu_v7_abs/checkpoints
run() {
  until [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -ge 5 ]; do sleep 30; done
  flock -o ~/work/.locks/heavy.lock flock -o ~/work/.locks/build.lock timeout 1200 "$@"
}
run .venv/bin/python $P/diagnose_swap.py --checkpoints $CK/002000 $CK/004000 $CK/006000 \
  > $P/logs/diagnose.out 2>&1; echo "diagnose exit $?"
for s in 004000 006000; do
  run .venv/bin/python $P/eval_run7.py --checkpoint $CK/$s --name v7_$s --seeds dev \
    > $P/logs/eval_${s}_dev_nominal.out 2>&1; echo "eval $s exit $?"
done
BEST=$(.venv/bin/python -c "
import json
k={s: json.load(open(f'$P/results/eval_v7_{s}_dev10500_nominal.json'))['conditions'][0]['successes'] for s in ('004000','006000')}
print('004000' if k['004000'] > k['006000'] else '006000')")
echo "best $BEST"
run .venv/bin/python $P/eval_run7.py --checkpoint $CK/$BEST --name v7_$BEST --seeds demo \
  > $P/logs/eval_${BEST}_demo_nominal.out 2>&1; echo "eval demo exit $?"
run .venv/bin/python $P/eval_run7.py --checkpoint $CK/$BEST --name v7_$BEST --seeds dev --distribution shifted \
  > $P/logs/eval_${BEST}_dev_shifted.out 2>&1; echo "eval shifted exit $?"
