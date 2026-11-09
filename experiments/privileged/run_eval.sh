#!/usr/bin/env bash
# Evaluate both checkpoints (50 eval seeds 10500-10549 + 10 demo seeds 0-9) and run the
# open-loop cube-pose sensitivity check, each command under the machine locks.
set -u
cd "$(dirname "$0")/../.."
export SDP_MENAGERIE_DIR=${SDP_MENAGERIE_DIR:-$HOME/work/menagerie}
P=experiments/privileged
run() {
  until [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -ge 5 ]; do sleep 30; done
  mkdir -p ~/work/.locks
  flock -o ~/work/.locks/heavy.lock flock -o ~/work/.locks/build.lock timeout 900 "$@"
}
for v in ${@:-delta abs}; do
  CK=$P/train/act_priv_$v/checkpoints/last
  run .venv/bin/python $P/eval_privileged.py --checkpoint $CK --mode $v --name act_priv_$v \
    > $P/logs/eval_$v.out 2>&1; echo "eval $v exit $?"
  run .venv/bin/python $P/sensitivity.py --checkpoint $CK --mode $v --name act_priv_$v \
    > $P/logs/sensitivity_$v.out 2>&1; echo "sensitivity $v exit $?"
done
