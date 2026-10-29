#!/usr/bin/env bash
# Train ACT on both privileged datasets, one after the other, under the machine locks.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH
NM='policy.normalization_mapping={"VISUAL":"MEAN_STD","STATE":"MEAN_STD","ENV":"MEAN_STD","ACTION":"MEAN_STD"}'
STEPS=${STEPS:-6000}
for v in ${@:-delta abs}; do
  until [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -ge 5 ]; do sleep 30; done
  flock -o ~/work/.locks/heavy.lock flock -o ~/work/.locks/build.lock \
    timeout 1200 python -m sdp.train.train --config experiments/privileged/act_priv_$v.yaml \
      --dataset experiments/privileged/datasets/priv_${v}_50 \
      --output-dir experiments/privileged/train/act_priv_$v \
      --steps "$STEPS" --set "$NM" --set save_freq=2000 \
      > experiments/privileged/logs/train_$v.out 2>&1
  echo "$v exit $?"
done
