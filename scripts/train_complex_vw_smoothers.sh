#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-0}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
APF_WARMUP_EPISODES="${APF_WARMUP_EPISODES:-2000}"

SMOOTHERS=(
  none
  current_kf
  rate_kf
  singer_kf
  ema
  second_order_lowpass
)

for SMOOTHER in "${SMOOTHERS[@]}"; do
  echo "===== Training complex env with smoother: ${SMOOTHER} ====="
  python -m train.complex_env \
    --smoother-type "${SMOOTHER}" \
    --use-kf 1 \
    --seed "${SEED}" \
    --total-steps "${TOTAL_STEPS}" \
    --apf-warmup-episodes "${APF_WARMUP_EPISODES}" \
    --show-progress 1
done
