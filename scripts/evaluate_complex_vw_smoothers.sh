#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-0}"
EPISODES="${EPISODES:-20}"

SMOOTHERS=(
  none
  current_kf
  rate_kf
  singer_kf
  ema
  second_order_lowpass
)

for SMOOTHER in "${SMOOTHERS[@]}"; do
  echo "===== Evaluating complex env with smoother: ${SMOOTHER} ====="
  python -m evaluate.complex_env \
    --smoother-type "${SMOOTHER}" \
    --use-kf 1 \
    --seed "${SEED}" \
    --episodes "${EPISODES}" \
    --deterministic 1
done
