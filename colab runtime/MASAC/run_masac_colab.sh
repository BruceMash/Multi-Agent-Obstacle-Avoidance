#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python scripts/train_masac_multi_agent_dmp.py \
  --device auto \
  --output-root "${MASAC_OUTPUT_ROOT:-artifacts/masac_colab}" \
  --total-steps "${MASAC_TOTAL_STEPS:-500000}" \
  --start-steps "${MASAC_START_STEPS:-5000}" \
  --save-interval "${MASAC_SAVE_INTERVAL:-50000}" \
  --log-interval "${MASAC_LOG_INTERVAL:-1000}" \
  --progress-interval "${MASAC_PROGRESS_INTERVAL:-100}" \
  "$@"
