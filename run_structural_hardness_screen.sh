#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
OUT=${OUT:-structural_hardness_screen}
BATCH_SIZE=${BATCH_SIZE:-16}

"$PY" structural_hardness_screen.py \
  --calibration-data data/gsm_join_train.jsonl \
  --test-data data/gsm_join_test.jsonl \
  --out-dir "$OUT" \
  --model "$MODEL" \
  --batch-size "$BATCH_SIZE"
