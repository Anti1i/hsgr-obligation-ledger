#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
OUT=${OUT:-hsgr_error_provenance_v1_1}
BATCH_SIZE=${BATCH_SIZE:-8}

"$PY" hsgr_error_provenance_ceiling_v1_1.py \
  --data data/gsm_chain_test.jsonl \
  --out-dir "$OUT" \
  --model "$MODEL" \
  --batch-size "$BATCH_SIZE"

