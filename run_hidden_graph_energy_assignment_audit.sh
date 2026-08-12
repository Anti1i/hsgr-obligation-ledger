#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
SCREEN_DIR=${SCREEN_DIR:?set SCREEN_DIR to the completed structural screen}
OUT=${OUT:-hidden_graph_energy_guide}
BATCH_SIZE=${BATCH_SIZE:-16}

"$PY" hidden_graph_energy_guide.py \
  --stage assignment-audit \
  --data data/gsm_join_train.jsonl \
  --screen-dir "$SCREEN_DIR" \
  --out-dir "$OUT" \
  --model "$MODEL" \
  --batch-size "$BATCH_SIZE"
