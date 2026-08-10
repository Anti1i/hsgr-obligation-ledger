#!/usr/bin/env bash
set -uo pipefail

PY="${PY:-python}"
NEW_LIMIT="${NEW_LIMIT:-400}"

"$PY" hsgr_edge_repair_ceiling.py \
  --data data/musique_ans_val.jsonl \
  --out-dir hsgr_edge_repair \
  --exclude-limit 200 \
  --new-limit "$NEW_LIMIT" \
  --seed 20260811 \
  --max-new 128 \
  --bs 8
