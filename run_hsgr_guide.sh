#!/usr/bin/env bash
set -uo pipefail

PY="${PY:-python}"
LIMIT="${LIMIT:-200}"

"$PY" hsgr_dependency_guide.py \
  --data data/musique_ans_val.jsonl \
  --out-dir hsgr_guide \
  --limit "$LIMIT" \
  --calib 40 \
  --seed 20260810 \
  --max-context 4096 \
  --max-new 128 \
  --bs-hidden 8 \
  --bs-generate 8
