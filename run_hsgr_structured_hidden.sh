#!/usr/bin/env bash
set -euo pipefail

SCR="/mnt/scratch/z/$USER/dch-hsgr"
PY="$SCR/venv/bin/python"

"$PY" hsgr_structured_hidden_verifier.py \
  --data data/musique_ans_val.jsonl \
  --sc-dir data/sc_cache \
  --out-dir hsgr_structured_hidden \
  --seed 20260816 \
  --bs 8
