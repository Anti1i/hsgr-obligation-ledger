#!/usr/bin/env bash
set -uo pipefail

PY="${PY:-python}"
NEW_LIMIT="${NEW_LIMIT:-400}"

"$PY" hsgr_focus_route_ceiling.py \
  --data data/musique_ans_val.jsonl \
  --prior-cases data/hsgr_edge_repair_cases_724849.jsonl \
  --out-dir hsgr_focus_route \
  --original-exclude-limit 200 \
  --prior-exclude-limit 400 \
  --prior-seed 20260811 \
  --new-limit "$NEW_LIMIT" \
  --seed 20260812 \
  --max-new 128 \
  --bs 8 \
  --permutation-samples 50000
