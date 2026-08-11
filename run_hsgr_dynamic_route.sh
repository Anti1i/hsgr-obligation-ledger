#!/usr/bin/env bash
set -uo pipefail

PY="${PY:-python}"

"$PY" hsgr_dynamic_route_read.py \
  --data data/musique_ans_val.jsonl \
  --exclude-cases data/hsgr_edge_repair_cases_724849.jsonl \
  --exclude-cases data/hsgr_focus_route_cases_725772.jsonl \
  --out-dir hsgr_dynamic_route \
  --original-exclude-limit 200 \
  --calib 80 \
  --test 320 \
  --seed 20260814 \
  --max-new 96 \
  --bs-hidden 4 \
  --bs-generate 8 \
  --permutation-samples 50000
