#!/usr/bin/env bash
set -uo pipefail

PY="${PY:-python}"

"$PY" hsgr_commit_guide.py \
  --data data/musique_ans_val.jsonl \
  --edge-cases data/hsgr_edge_repair_cases_724849.jsonl \
  --focus-cases data/hsgr_focus_route_cases_725772.jsonl \
  --out-dir hsgr_commit_guide \
  --seed 20260815 \
  --max-new 96 \
  --bs-hidden 2 \
  --bs-generate 8
