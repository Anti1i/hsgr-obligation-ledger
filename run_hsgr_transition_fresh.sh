#!/usr/bin/env bash
set -euo pipefail

SCR="/mnt/scratch/z/$USER/dch-hsgr"
PY="$SCR/venv/bin/python"

"$PY" hsgr_transition_guide_fresh.py \
  --data data/musique_ans_val.jsonl \
  --edge-cases data/hsgr_edge_repair_cases_724849.jsonl \
  --focus-cases data/hsgr_focus_route_cases_725772.jsonl \
  --dev-features "$SCR/logs/structured_hidden_features_726228.pt" \
  --out-dir hsgr_transition_fresh
