#!/usr/bin/env bash
set -euo pipefail

SCR="/mnt/scratch/z/$USER/dch-hsgr"
PY="$SCR/venv/bin/python"

"$PY" hsgr_listwise_guide_verifier.py \
  --data data/musique_ans_val.jsonl \
  --dev-features "$SCR/logs/structured_hidden_features_726228.pt" \
  --first-features "$SCR/logs/dual_route_hidden_features_726354.pt" \
  --second-features "$SCR/logs/transition_fresh_features_726389.pt" \
  --out-dir hsgr_listwise_guide_verifier \
  --threads 16

