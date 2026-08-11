#!/usr/bin/env bash
set -euo pipefail

SCR="/mnt/scratch/z/$USER/dch-hsgr"
PY="$SCR/venv/bin/python"

"$PY" hsgr_dual_route_guide_diagnostic.py \
  --data data/musique_ans_val.jsonl \
  --dev-features "$SCR/logs/structured_hidden_features_726228.pt" \
  --heldout-features "$SCR/logs/heldout_hidden_features_726310.pt" \
  --combined-features "$SCR/logs/dual_route_hidden_features_726354.pt" \
  --out-dir hsgr_dual_route_diagnostic
