#!/usr/bin/env bash
set -euo pipefail

SCR="/mnt/scratch/z/$USER/dch-hsgr"
PY="$SCR/venv/bin/python"
OUT="${HSGR_RC_OUT:-hsgr_route_counterfactual}"

"$PY" hsgr_route_counterfactual_features.py \
  --data data/musique_ans_val.jsonl \
  --source-features \
    "$SCR/logs/structured_hidden_features_726228.pt" \
    "$SCR/logs/dual_route_hidden_features_726354.pt" \
    "$SCR/logs/transition_fresh_features_726389.pt" \
  --out-dir "$OUT" \
  --bs-planner 8 \
  --bs-executor 16 \
  --bs-hidden 4 \
  --max-context 8192

"$PY" hsgr_route_counterfactual_eval.py \
  --features "$OUT/route_counterfactual_features.pt" \
  --out-dir "$OUT/eval" \
  --threads 16
