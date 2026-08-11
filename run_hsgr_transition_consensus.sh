#!/usr/bin/env bash
set -euo pipefail

SCR="/mnt/scratch/z/$USER/dch-hsgr"
PY="$SCR/venv/bin/python"

"$PY" hsgr_transition_consensus_diagnostic.py \
  --data data/musique_ans_val.jsonl \
  --dev-features "$SCR/logs/structured_hidden_features_726228.pt" \
  --heldout-features "$SCR/logs/transition_fresh_features_726389.pt" \
  --out-dir hsgr_transition_consensus \
  --threads 16

