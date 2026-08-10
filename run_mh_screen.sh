#!/usr/bin/env bash
# CPU: download MuSiQue + structural four-metric screen.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs data

PY=${PY:-python}

echo "=== fetch musique ==="
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
$PY -m pip install --quiet datasets pyarrow 2>/dev/null || true
$PY fetch_data.py --which musique

echo "=== screen ==="
$PY multihop_screen.py --data data/musique_ans_val.jsonl --name musique \
  | tee logs/mh_screen.txt
