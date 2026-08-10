#!/usr/bin/env bash
# MuSiQue E0: gold-decomposition hops vs open-book SC@1 (decomposition tax + headroom).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs data

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
LIMIT=${LIMIT:-200}
REUSE=${REUSE_SC:-}

echo "=== ensure musique jsonl ==="
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
$PY -m pip install --quiet datasets pyarrow 2>/dev/null || true
$PY fetch_data.py --which musique

echo "=== mh E0 limit=$LIMIT ==="
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE_KEEP:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_KEEP:-1}
$PY mh_e0.py --data data/musique_ans_val.jsonl --out-dir mh_e0 \
  --limit "$LIMIT" --model "$MODEL" \
  ${REUSE:+--reuse-sc "$REUSE"} \
  | tee logs/mh_e0.txt
