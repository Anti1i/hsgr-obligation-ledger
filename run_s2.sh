#!/usr/bin/env bash
# S2: depth-3 hierarchy + gate + bidirectional message-passing retest.
# Default dataset is the compositional 2-hop chain, whose dependency structure is
# exactly what the depth-3 pipeline is for. Set DATA/OUT to also run gsm_deep.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
DATA=${DATA:-data/gsm_chain_test.jsonl}
OUT=${OUT:-outputs_chain3}
LIMIT=${LIMIT:-200}
GPU=${GPU:-0}

for stage in tree leaf mid root rootcot; do
  echo "=== S2 stage: $stage ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY s2_deep_hierarchy.py \
    --stage "$stage" --data "$DATA" --out-dir "$OUT" --limit "$LIMIT" \
    --model "$MODEL" 2>&1 | tee "logs/s2_${OUT}_${stage}.log"
done

echo "=== S2 gate (CPU) ==="
$PY s2_deep_hierarchy.py --stage gate --out-dir "$OUT" --gate 0.10 \
  | tee "logs/s2_${OUT}_gate.txt"

echo "=== S2 bp (CPU) ==="
$PY s2_deep_hierarchy.py --stage bp --out-dir "$OUT" --rounds 4 \
  | tee "logs/s2_${OUT}_bp.txt"
