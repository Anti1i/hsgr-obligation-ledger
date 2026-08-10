#!/usr/bin/env bash
# E0 (socratic): measure decomposition tax under oracle hierarchy.
# Goals come from GSM8K-socratic subquestions (text before **), not prose scrub.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs data

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
NS=${NS:-1}
LIMIT=${LIMIT:-300}
REUSE=${REUSE_ROOTEXT:-}
OUTDIR=${OUTDIR:-e0_soc}

echo "=== fetch gsm8k socratic (skip if present) ==="
# Allow hub/network for this small file even if model weights are offline.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
$PY fetch_data.py --which socratic

echo "=== build oracle DAG (socratic goals) ==="
$PY oracle_hierarchy.py --data data/gsm_deep_test.jsonl \
  --out data/gsm_oracle_soc.jsonl --min-steps 3 \
  --goal-source data/gsm8k_socratic_test.jsonl --goal-mode socratic

echo "=== E0 run -> $OUTDIR (limit $LIMIT) ==="
# Restore offline for model load if weights are cached.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE_KEEP:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_KEEP:-1}
for s in $(seq 0 $((NS-1))); do
  CUDA_VISIBLE_DEVICES=$s $PY oracle_e0.py \
    --data data/gsm_oracle_soc.jsonl --out-dir "$OUTDIR" \
    --shard "$s" --num-shards "$NS" --limit "$LIMIT" --model "$MODEL" \
    ${REUSE:+--reuse-rootext "$REUSE"} \
    > "logs/${OUTDIR}_s${s}.log" 2>&1 &
done
wait

$PY oracle_analyze.py --dir "$OUTDIR" --data data/gsm_oracle_soc.jsonl \
  | tee "logs/${OUTDIR}_analyze.txt"
