#!/usr/bin/env bash
# E0: oracle-hierarchy ceiling vs SC@k (budget-matched).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs data

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
NS=${NS:-1}
LIMIT=${LIMIT:-300}
# Optional: path to a prior dd_* dir whose rootext.s*.jsonl we can reuse
REUSE=${REUSE_ROOTEXT:-}

echo "=== build oracle DAG from gsm_deep (NL goals from raw gsm8k) ==="
$PY oracle_hierarchy.py --data data/gsm_deep_test.jsonl \
  --out data/gsm_oracle_deep.jsonl --min-steps 3 \
  --goal-source data/gsm8k_test.jsonl

echo "=== E0 run (limit $LIMIT) ==="
for s in $(seq 0 $((NS-1))); do
  CUDA_VISIBLE_DEVICES=$s $PY oracle_e0.py \
    --data data/gsm_oracle_deep.jsonl --out-dir e0_deep \
    --shard "$s" --num-shards "$NS" --limit "$LIMIT" --model "$MODEL" \
    ${REUSE:+--reuse-rootext "$REUSE"} \
    > "logs/e0_deep_s${s}.log" 2>&1 &
done
wait

$PY oracle_analyze.py --dir e0_deep --data data/gsm_oracle_deep.jsonl \
  | tee logs/e0_analyze.txt
