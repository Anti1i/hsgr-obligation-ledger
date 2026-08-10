#!/usr/bin/env bash
# MuSiQue open-book SC ceiling (after structural screen PASS/WARN).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs data

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
NS=${NS:-1}
LIMIT=${LIMIT:-200}

echo "=== ensure musique with support paragraphs ==="
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
$PY -m pip install --quiet datasets pyarrow 2>/dev/null || true
$PY fetch_data.py --which musique --force

echo "=== mh ceiling limit=$LIMIT ==="
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE_KEEP:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_KEEP:-1}
for s in $(seq 0 $((NS-1))); do
  CUDA_VISIBLE_DEVICES=$s $PY mh_ceiling.py \
    --data data/musique_ans_val.jsonl --out-dir mh_ceil \
    --shard "$s" --num-shards "$NS" --limit "$LIMIT" --n-samples 8 \
    --model "$MODEL" > "logs/mh_ceil_s${s}.log" 2>&1 &
done
wait
$PY mh_ceiling.py --data data/musique_ans_val.jsonl --out-dir mh_ceil --analyze-only \
  | tee logs/mh_ceil_analyze.txt
