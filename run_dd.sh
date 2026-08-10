#!/usr/bin/env bash
# Decomposition-level candidate domain: sample M decompositions per problem,
# answer nodes greedily, aggregate once per decomposition, and sample enough
# direct CoT for a budget-matched SC@k curve.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
NS=${NS:-2}

run_set () {
  local data=$1 out=$2 limit=$3
  echo "=== decomp domain: $data -> $out (limit $limit) ==="
  for s in $(seq 0 $((NS-1))); do
    CUDA_VISIBLE_DEVICES=$s $PY decomp_domain.py \
      --data "$data" --out-dir "$out" --shard "$s" --num-shards "$NS" \
      --limit "$limit" --model "$MODEL" > "logs/dd_${out}_s${s}.log" 2>&1 &
  done
  wait
  echo "=== done $out ==="
}

run_set data/gsm_deep_test.jsonl dd_deep   300
run_set data/math_l5.jsonl       dd_mathl5 134

$PY decomp_analyze.py --dir dd_deep,dd_mathl5 \
  --data data/gsm_deep_test.jsonl,data/math_l5.jsonl | tee logs/dd_analyze.txt
