#!/usr/bin/env bash
# S1: depth-2 pilot on the harder single-narrative sets, then the gate.
# (gsm_chain is an explicit 2-hop chain whose siblings are NOT independent, so it
#  goes through the depth-3 pipeline in run_s2.sh instead.)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
NS=${NS:-4}

run_set () {
  local data=$1 out=$2 limit=$3
  echo "=== S1 pilot: $data -> $out (limit $limit) ==="
  for s in $(seq 0 $((NS-1))); do
    CUDA_VISIBLE_DEVICES=$s $PY pilot.py \
      --data "$data" --out-dir "$out" --shard "$s" --num-shards "$NS" \
      --limit "$limit" --model "$MODEL" > "logs/s1_${out}_s${s}.log" 2>&1 &
  done
  wait
  echo "=== done $out ==="
}

run_set data/gsm_deep_test.jsonl outputs_deep2 300
run_set data/math_l5.jsonl       outputs_mathl5 134

$PY s1_gate.py --dirs outputs_deep2,outputs_mathl5 --gate 0.10 | tee logs/s1_gate.txt
