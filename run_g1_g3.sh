#!/usr/bin/env bash
# G1: extend root candidates to N=20 for a token-matched SC@k curve.
# G3: hidden-state probes on real reasoning traces (incl. mid-generation).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}

echo "=== G1 gen ==="
CUDA_VISIBLE_DEVICES=${G1_GPU:-0} $PY g1_root_extend.py --stage gen \
  --dir outputs --n-ext 20 --bs 10 --model "$MODEL" 2>&1 | tee logs/g1_math.log
CUDA_VISIBLE_DEVICES=${G1_GPU:-0} $PY g1_root_extend.py --stage gen \
  --dir outputs_gsm_test --n-ext 20 --bs 10 --model "$MODEL" 2>&1 | tee logs/g1_gsm.log

echo "=== G1 report ==="
$PY g1_root_extend.py --stage report --dir outputs          | tee logs/g1_report_math.txt
$PY g1_root_extend.py --stage report --dir outputs_gsm_test | tee logs/g1_report_gsm.txt

echo "=== G3 ==="
for st in gen extract; do
  CUDA_VISIBLE_DEVICES=${G3_GPU:-1} $PY g3_trace_probe.py --stage $st \
    --dir outputs --limit 120 --n 4 --model "$MODEL" 2>&1 | tee "logs/g3_${st}.log"
done
$PY g3_trace_probe.py --stage probe --dir outputs | tee logs/g3_probe.txt
