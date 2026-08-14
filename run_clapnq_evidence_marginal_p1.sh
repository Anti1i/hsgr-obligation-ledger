#!/usr/bin/env bash
set -euo pipefail

DATA="${CLAPNQ_DATA:-data/longform_cache/clapnq_dev_answerable.jsonl}"
if [ ! -s "$DATA" ]; then
  mkdir -p "$(dirname "$DATA")"
  curl --fail --location --retry 3 \
    --output "$DATA" \
    https://raw.githubusercontent.com/primeqa/clapnq/main/annotated_data/dev/clapnq_dev_answerable.jsonl
fi

"${PY:-python}" -X utf8 -m unittest -v \
  test_longform_structure_audit.py \
  test_clapnq_evidence_marginal_p1.py

"${PY:-python}" -X utf8 clapnq_evidence_marginal_p1.py \
  --data "$DATA" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --seed 20260814 \
  --calib-n 96 \
  --test-n 96 \
  --layers 13 20 27 \
  --hidden-batch-size 2 \
  --score-batch-size 4 \
  --out-dir clapnq_evidence_marginal_p1

