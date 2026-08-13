#!/usr/bin/env bash
set -euo pipefail

"${PY:-python}" -X utf8 -m unittest -v test_dependency_patch_p1.py
"${PY:-python}" -X utf8 dependency_patch_p1.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --seed 20260816 \
  --calib-n 96 \
  --test-n 192 \
  --layers 7 14 21 \
  --batch-size 16 \
  --out-dir dependency_patch_p1

