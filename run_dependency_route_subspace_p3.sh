#!/usr/bin/env bash
set -euo pipefail

"${PY:-python}" -X utf8 -m unittest -v test_dependency_route_subspace_p3.py
"${PY:-python}" -X utf8 dependency_route_subspace_p3.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --seed 20260818 \
  --calib-n 96 \
  --test-n 192 \
  --layers 19 20 21 \
  --batch-size 16 \
  --out-dir dependency_route_subspace_p3
