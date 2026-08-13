#!/usr/bin/env bash
set -euo pipefail

"${PY:-python}" -X utf8 -m unittest -v test_dependency_route_swap_p2.py
"${PY:-python}" -X utf8 dependency_route_swap_p2.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --seed 20260817 \
  --calib-n 96 \
  --test-n 192 \
  --layers 21 \
  --batch-size 16 \
  --out-dir dependency_route_swap_p2
