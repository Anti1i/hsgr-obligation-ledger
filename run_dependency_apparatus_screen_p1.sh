#!/usr/bin/env bash
set -euo pipefail

"${PY:-python}" -X utf8 -m unittest -v test_dependency_apparatus_screen_p1.py
"${PY:-python}" -X utf8 dependency_apparatus_screen_p1.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --seed 20260815 \
  --n-per-family 96 \
  --batch-size 32 \
  --output dependency_apparatus_screen_p1_report.json

